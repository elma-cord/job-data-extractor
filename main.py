import csv
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

import pandas as pd
from bs4 import BeautifulSoup

from classifiers import (
    ai_check_relevance,
    ai_extract_core_fields,
    ai_generate_additional_skills,
    ai_generate_skills_full,
    ai_map_job_titles_only,
)
from fetch_extract import extract_best_content, fetch_html
from formatters import exact_match_skills_in_order
from validators import (
    detect_job_type_rule_based,
    detect_remote_days_rule_based,
    detect_remote_preferences_rule_based,
    detect_visa_rule_based,
    fallback_seniorities,
    has_explicit_remote_days_evidence,
    is_tp_by_rules,
    location_value_has_evidence,
    normalize_category_for_skills,
    normalize_job_title_from_list,
    normalize_location_rule_based,
    normalize_quotes,
    normalize_seniority_list,
    parse_explicit_salary,
    postprocess_job_titles,
    refine_seniorities_rule_based,
    snap_salary_value,
)

INPUT_JOBS_FILE = "jobs_input.csv"
OUTPUT_CSV_FILE = "results.csv"

LOCATIONS_CSV = "predefined_locations.csv"
SALARIES_CSV = "predefined_salaries.csv"
TP_SKILLS_CSV = "predefined_tp_skills.csv"
NONTP_SKILLS_CSV = "predefined_nontp_skills.csv"
JOB_TITLES_CSV = "predefined_job_titles.csv"

MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
MIN_EXACT_SKILLS_TO_SKIP_AI = int(os.getenv("MIN_EXACT_SKILLS_TO_SKIP_AI", "4"))
ENABLE_URL_FALLBACK = os.getenv("ENABLE_URL_FALLBACK", "1").strip().lower() in {"1", "true", "yes", "y"}

TITLE_STRONG_MATCH_RATIO = float(os.getenv("TITLE_STRONG_MATCH_RATIO", "0.84"))
TITLE_TOKEN_MATCH_RATIO = float(os.getenv("TITLE_TOKEN_MATCH_RATIO", "0.67"))


@dataclass
class JobInput:
    row_id: str = ""
    company_name: str = ""
    job_title: str = ""
    job_url: str = ""
    job_description: str = ""


@dataclass
class JobResult:
    row_id: str = ""
    company_name: str = ""
    input_job_title: str = ""
    job_url: str = ""

    role_relevance: str = ""
    role_relevance_reason: str = ""
    job_category: str = ""
    job_location: str = ""
    remote_preferences: str = ""
    remote_days: str = ""
    salary_min: str = ""
    salary_max: str = ""
    salary_currency: str = ""
    salary_period: str = ""
    visa_sponsorship: str = ""
    job_type: str = ""
    notes: str = ""

    job_title_tag_1: str = ""
    job_title_tag_2: str = ""
    job_title_tag_3: str = ""

    seniority_1: str = ""
    seniority_2: str = ""
    seniority_3: str = ""

    skill_1: str = ""
    skill_2: str = ""
    skill_3: str = ""
    skill_4: str = ""
    skill_5: str = ""
    skill_6: str = ""
    skill_7: str = ""
    skill_8: str = ""
    skill_9: str = ""
    skill_10: str = ""


def clean_whitespace(text: str) -> str:
    text = text or ""
    text = text.replace("\xa0", " ")
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_lines(text: str) -> List[str]:
    return [clean_whitespace(x) for x in (text or "").splitlines() if clean_whitespace(x)]


def dedupe_keep_order(values: List[str]) -> List[str]:
    seen = set()
    out = []
    for v in values:
        key = v.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(v)
    return out


def strip_html(text: str) -> str:
    return clean_whitespace(BeautifulSoup(text or "", "lxml").get_text("\n", strip=True))


def looks_like_html(text: str) -> bool:
    return bool(re.search(r"<[a-z][\s\S]*?>", text or "", flags=re.I))


def normalize_remote_preferences_value(value: str) -> str:
    value = clean_whitespace(value)
    if not value:
        return ""

    low = normalize_quotes(value).lower()
    if low in {"not specified", "not_specified"}:
        return "not specified"

    found = []
    for pref in ["onsite", "hybrid", "remote"]:
        if re.search(rf"(?<![a-z]){re.escape(pref)}(?![a-z])", low):
            found.append(pref)

    if not found:
        return ""

    return ", ".join(found)


def load_text_list(filepath: str) -> List[str]:
    if not os.path.exists(filepath):
        print(f"[WARN] Missing file: {filepath}")
        return []
    df = pd.read_csv(filepath)
    col = df.columns[0]
    return (
        df[col]
        .dropna()
        .astype(str)
        .map(lambda x: x.strip())
        .loc[lambda s: s != ""]
        .tolist()
    )


def load_salary_list(filepath: str) -> List[int]:
    if not os.path.exists(filepath):
        print(f"[WARN] Missing file: {filepath}")
        return []
    df = pd.read_csv(filepath)
    col = df.columns[0]
    vals = []
    for x in df[col].dropna().tolist():
        try:
            vals.append(int(float(str(x).replace(",", "").strip())))
        except Exception:
            pass
    return sorted(set(vals))


def normalize_column_name(name: str) -> str:
    name = (name or "").strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def load_jobs_input(filepath: str) -> List[JobInput]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing input file: {filepath}")

    df = pd.read_csv(filepath, dtype=str, keep_default_na=False)
    original_columns = list(df.columns)
    normalized_map = {normalize_column_name(c): c for c in original_columns}

    required = ["company_name", "job_title", "job_url", "job_description"]
    missing = [c for c in required if c not in normalized_map]
    if missing:
        raise ValueError(
            f"Missing required columns in {filepath}: {missing}. "
            f"Expected at least: company_name, job_title, job_url, job_description"
        )

    row_id_col = ""
    for candidate in ["row_id", "id", "row", "index"]:
        if candidate in normalized_map:
            row_id_col = normalized_map[candidate]
            break

    jobs: List[JobInput] = []
    for idx, row in df.iterrows():
        jobs.append(
            JobInput(
                row_id=clean_whitespace(str(row.get(row_id_col, ""))) if row_id_col else str(idx + 1),
                company_name=clean_whitespace(str(row[normalized_map["company_name"]])),
                job_title=clean_whitespace(str(row[normalized_map["job_title"]])),
                job_url=clean_whitespace(str(row[normalized_map["job_url"]])),
                job_description=str(row[normalized_map["job_description"]]).strip(),
            )
        )

    return jobs


def clean_job_title(raw_title: str) -> str:
    if not raw_title:
        return ""

    title = clean_whitespace(normalize_quotes(raw_title))

    stop_patterns = [
        r"\s+\|\s+.*$",
        r"\s+·\s+.*$",
        r"\s+@\s+.*$",
        r"\s+\-\s+(remote|hybrid|onsite|on-site|on site|contract|full[- ]time|part[- ]time).*$",
        r"\s*,\s*(remote|hybrid|onsite|on-site|on site).*$",
        r"\s+\-\s+[A-Z][a-z]+,\s*[A-Z]{2,}.*$",
    ]
    for pat in stop_patterns:
        title = re.sub(pat, "", title, flags=re.I)

    return clean_whitespace(title.strip(" -|,·"))


def infer_title_from_description(description_text: str) -> str:
    for line in split_lines(description_text)[:12]:
        low = line.lower()
        if len(line) < 3 or len(line) > 160:
            continue
        if any(
            marker in low
            for marker in [
                "about the role",
                "job description",
                "overview",
                "responsibilities",
                "requirements",
                "qualifications",
                "about us",
                "application",
            ]
        ):
            continue
        if re.search(r"\b(salary|location|apply|employment type|department)\b", low):
            continue
        return clean_job_title(line)
    return ""


def prepare_description_text(raw_description: str) -> str:
    raw_description = raw_description or ""
    if looks_like_html(raw_description):
        text = strip_html(raw_description)
    else:
        text = clean_whitespace(raw_description)

    text = normalize_quotes(text)
    text = clean_whitespace(text)
    return text


def title_tokens(text: str) -> List[str]:
    text = normalize_quotes(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    toks = [t for t in text.split() if t]
    stop = {
        "the", "and", "of", "for", "to", "in", "on", "with", "a", "an",
        "senior", "junior", "lead", "principal", "staff", "mid", "level"
    }
    return [t for t in toks if t not in stop]


def title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_quotes(a).lower(), normalize_quotes(b).lower()).ratio()


def best_allowed_title_match(job_title: str, allowed_job_titles: List[str]) -> Tuple[str, float, float]:
    clean_title = clean_whitespace(job_title)
    if not clean_title or not allowed_job_titles:
        return "", 0.0, 0.0

    exact = normalize_job_title_from_list(clean_title, allowed_job_titles)
    if exact:
        return exact, 1.0, 1.0

    jtoks = set(title_tokens(clean_title))
    best_title = ""
    best_ratio = 0.0
    best_token_ratio = 0.0

    for allowed in allowed_job_titles:
        ratio = title_similarity(clean_title, allowed)

        atoks = set(title_tokens(allowed))
        overlap = len(jtoks & atoks)
        token_ratio = 0.0
        if jtoks and atoks:
            token_ratio = overlap / max(1, min(len(jtoks), len(atoks)))

        compact_job = re.sub(r"[^a-z0-9]", "", clean_title.lower())
        compact_allowed = re.sub(r"[^a-z0-9]", "", allowed.lower())
        if compact_job == compact_allowed:
            ratio = 1.0
            token_ratio = 1.0

        if ratio > best_ratio or (abs(ratio - best_ratio) < 0.0001 and token_ratio > best_token_ratio):
            best_title = allowed
            best_ratio = ratio
            best_token_ratio = token_ratio

    return best_title, best_ratio, best_token_ratio


def is_strong_allowed_title_match(job_title: str, allowed_job_titles: List[str]) -> Tuple[bool, str, str]:
    best_title, ratio, token_ratio = best_allowed_title_match(job_title, allowed_job_titles)

    if not best_title:
        return False, "", ""

    if ratio >= 0.999:
        return True, best_title, f"Exact allowed title match: {best_title}"

    if ratio >= TITLE_STRONG_MATCH_RATIO:
        return True, best_title, f"Strong title match to allowed title: {best_title}"

    if token_ratio >= TITLE_TOKEN_MATCH_RATIO and ratio >= 0.58:
        return True, best_title, f"Token-based title match to allowed title: {best_title}"

    return False, best_title, ""


def derive_category_from_title_or_text(matched_title: str, clean_title: str, description_text: str) -> str:
    combined = "\n".join([matched_title or "", clean_title or "", description_text or ""])
    return "T&P" if is_tp_by_rules(matched_title or clean_title, combined) else "NonT&P"


def blank_result_with_relevance(job: JobInput, role_relevance: str, reason: str, notes: str = "") -> JobResult:
    return JobResult(
        row_id=job.row_id,
        company_name=job.company_name,
        input_job_title=job.job_title,
        job_url=job.job_url,
        role_relevance=role_relevance,
        role_relevance_reason=reason,
        notes=notes or "description only",
    )


def extract_location_and_remote_from_url(url: str, allowed_locations: List[str]) -> Dict[str, str]:
    if not clean_whitespace(url):
        return {
            "job_location": "",
            "remote_preferences": "",
            "notes_suffix": "link missing",
        }

    try:
        html, status = fetch_html(url)
        if not html:
            return {
                "job_location": "",
                "remote_preferences": "",
                "notes_suffix": f"link fetch failed ({status})",
            }

        parsed = extract_best_content(html)
        if not parsed:
            return {
                "job_location": "",
                "remote_preferences": "",
                "notes_suffix": "link parse failed",
            }

        structured = parsed["structured"]
        header_text = parsed["header_text"]
        role_body_text = parsed["role_body_text"]
        role_context_text = parsed["role_context_text"]

        evidence_text = "\n".join(
            [
                header_text,
                role_body_text,
                structured.get("location_raw", ""),
                role_context_text,
            ]
        )
        evidence_text = normalize_quotes(evidence_text)

        fallback_location = normalize_location_rule_based(evidence_text, allowed_locations)
        fallback_remote_preferences = normalize_remote_preferences_value(
            detect_remote_preferences_rule_based(evidence_text)
        )

        return {
            "job_location": fallback_location,
            "remote_preferences": fallback_remote_preferences,
            "notes_suffix": "link checked",
        }
    except Exception as e:
        return {
            "job_location": "",
            "remote_preferences": "",
            "notes_suffix": f"link error ({e})",
        }


def build_notes(
    used_link_for_location: bool,
    used_link_for_remote_preferences: bool,
    extra_note: str = "",
) -> str:
    if used_link_for_location and used_link_for_remote_preferences:
        base = "link used for location and remote preferences"
    elif used_link_for_location:
        base = "link used for location"
    elif used_link_for_remote_preferences:
        base = "link used for remote preferences"
    else:
        base = "description only"

    extra_note = clean_whitespace(extra_note)
    if extra_note and extra_note not in {"link checked"}:
        return f"{base} | {extra_note}"
    return base


def determine_relevance_and_primary_title(
    clean_title: str,
    description_text: str,
    role_context_text: str,
    allowed_job_titles: List[str],
) -> Tuple[str, str, str, str]:
    strong_match, matched_title, match_reason = is_strong_allowed_title_match(clean_title, allowed_job_titles)

    if strong_match:
        category_seed = derive_category_from_title_or_text(matched_title, clean_title, description_text)
        return "Relevant", match_reason, matched_title, category_seed

    relevance = ai_check_relevance(
        job_title=clean_title,
        role_context_text=role_context_text,
        fallback_role_relevance="Not relevant",
        fallback_reason="Title does not clearly map to predefined job titles.",
        fallback_job_category="",
    )

    role_relevance = relevance.get("role_relevance", "") or "Not relevant"
    role_relevance_reason = relevance.get("role_relevance_reason", "") or "No reason returned."
    category_seed = normalize_category_for_skills(relevance.get("job_category", ""))

    return role_relevance, role_relevance_reason, matched_title, category_seed


def process_job(
    job: JobInput,
    allowed_locations: List[str],
    allowed_salaries: List[int],
    tp_skills: List[str],
    nontp_skills: List[str],
    allowed_job_titles: List[str],
) -> JobResult:
    description_text = prepare_description_text(job.job_description)
    clean_title = clean_job_title(job.job_title) or infer_title_from_description(description_text)

    if not description_text:
        return blank_result_with_relevance(
            job=job,
            role_relevance="Not relevant",
            reason="Missing job description in input.",
            notes="description missing",
        )

    header_text = "\n".join([x for x in [job.company_name, clean_title] if x]).strip()
    role_context_text = "\n\n".join([x for x in [job.company_name, clean_title, description_text] if x]).strip()

    fallback_location = normalize_location_rule_based(description_text, allowed_locations)
    fallback_remote_preferences = normalize_remote_preferences_value(
        detect_remote_preferences_rule_based(description_text)
    )
    fallback_remote_days = detect_remote_days_rule_based(description_text, fallback_remote_preferences)
    fallback_salary_min, fallback_salary_max, fallback_salary_currency, fallback_salary_period = parse_explicit_salary(
        description_text, allowed_salaries
    )
    fallback_visa = detect_visa_rule_based(description_text)
    fallback_job_type = detect_job_type_rule_based(description_text, "")

    role_relevance, role_relevance_reason, title_match_hint, relevance_job_category = determine_relevance_and_primary_title(
        clean_title=clean_title,
        description_text=description_text,
        role_context_text=role_context_text,
        allowed_job_titles=allowed_job_titles,
    )

    if role_relevance == "Not relevant":
        return blank_result_with_relevance(
            job=job,
            role_relevance=role_relevance,
            reason=role_relevance_reason,
            notes="description only",
        )

    result = JobResult(
        row_id=job.row_id,
        company_name=job.company_name,
        input_job_title=job.job_title,
        job_url=job.job_url,
        role_relevance=role_relevance,
        role_relevance_reason=role_relevance_reason,
        job_category=relevance_job_category,
    )

    core_fields = ai_extract_core_fields(
        job_title=clean_title,
        header_text=header_text,
        role_body_text=description_text,
        allowed_locations=allowed_locations,
        fallback_job_category=result.job_category,
        fallback_location=fallback_location,
        fallback_remote_preferences=fallback_remote_preferences,
        fallback_remote_days=fallback_remote_days,
        fallback_salary_min=fallback_salary_min,
        fallback_salary_max=fallback_salary_max,
        fallback_salary_currency=fallback_salary_currency,
        fallback_salary_period=fallback_salary_period,
        fallback_visa=fallback_visa,
        fallback_job_type=fallback_job_type,
    )

    result.job_category = normalize_category_for_skills(core_fields.get("job_category", "")) or result.job_category
    if not result.job_category:
        result.job_category = derive_category_from_title_or_text(title_match_hint, clean_title, description_text)

    ai_location = core_fields.get("job_location", "") or ""
    if ai_location and location_value_has_evidence(ai_location, description_text, allowed_locations):
        result.job_location = ai_location
    else:
        result.job_location = fallback_location

    result.remote_preferences = normalize_remote_preferences_value(
        core_fields.get("remote_preferences", "") or fallback_remote_preferences
    )
    result.visa_sponsorship = core_fields.get("visa_sponsorship", "") or fallback_visa
    result.job_type = core_fields.get("job_type", "") or fallback_job_type

    ai_remote_days = str(core_fields.get("remote_days", "") or "").strip()
    if ai_remote_days.lower() == "not specified":
        ai_remote_days = "not specified"

    explicit_remote_days_allowed = has_explicit_remote_days_evidence(description_text)
    if explicit_remote_days_allowed and ai_remote_days:
        result.remote_days = ai_remote_days
    else:
        result.remote_days = fallback_remote_days

    result.salary_min = core_fields.get("salary_min", "") or fallback_salary_min
    result.salary_max = core_fields.get("salary_max", "") or fallback_salary_max
    result.salary_currency = core_fields.get("salary_currency", "") or fallback_salary_currency
    result.salary_period = core_fields.get("salary_period", "") or fallback_salary_period

    needs_location_fallback = not clean_whitespace(result.job_location)
    needs_remote_fallback = (
        not clean_whitespace(result.remote_preferences)
        or result.remote_preferences.lower() == "not specified"
    )

    used_link_for_location = False
    used_link_for_remote_preferences = False
    extra_note = ""

    if ENABLE_URL_FALLBACK and needs_location_fallback and needs_remote_fallback and clean_whitespace(job.job_url):
        url_enrichment = extract_location_and_remote_from_url(job.job_url, allowed_locations)
        extra_note = url_enrichment.get("notes_suffix", "")

        if clean_whitespace(url_enrichment.get("job_location", "")):
            result.job_location = url_enrichment["job_location"]
            used_link_for_location = True

        if clean_whitespace(url_enrichment.get("remote_preferences", "")):
            result.remote_preferences = normalize_remote_preferences_value(
                url_enrichment["remote_preferences"]
            ) or result.remote_preferences
            used_link_for_remote_preferences = True

    explicit_salary_check = parse_explicit_salary(normalize_quotes(description_text), allowed_salaries)
    if explicit_salary_check != ("", "", "", ""):
        result.salary_min, result.salary_max, result.salary_currency, result.salary_period = explicit_salary_check

    result.salary_min = snap_salary_value(result.salary_min, allowed_salaries)
    result.salary_max = snap_salary_value(result.salary_max, allowed_salaries)

    exact_title = normalize_job_title_from_list(clean_title, allowed_job_titles)
    if exact_title:
        job_titles = [exact_title]
    elif title_match_hint:
        job_titles = [title_match_hint]
    else:
        job_titles = ai_map_job_titles_only(
            position_name=clean_title,
            description=description_text,
            allowed_job_titles=allowed_job_titles,
        )

    job_titles = postprocess_job_titles(
        job_title=clean_title,
        description=description_text,
        predicted_titles=job_titles,
        allowed_job_titles=allowed_job_titles,
    )

    result.job_title_tag_1 = job_titles[0] if len(job_titles) > 0 else ""
    result.job_title_tag_2 = job_titles[1] if len(job_titles) > 1 else ""
    result.job_title_tag_3 = job_titles[2] if len(job_titles) > 2 else ""

    seniorities = fallback_seniorities(clean_title, description_text)
    seniorities = refine_seniorities_rule_based(clean_title, description_text, seniorities)
    seniorities = normalize_seniority_list(seniorities)

    result.seniority_1 = seniorities[0] if len(seniorities) > 0 else ""
    result.seniority_2 = seniorities[1] if len(seniorities) > 1 else ""
    result.seniority_3 = seniorities[2] if len(seniorities) > 2 else ""

    skill_list = tp_skills if result.job_category == "T&P" else nontp_skills
    exact_skills = exact_match_skills_in_order(description_text, skill_list, limit=10)
    exact_skills = dedupe_keep_order(exact_skills)

    if len(exact_skills) >= MIN_EXACT_SKILLS_TO_SKIP_AI:
        final_skills = exact_skills[:10]
    else:
        if len(exact_skills) == 0:
            ai_skills = ai_generate_skills_full(
                role_category=result.job_category,
                description=description_text,
                candidate_skills=[],
                allowed_skills=skill_list,
            )
            final_skills = ai_skills[:10]
        else:
            additional_skills = ai_generate_additional_skills(
                role_category=result.job_category,
                description=description_text,
                existing_skills=exact_skills,
                candidate_skills=exact_skills,
                allowed_skills=skill_list,
            )
            final_skills = (exact_skills + additional_skills)[:10]

    allowed_lower = {s.lower(): s for s in skill_list}
    final_skills = [allowed_lower[sk.lower()] for sk in final_skills if sk.lower() in allowed_lower]
    final_skills = dedupe_keep_order(final_skills)[:10]

    padded_skills = (final_skills + [""] * 10)[:10]
    result.skill_1 = padded_skills[0]
    result.skill_2 = padded_skills[1]
    result.skill_3 = padded_skills[2]
    result.skill_4 = padded_skills[3]
    result.skill_5 = padded_skills[4]
    result.skill_6 = padded_skills[5]
    result.skill_7 = padded_skills[6]
    result.skill_8 = padded_skills[7]
    result.skill_9 = padded_skills[8]
    result.skill_10 = padded_skills[9]

    result.notes = build_notes(
        used_link_for_location=used_link_for_location,
        used_link_for_remote_preferences=used_link_for_remote_preferences,
        extra_note=extra_note,
    )

    return result


def write_results(rows: List[JobResult], filepath: str) -> None:
    if not rows:
        return

    data = [asdict(r) for r in rows]
    fieldnames = list(data[0].keys())

    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def main() -> None:
    start = time.time()

    jobs = load_jobs_input(INPUT_JOBS_FILE)
    allowed_locations = load_text_list(LOCATIONS_CSV)
    allowed_salaries = load_salary_list(SALARIES_CSV)
    tp_skills = load_text_list(TP_SKILLS_CSV)
    nontp_skills = load_text_list(NONTP_SKILLS_CSV)
    allowed_job_titles = load_text_list(JOB_TITLES_CSV)

    print(f"[INFO] Jobs loaded: {len(jobs)}")
    print(f"[INFO] Allowed locations loaded: {len(allowed_locations)}")
    print(f"[INFO] Allowed salaries loaded: {len(allowed_salaries)}")
    print(f"[INFO] T&P skills loaded: {len(tp_skills)}")
    print(f"[INFO] NonT&P skills loaded: {len(nontp_skills)}")
    print(f"[INFO] Job titles loaded: {len(allowed_job_titles)}")
    print(f"[INFO] MAX_WORKERS: {MAX_WORKERS}")
    print(f"[INFO] ENABLE_URL_FALLBACK: {'yes' if ENABLE_URL_FALLBACK else 'no'}")

    results_map = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(
                process_job,
                job,
                allowed_locations,
                allowed_salaries,
                tp_skills,
                nontp_skills,
                allowed_job_titles,
            ): idx
            for idx, job in enumerate(jobs, start=1)
        }

        done_count = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            done_count += 1
            current_job = jobs[idx - 1]
            try:
                row = future.result()
            except Exception as e:
                row = JobResult(
                    row_id=current_job.row_id,
                    company_name=current_job.company_name,
                    input_job_title=current_job.job_title,
                    job_url=current_job.job_url,
                    role_relevance="Not relevant",
                    role_relevance_reason=f"Unhandled error: {e}",
                    notes="unhandled error",
                )
            results_map[idx] = row
            print(f"[{done_count}/{len(jobs)}] Finished index {idx}")

    results = [results_map[i] for i in sorted(results_map.keys())]
    write_results(results, OUTPUT_CSV_FILE)

    elapsed = round(time.time() - start, 2)
    print(f"[DONE] Wrote {len(results)} rows to {OUTPUT_CSV_FILE} in {elapsed}s")


if __name__ == "__main__":
    main()
