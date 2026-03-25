import csv
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import List

import pandas as pd
from bs4 import BeautifulSoup

from classifiers import (
    ai_check_relevance,
    ai_extract_core_fields,
    ai_extract_salary_only,
    ai_generate_additional_skills,
    ai_generate_skills_full,
    ai_map_job_titles_only,
    ai_map_seniority_only,
)
from fetch_extract import (
    extract_best_content,
    fetch_html,
    fetch_with_playwright,
    should_try_playwright,
)
from formatters import (
    build_skills_source_text,
    exact_match_skills_in_order,
    plain_text_to_html_preserve_structure,
)
from validators import (
    detect_job_type_rule_based,
    detect_remote_days_rule_based,
    detect_remote_preferences_rule_based,
    detect_visa_rule_based,
    fallback_seniorities,
    has_explicit_remote_days_evidence,
    is_relevant_by_rules,
    is_tp_by_rules,
    line_has_compensation_anchor,
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

INPUT_URLS_FILE = "job_urls.txt"
OUTPUT_CSV_FILE = "results.csv"

LOCATIONS_CSV = "predefined_locations.csv"
SALARIES_CSV = "predefined_salaries.csv"
TP_SKILLS_CSV = "predefined_tp_skills.csv"
NONTP_SKILLS_CSV = "predefined_nontp_skills.csv"
JOB_TITLES_CSV = "predefined_job_titles.csv"

MAX_WORKERS = int(os.getenv("MAX_WORKERS", "6"))
FAST_MODE = os.getenv("FAST_MODE", "1").strip().lower() in {"1", "true", "yes", "y"}
MIN_EXACT_SKILLS_TO_SKIP_AI = int(os.getenv("MIN_EXACT_SKILLS_TO_SKIP_AI", "3"))


@dataclass
class JobResult:
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
    job_description: str = ""

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


def load_urls(filepath: str) -> List[str]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing input file: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


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

    company_words = [
        " ltd", " limited", " inc", " llc", " plc", " gmbh", " s.a.", " remote",
        " full-time", " full time", " part-time", " part time", " contract",
        " freelance", " location", " office"
    ]
    low = title.lower()
    cut = len(title)
    for w in company_words:
        idx = low.find(w)
        if idx != -1:
            cut = min(cut, idx)

    title = title[:cut].strip(" -|,·")
    return clean_whitespace(title)


def extract_best_title_candidate(
    soup: BeautifulSoup,
    structured_title: str,
    title_tag_text: str,
    header_text: str,
    role_body_text: str,
) -> str:
    candidates: List[str] = []

    if structured_title:
        candidates.append(structured_title)
    if title_tag_text:
        candidates.append(title_tag_text)

    for tag_name in ["h1", "h2", "h3"]:
        for tag in soup.find_all(tag_name):
            txt = clean_whitespace(tag.get_text(" ", strip=True))
            if txt and 3 <= len(txt) <= 220:
                candidates.append(txt)

    meta_props = [
        ("property", "og:title"),
        ("name", "twitter:title"),
        ("name", "title"),
    ]
    for attr, val in meta_props:
        tag = soup.find("meta", attrs={attr: val})
        if tag and tag.get("content"):
            txt = clean_whitespace(tag.get("content"))
            if txt:
                candidates.append(txt)

    early_lines = split_lines("\n".join([header_text, role_body_text]))[:20]
    for line in early_lines:
        low = line.lower()
        if len(line) > 200:
            continue
        if any(x in low for x in ["location:", "salary", "rate:", "contract", "apply", "posted", "remote in ", "hybrid", "onsite", "work location"]):
            continue
        if re.search(r"\b(job description|overview|responsibilities|qualifications|about the role|description|requirements)\b", low):
            continue
        candidates.append(line)

    cleaned = []
    for cand in candidates:
        ct = clean_job_title(cand)
        low = ct.lower().strip()
        if not ct:
            continue
        if low in {"job openings", "careers", "apply", "overview", "description", "requirements"}:
            continue
        if len(low) < 3:
            continue
        cleaned.append(ct)

    cleaned = dedupe_keep_order(cleaned)
    if cleaned:
        cleaned.sort(key=lambda x: ((" - " in x) or (" | " in x), len(x)))
        return cleaned[0]
    return ""


def should_skip_relevance_ai(clean_title: str, role_body_text: str, header_text: str) -> tuple[bool, str, str]:
    strong_rule_relevant = is_relevant_by_rules(clean_title, role_body_text, header_text)
    substantial_role_text = len(clean_whitespace(role_body_text)) >= 700

    if strong_rule_relevant and substantial_role_text:
        return True, "Relevant", "Rule-based relevance: clear allowed role with substantial extracted content."

    return False, "", ""


def should_run_core_fields_ai(
    fallback_location: str,
    fallback_remote_preferences: str,
    fallback_remote_days: str,
    fallback_salary_min: str,
    fallback_salary_max: str,
    fallback_salary_currency: str,
    fallback_salary_period: str,
    fallback_visa: str,
    fallback_job_type: str,
) -> bool:
    missing_count = sum(
        1 for x in [
            fallback_location,
            fallback_remote_preferences,
            fallback_remote_days,
            fallback_salary_min,
            fallback_salary_max,
            fallback_salary_currency,
            fallback_salary_period,
            fallback_visa,
            fallback_job_type,
        ]
        if not str(x).strip()
    )
    return missing_count >= 3


def should_run_salary_ai(evidence_text: str) -> bool:
    candidate_lines = split_lines(evidence_text)[:220]
    return any(line_has_compensation_anchor(line) for line in candidate_lines)


def should_run_titles_ai(clean_title: str, allowed_job_titles: List[str]) -> bool:
    exact = normalize_job_title_from_list(clean_title, allowed_job_titles)
    return not bool(exact)


def should_run_seniority_ai(clean_title: str, role_text: str) -> bool:
    fallback = fallback_seniorities(clean_title, role_text)
    return len(fallback) == 0


def blank_result_with_relevance(job_url: str, role_relevance: str, reason: str) -> JobResult:
    return JobResult(
        job_url=job_url,
        role_relevance=role_relevance,
        role_relevance_reason=reason,
    )


def fetch_and_extract(url: str):
    html, status = fetch_html(url)

    if html:
        parsed = extract_best_content(html)

        if should_try_playwright(url, html, parsed["role_context_text"]):
            pw_html, pw_status = fetch_with_playwright(url)
            if pw_html:
                html = pw_html
                parsed = extract_best_content(html)
                status = pw_status

        return html, parsed, status

    pw_html, pw_status = fetch_with_playwright(url)
    if not pw_html:
        return None, None, f"{status}; {pw_status}"

    parsed = extract_best_content(pw_html)
    return pw_html, parsed, pw_status


def process_url(
    url: str,
    allowed_locations: List[str],
    allowed_salaries: List[int],
    tp_skills: List[str],
    nontp_skills: List[str],
    allowed_job_titles: List[str],
) -> JobResult:
    html, parsed, fetch_status = fetch_and_extract(url)

    if not html or not parsed:
        return blank_result_with_relevance(
            job_url=url,
            role_relevance="Not relevant",
            reason="Could not fetch or extract job content.",
        )

    structured = parsed["structured"]
    title_tag_text = parsed["title_tag_text"]
    header_text = parsed["header_text"]
    role_body_text = parsed["role_body_text"]
    role_context_text = parsed["role_context_text"]

    raw_title = extract_best_title_candidate(
        soup=parsed["soup"],
        structured_title=structured.get("title", ""),
        title_tag_text=title_tag_text,
        header_text=header_text,
        role_body_text=role_body_text,
    )
    clean_title = clean_job_title(raw_title)

    evidence_text = "\n".join([header_text, role_body_text, structured.get("location_raw", "")])
    evidence_text = normalize_quotes(evidence_text)

    fallback_location = normalize_location_rule_based(evidence_text, allowed_locations)
    fallback_remote_preferences = normalize_remote_preferences_value(
        detect_remote_preferences_rule_based(evidence_text)
    )
    fallback_remote_days = detect_remote_days_rule_based(evidence_text, fallback_remote_preferences)
    fallback_salary_min, fallback_salary_max, fallback_salary_currency, fallback_salary_period = parse_explicit_salary(
        evidence_text, allowed_salaries
    )
    fallback_visa = detect_visa_rule_based(role_context_text)
    fallback_job_type = detect_job_type_rule_based(evidence_text, structured.get("employment_type_raw", ""))
    fallback_description = clean_whitespace(role_body_text)
    fallback_job_category = "T&P" if is_tp_by_rules(clean_title, role_body_text) else "NonT&P"

    strong_rule_relevant = bool(clean_title and is_relevant_by_rules(clean_title, role_body_text, header_text))
    substantial_role_text = len(clean_whitespace(role_body_text)) >= 700

    fallback_role_relevance = "Relevant" if strong_rule_relevant else "Not relevant"
    fallback_reason = "Fallback classification based on extracted title and role text."

    skip_rel_ai, rel_value, rel_reason = should_skip_relevance_ai(clean_title, role_body_text, header_text)
    if skip_rel_ai:
        role_relevance = rel_value
        role_relevance_reason = rel_reason
        relevance_job_category = fallback_job_category
    else:
        relevance = ai_check_relevance(
            job_title=clean_title,
            role_context_text=role_context_text,
            fallback_role_relevance=fallback_role_relevance,
            fallback_reason=fallback_reason,
            fallback_job_category=fallback_job_category,
        )
        role_relevance = relevance.get("role_relevance", "") or fallback_role_relevance
        role_relevance_reason = relevance.get("role_relevance_reason", "") or fallback_reason
        relevance_job_category = normalize_category_for_skills(relevance.get("job_category", "")) or fallback_job_category

    if role_relevance == "Not relevant":
        return blank_result_with_relevance(
            job_url=url,
            role_relevance=role_relevance,
            reason=role_relevance_reason,
        )

    if not substantial_role_text and len(clean_whitespace(role_context_text)) < 350:
        return blank_result_with_relevance(
            job_url=url,
            role_relevance="Not relevant",
            reason="Insufficient extracted role content for reliable enrichment.",
        )

    result = JobResult(
        job_url=url,
        role_relevance=role_relevance,
        role_relevance_reason=role_relevance_reason,
    )
    result.job_category = relevance_job_category

    if should_run_core_fields_ai(
        fallback_location=fallback_location,
        fallback_remote_preferences=fallback_remote_preferences,
        fallback_remote_days=fallback_remote_days,
        fallback_salary_min=fallback_salary_min,
        fallback_salary_max=fallback_salary_max,
        fallback_salary_currency=fallback_salary_currency,
        fallback_salary_period=fallback_salary_period,
        fallback_visa=fallback_visa,
        fallback_job_type=fallback_job_type,
    ):
        core_fields = ai_extract_core_fields(
            job_title=clean_title,
            header_text=header_text,
            role_body_text=role_body_text,
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
    else:
        core_fields = {
            "job_category": result.job_category,
            "job_location": fallback_location,
            "remote_preferences": fallback_remote_preferences,
            "remote_days": fallback_remote_days,
            "salary_min": fallback_salary_min,
            "salary_max": fallback_salary_max,
            "salary_currency": fallback_salary_currency,
            "salary_period": fallback_salary_period,
            "visa_sponsorship": fallback_visa,
            "job_type": fallback_job_type,
        }

    result.job_category = normalize_category_for_skills(core_fields.get("job_category", "")) or result.job_category

    ai_location = core_fields.get("job_location", "") or ""
    if ai_location and location_value_has_evidence(ai_location, evidence_text, allowed_locations):
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

    explicit_remote_days_allowed = has_explicit_remote_days_evidence("\n".join([header_text, role_body_text]))
    if explicit_remote_days_allowed and ai_remote_days:
        result.remote_days = ai_remote_days
    else:
        result.remote_days = fallback_remote_days

    result.salary_min = core_fields.get("salary_min", "") or fallback_salary_min
    result.salary_max = core_fields.get("salary_max", "") or fallback_salary_max
    result.salary_currency = core_fields.get("salary_currency", "") or fallback_salary_currency
    result.salary_period = core_fields.get("salary_period", "") or fallback_salary_period

    if should_run_salary_ai(evidence_text):
        salary_fields = ai_extract_salary_only(
            job_title=clean_title,
            header_text=header_text,
            role_body_text=role_body_text,
            fallback_salary_min=result.salary_min,
            fallback_salary_max=result.salary_max,
            fallback_salary_currency=result.salary_currency,
            fallback_salary_period=result.salary_period,
        )
        result.salary_min = salary_fields.get("salary_min", "") or result.salary_min
        result.salary_max = salary_fields.get("salary_max", "") or result.salary_max
        result.salary_currency = salary_fields.get("salary_currency", "") or result.salary_currency
        result.salary_period = salary_fields.get("salary_period", "") or result.salary_period

    final_html_description = plain_text_to_html_preserve_structure(role_body_text)
    if not final_html_description:
        final_html_description = plain_text_to_html_preserve_structure(fallback_description)
    result.job_description = final_html_description

    hard_evidence_text = "\n".join([header_text, strip_html(result.job_description), structured.get("location_raw", "")])
    hard_evidence_text = normalize_quotes(hard_evidence_text)

    result.job_location = normalize_location_rule_based(hard_evidence_text, allowed_locations) or result.job_location
    result.remote_preferences = normalize_remote_preferences_value(
        detect_remote_preferences_rule_based(hard_evidence_text)
    ) or result.remote_preferences

    remote_days_candidates = [
        detect_remote_days_rule_based(header_text, result.remote_preferences),
        detect_remote_days_rule_based(role_body_text, result.remote_preferences),
        detect_remote_days_rule_based(strip_html(result.job_description), result.remote_preferences),
        detect_remote_days_rule_based(hard_evidence_text, result.remote_preferences),
        result.remote_days,
        fallback_remote_days,
    ]
    result.remote_days = next((x for x in remote_days_candidates if str(x).strip() != ""), "")

    explicit_salary_check = parse_explicit_salary(hard_evidence_text, allowed_salaries)
    if explicit_salary_check != ("", "", "", ""):
        result.salary_min, result.salary_max, result.salary_currency, result.salary_period = explicit_salary_check

    result.salary_min = snap_salary_value(result.salary_min, allowed_salaries)
    result.salary_max = snap_salary_value(result.salary_max, allowed_salaries)

    if should_run_titles_ai(clean_title, allowed_job_titles):
        job_titles = ai_map_job_titles_only(
            position_name=clean_title,
            description=strip_html(result.job_description) or role_body_text,
            allowed_job_titles=allowed_job_titles,
        )
    else:
        exact_title = normalize_job_title_from_list(clean_title, allowed_job_titles)
        job_titles = [exact_title] if exact_title else []

    job_titles = postprocess_job_titles(
        job_title=clean_title,
        description=strip_html(result.job_description) or role_body_text,
        predicted_titles=job_titles,
        allowed_job_titles=allowed_job_titles,
    )

    result.job_title_tag_1 = job_titles[0] if len(job_titles) > 0 else ""
    result.job_title_tag_2 = job_titles[1] if len(job_titles) > 1 else ""
    result.job_title_tag_3 = job_titles[2] if len(job_titles) > 2 else ""

    role_text_for_seniority = strip_html(result.job_description) or role_body_text
    if should_run_seniority_ai(clean_title, role_text_for_seniority):
        seniorities = ai_map_seniority_only(
            position_name=clean_title,
            description=role_text_for_seniority,
        )
    else:
        seniorities = fallback_seniorities(clean_title, role_text_for_seniority)

    if not seniorities:
        seniorities = fallback_seniorities(clean_title, role_text_for_seniority)

    seniorities = refine_seniorities_rule_based(
        clean_title,
        role_text_for_seniority,
        seniorities
    )
    seniorities = normalize_seniority_list(seniorities)

    result.seniority_1 = seniorities[0] if len(seniorities) > 0 else ""
    result.seniority_2 = seniorities[1] if len(seniorities) > 1 else ""
    result.seniority_3 = seniorities[2] if len(seniorities) > 2 else ""

    skills_source_text = build_skills_source_text(result.job_description)
    skill_list = tp_skills if result.job_category == "T&P" else nontp_skills

    exact_skills = exact_match_skills_in_order(skills_source_text, skill_list, limit=10)
    exact_skills = dedupe_keep_order(exact_skills)

    if FAST_MODE and len(exact_skills) >= MIN_EXACT_SKILLS_TO_SKIP_AI:
        final_skills = exact_skills[:10]
    else:
        if len(exact_skills) == 0:
            ai_skills = ai_generate_skills_full(
                role_category=result.job_category,
                description=skills_source_text,
                candidate_skills=[],
                allowed_skills=skill_list,
            )
            final_skills = ai_skills[:10]
        else:
            additional_skills = ai_generate_additional_skills(
                role_category=result.job_category,
                description=skills_source_text,
                existing_skills=exact_skills,
                candidate_skills=exact_skills,
                allowed_skills=skill_list,
            )
            final_skills = (exact_skills + additional_skills)[:10]

    allowed_lower = {s.lower(): s for s in skill_list}
    final_skills = [
        allowed_lower[sk.lower()]
        for sk in final_skills
        if sk.lower() in allowed_lower
    ]
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

    urls = load_urls(INPUT_URLS_FILE)
    allowed_locations = load_text_list(LOCATIONS_CSV)
    allowed_salaries = load_salary_list(SALARIES_CSV)
    tp_skills = load_text_list(TP_SKILLS_CSV)
    nontp_skills = load_text_list(NONTP_SKILLS_CSV)
    allowed_job_titles = load_text_list(JOB_TITLES_CSV)

    print(f"[INFO] URLs loaded: {len(urls)}")
    print(f"[INFO] Allowed locations loaded: {len(allowed_locations)}")
    print(f"[INFO] Allowed salaries loaded: {len(allowed_salaries)}")
    print(f"[INFO] T&P skills loaded: {len(tp_skills)}")
    print(f"[INFO] NonT&P skills loaded: {len(nontp_skills)}")
    print(f"[INFO] Job titles loaded: {len(allowed_job_titles)}")
    print(f"[INFO] FAST_MODE: {'yes' if FAST_MODE else 'no'}")
    print(f"[INFO] MAX_WORKERS: {MAX_WORKERS}")

    results_map = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(
                process_url,
                url,
                allowed_locations,
                allowed_salaries,
                tp_skills,
                nontp_skills,
                allowed_job_titles,
            ): idx
            for idx, url in enumerate(urls, start=1)
        }

        done_count = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            done_count += 1
            try:
                row = future.result()
            except Exception as e:
                row = JobResult(
                    job_url=urls[idx - 1],
                    role_relevance="Not relevant",
                    role_relevance_reason=f"Unhandled error: {e}",
                )
            results_map[idx] = row
            print(f"[{done_count}/{len(urls)}] Finished index {idx}")

    results = [results_map[i] for i in sorted(results_map.keys())]
    write_results(results, OUTPUT_CSV_FILE)

    elapsed = round(time.time() - start, 2)
    print(f"[DONE] Wrote {len(results)} rows to {OUTPUT_CSV_FILE} in {elapsed}s")


if __name__ == "__main__":
    main()
