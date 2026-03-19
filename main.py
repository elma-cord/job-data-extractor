import csv
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, List

import pandas as pd
from bs4 import BeautifulSoup

from classifiers import (
    ai_check_relevance,
    ai_enrich_skills,
    ai_map_job_titles_only,
    ai_map_seniority_only,
    ai_tag_relevant_job,
)
from fetch_extract import (
    extract_best_content,
    fetch_html,
    fetch_with_playwright,
    looks_like_js_shell,
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
    is_relevant_by_rules,
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


INPUT_URLS_FILE = "job_urls.txt"
OUTPUT_CSV_FILE = "results.csv"

LOCATIONS_CSV = "predefined_locations.csv"
SALARIES_CSV = "predefined_salaries.csv"
TP_SKILLS_CSV = "predefined_tp_skills.csv"
NONTP_SKILLS_CSV = "predefined_nontp_skills.csv"
JOB_TITLES_CSV = "predefined_job_titles.csv"


@dataclass
class JobResult:
    job_url: str = ""
    job_title: str = ""
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

    source_method: str = ""
    status: str = ""
    notes: str = ""


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


def extract_best_title_candidate(soup: BeautifulSoup, structured_title: str, title_tag_text: str, header_text: str, role_body_text: str) -> str:
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


def process_url(
    url: str,
    allowed_locations: List[str],
    allowed_salaries: List[int],
    tp_skills: List[str],
    nontp_skills: List[str],
    allowed_job_titles: List[str],
) -> JobResult:
    result = JobResult(job_url=url)

    html, status = fetch_html(url)
    source_method = "html"
    notes = [status]

    if html:
        parsed = extract_best_content(html)
        if looks_like_js_shell(html, parsed["role_context_text"]):
            pw_html, pw_status = fetch_with_playwright(url)
            notes.append(pw_status)
            if pw_html:
                html = pw_html
                source_method = "playwright"
                parsed = extract_best_content(html)
    else:
        pw_html, pw_status = fetch_with_playwright(url)
        notes.append(pw_status)
        if not pw_html:
            result.status = "failed"
            result.source_method = "none"
            result.notes = " | ".join(notes)
            return result
        html = pw_html
        source_method = "playwright"
        parsed = extract_best_content(html)

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
    fallback_remote_preferences = detect_remote_preferences_rule_based(evidence_text)
    fallback_remote_days = detect_remote_days_rule_based(evidence_text, fallback_remote_preferences)
    fallback_salary_min, fallback_salary_max, fallback_salary_currency, fallback_salary_period = parse_explicit_salary(
        evidence_text, allowed_salaries
    )
    fallback_visa = detect_visa_rule_based(role_context_text)
    fallback_job_type = detect_job_type_rule_based(evidence_text, structured.get("employment_type_raw", ""))
    fallback_description = clean_whitespace(role_body_text)

    substantial_role_text = len(clean_whitespace(role_body_text)) >= 700
    fallback_role_relevance = "Relevant" if ((clean_title and is_relevant_by_rules(clean_title, role_body_text, header_text)) or substantial_role_text) else "Not relevant"
    fallback_reason = "Fallback classification based on extracted title and role text."

    if not clean_title and role_body_text:
        if is_relevant_by_rules(role_body_text[:300], role_body_text, header_text):
            fallback_role_relevance = "Relevant"

    relevance = ai_check_relevance(
        job_title=clean_title,
        role_context_text=role_context_text,
        fallback_role_relevance=fallback_role_relevance,
        fallback_reason=fallback_reason,
    )

    result.job_title = clean_title
    result.role_relevance = relevance.get("role_relevance", "") or fallback_role_relevance
    result.role_relevance_reason = relevance.get("role_relevance_reason", "") or fallback_reason
    result.source_method = source_method
    result.status = "ok"

    if result.role_relevance == "Not relevant" and substantial_role_text and (
        is_relevant_by_rules(clean_title, role_body_text, header_text) or is_relevant_by_rules(role_body_text[:300], role_body_text, header_text)
    ):
        result.role_relevance = "Relevant"
        result.role_relevance_reason = "Rule-based override: extracted title and role description indicate an allowed relevant role."

    if result.role_relevance == "Not relevant":
        result.notes = " | ".join(notes + ["stopped_after_relevance"])
        return result

    fallback_job_category = "T&P" if is_tp_by_rules(clean_title, role_body_text) else "NonT&P"

    tagged = ai_tag_relevant_job(
        job_title=clean_title,
        header_text=header_text,
        role_body_text=role_body_text,
        allowed_locations=allowed_locations,
        allowed_salaries=allowed_salaries,
        allowed_job_titles=allowed_job_titles,
        fallback_job_category=fallback_job_category,
        fallback_location=fallback_location,
        fallback_remote_preferences=fallback_remote_preferences,
        fallback_remote_days=fallback_remote_days,
        fallback_salary_min=fallback_salary_min,
        fallback_salary_max=fallback_salary_max,
        fallback_salary_currency=fallback_salary_currency,
        fallback_salary_period=fallback_salary_period,
        fallback_visa=fallback_visa,
        fallback_job_type=fallback_job_type,
        fallback_job_description=fallback_description,
    )

    result.job_category = normalize_category_for_skills(tagged.get("job_category", "")) or fallback_job_category

    ai_location = tagged.get("job_location", "") or ""
    if ai_location and location_value_has_evidence(ai_location, evidence_text, allowed_locations):
        result.job_location = ai_location
    else:
        result.job_location = fallback_location

    result.remote_preferences = tagged.get("remote_preferences", "") or fallback_remote_preferences
    result.salary_min = tagged.get("salary_min", "") or fallback_salary_min
    result.salary_max = tagged.get("salary_max", "") or fallback_salary_max
    result.salary_currency = tagged.get("salary_currency", "") or fallback_salary_currency
    result.salary_period = tagged.get("salary_period", "") or fallback_salary_period
    result.visa_sponsorship = tagged.get("visa_sponsorship", "") or fallback_visa
    result.job_type = tagged.get("job_type", "") or fallback_job_type

    final_html_description = plain_text_to_html_preserve_structure(role_body_text)
    if not final_html_description:
        final_html_description = plain_text_to_html_preserve_structure(tagged.get("job_description", "") or fallback_description)
    result.job_description = final_html_description

    hard_evidence_text = "\n".join([header_text, strip_html(result.job_description), structured.get("location_raw", "")])
    hard_evidence_text = normalize_quotes(hard_evidence_text)

    result.job_location = normalize_location_rule_based(hard_evidence_text, allowed_locations) or result.job_location
    result.remote_preferences = detect_remote_preferences_rule_based(hard_evidence_text) or result.remote_preferences

    remote_days_candidates = [
        detect_remote_days_rule_based(header_text, result.remote_preferences),
        detect_remote_days_rule_based(role_body_text, result.remote_preferences),
        detect_remote_days_rule_based(strip_html(result.job_description), result.remote_preferences),
        detect_remote_days_rule_based(hard_evidence_text, result.remote_preferences),
        fallback_remote_days,
    ]
    result.remote_days = next((x for x in remote_days_candidates if str(x).strip() != ""), "")

    explicit_salary_check = parse_explicit_salary(hard_evidence_text, allowed_salaries)
    if explicit_salary_check == ("", "", "", ""):
        result.salary_min = ""
        result.salary_max = ""
        result.salary_currency = ""
        result.salary_period = ""
    else:
        result.salary_min, result.salary_max, result.salary_currency, result.salary_period = explicit_salary_check

    result.salary_min = snap_salary_value(result.salary_min, allowed_salaries)
    result.salary_max = snap_salary_value(result.salary_max, allowed_salaries)

    job_titles = tagged.get("job_titles", []) if isinstance(tagged.get("job_titles", []), list) else []
    if not job_titles:
        job_titles = ai_map_job_titles_only(
            position_name=clean_title,
            description=strip_html(result.job_description) or role_body_text,
            allowed_job_titles=allowed_job_titles,
        )

    job_titles = postprocess_job_titles(
        job_title=clean_title,
        description=strip_html(result.job_description) or role_body_text,
        predicted_titles=job_titles,
        allowed_job_titles=allowed_job_titles,
    )

    result.job_title_tag_1 = job_titles[0] if len(job_titles) > 0 else ""
    result.job_title_tag_2 = job_titles[1] if len(job_titles) > 1 else ""
    result.job_title_tag_3 = job_titles[2] if len(job_titles) > 2 else ""

    seniorities = tagged.get("seniorities", []) if isinstance(tagged.get("seniorities", []), list) else []
    if not seniorities:
        seniorities = ai_map_seniority_only(
            position_name=clean_title,
            description=strip_html(result.job_description) or role_body_text,
        )
    if not seniorities:
        seniorities = fallback_seniorities(clean_title, strip_html(result.job_description) or role_body_text)

    seniorities = refine_seniorities_rule_based(
        clean_title,
        strip_html(result.job_description) or role_body_text,
        seniorities
    )
    seniorities = normalize_seniority_list(seniorities)

    result.seniority_1 = seniorities[0] if len(seniorities) > 0 else ""
    result.seniority_2 = seniorities[1] if len(seniorities) > 1 else ""
    result.seniority_3 = seniorities[2] if len(seniorities) > 2 else ""

    skills_source_text = build_skills_source_text(result.job_description)
    skill_list = tp_skills if result.job_category == "T&P" else nontp_skills

    exact_skills = exact_match_skills_in_order(skills_source_text, skill_list, limit=10)
    final_skills = ai_enrich_skills(
        role_category=result.job_category,
        description=skills_source_text,
        exact_skills=exact_skills,
        allowed_skills=skill_list,
    )

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

    note_parts = notes + [
        "step3_validators_and_formatters_modularized",
        "formatted_position_name_removed",
        "deterministic_html_description_added",
        "remote_days_explicit_evidence_only",
        "seniority_rule_updated_v3",
        "job_title_postprocess_updated_v3",
        "salary_snapped_to_closest_allowed_value",
    ]
    if len(job_titles) == 0:
        note_parts.append("job_titles_empty")
    else:
        note_parts.append("job_titles_present")
    if len(exact_skills) > 0:
        note_parts.append("skills_from_clean_role_text_exact_plus_ai_enrichment")
    else:
        note_parts.append("skills_from_clean_role_text_ai_only")
    result.notes = " | ".join(note_parts)

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

    results: List[JobResult] = []

    for idx, url in enumerate(urls, start=1):
        print(f"[{idx}/{len(urls)}] Processing: {url}")
        try:
            row = process_url(
                url=url,
                allowed_locations=allowed_locations,
                allowed_salaries=allowed_salaries,
                tp_skills=tp_skills,
                nontp_skills=nontp_skills,
                allowed_job_titles=allowed_job_titles,
            )
        except Exception as e:
            row = JobResult(
                job_url=url,
                status="failed",
                notes=f"unhandled_error: {e}"
            )
        results.append(row)

    write_results(results, OUTPUT_CSV_FILE)

    elapsed = round(time.time() - start, 2)
    print(f"[DONE] Wrote {len(results)} rows to {OUTPUT_CSV_FILE} in {elapsed}s")


if __name__ == "__main__":
    main()
