import csv
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

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


def strip_html(text: str) -> str:
    return clean_whitespace(BeautifulSoup(text or "", "lxml").get_text("\n", strip=True))


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


def normalize_quotes(text: str) -> str:
    return (
        (text or "")
        .replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("•", "-")
        .replace("-", "-")
    )


def canonical_label(text: str) -> str:
    s = normalize_quotes(text or "").lower().strip()
    s = s.replace("&", "and")
    s = s.replace("/", "")
    s = s.replace("\\", "")
    s = s.replace(",", "")
    s = s.replace("-", "")
    s = re.sub(r"\s+", "", s)
    return s


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


def normalize_category_for_skills(job_category: str) -> str:
    low = (job_category or "").strip().lower()
    if low in {"t&p", "tp", "tech & product", "tech and product"}:
        return "T&P"
    if low in {"nont&p", "non-t&p", "non tp", "not t&p", "not tp", "nontp", "non tech", "non-tech"}:
        return "NonT&P"
    return ""


def build_location_lookup(allowed_locations: List[str]) -> Dict[str, str]:
    city_lookup = {}
    for loc in allowed_locations:
        city = loc.split(",")[0].strip().lower()
        if city and city not in city_lookup:
            city_lookup[city] = loc
    return city_lookup


def gather_location_lines(text: str) -> List[str]:
    from fetch_extract import normalize_common_location_aliases

    out = []
    lines = split_lines(normalize_common_location_aliases(text))

    strong_patterns = [
        r"^location\s*:",
        r"^locations\s*:",
        r"^country\s*:",
        r"^city\s*:",
        r"^office\s*:",
        r"^based at\b",
        r"^based in\b",
        r"^home based\b",
        r"^position role type\s*:",
        r"\blocated in\b",
        r"\bthis is a .* position located in\b",
        r"\bwork location\b",
        r"\bideal locations\b",
        r"\bterritories available\b",
        r"\bremote in\b",
        r"\bright to work in the\b",
        r"\boffice attendance\b",
        r"\boffice based\b",
        r"\bworking week must be office based\b",
        r"^location\b",
    ]
    soft_tokens = [
        " hybrid", " remote", " onsite", " on-site", " home based", " from home",
        " united kingdom", " office attendance", " days per week", " remote-enabled",
        " office based", "% of your working week", "guildford", "london", "bristol"
    ]

    for idx, line in enumerate(lines[:360]):
        low = f" {normalize_quotes(line).lower()} "
        if any(re.search(p, low, flags=re.I) for p in strong_patterns):
            out.append(line)
            continue
        if idx < 120 and any(tok in low for tok in soft_tokens):
            out.append(line)

    out.extend(lines[:50])
    return dedupe_keep_order(out)


def location_value_has_evidence(location_value: str, text: str, allowed_locations: List[str]) -> bool:
    if not location_value or not text:
        return False

    candidate_lines = gather_location_lines(text)
    if not candidate_lines:
        return False

    joined = normalize_quotes("\n".join(candidate_lines)).lower()
    loc_low = normalize_quotes(location_value).lower().strip()
    if loc_low and re.search(rf"(?<![a-z]){re.escape(loc_low)}(?![a-z])", joined):
        return True

    city_lookup = build_location_lookup(allowed_locations)
    city = loc_low.split(",")[0].strip()
    mapped = city_lookup.get(city, "")
    city_to_check = mapped.split(",")[0].strip().lower() if mapped else city
    if city_to_check and re.search(rf"\b{re.escape(city_to_check)}\b", joined):
        return True

    return False


def location_specificity_score(location: str) -> int:
    if not location:
        return 0
    parts = [p.strip() for p in location.split(",") if p.strip()]
    score = len(parts) * 100 + len(location)
    low = location.lower()
    if "united kingdom" in low:
        score += 5
    if len(parts) >= 3:
        score += 50
    return score


def score_location_line(line: str) -> int:
    low = normalize_quotes(line).lower().strip()
    score = 0
    if low.startswith("location:"):
        score += 1000
    if low.startswith("locations:"):
        score += 900
    if low.startswith("work location"):
        score += 850
    if low.startswith("position role type"):
        score += 800
    if "based in" in low:
        score += 500
    if "office based" in low or "office attendance" in low:
        score += 400
    if len(line) <= 120:
        score += 80
    return score


def normalize_location_rule_based(text: str, allowed_locations: List[str]) -> str:
    from fetch_extract import normalize_common_location_aliases

    if not allowed_locations or not text:
        return ""

    city_lookup = build_location_lookup(allowed_locations)
    candidate_lines = gather_location_lines(text)
    if not candidate_lines:
        return ""

    matches: List[tuple] = []

    for line_idx, line in enumerate(candidate_lines):
        line_norm = normalize_common_location_aliases(line)
        line_low = normalize_quotes(line_norm).lower()
        line_score = score_location_line(line)

        for loc in allowed_locations:
            loc_norm = normalize_common_location_aliases(loc)
            loc_low = normalize_quotes(loc_norm).lower()
            m = re.search(rf"(?<![a-z]){re.escape(loc_low)}(?![a-z])", line_low)
            if m:
                matches.append((-line_score, line_idx, -location_specificity_score(loc_norm), loc))

        for city, full in city_lookup.items():
            m = re.search(rf"\b{re.escape(city)}\b", line_low)
            if m:
                matches.append((-line_score, line_idx, -location_specificity_score(full), full))

    joined = normalize_common_location_aliases("\n".join(candidate_lines))
    joined_low = normalize_quotes(joined).lower()

    if re.search(r"\b(united kingdom|great britain)\b", joined_low):
        for loc in allowed_locations:
            if loc.lower() in {"united kingdom", "england, united kingdom"}:
                matches.append((-10, 999999, -location_specificity_score(loc), loc))

    broad_fallbacks = [
        "United Kingdom",
        "England, United Kingdom",
        "Scotland, United Kingdom",
        "Wales, United Kingdom",
        "Northern Ireland, United Kingdom",
        "Ireland",
    ]
    for broad in broad_fallbacks:
        for loc in allowed_locations:
            if loc.lower() == broad.lower() and re.search(rf"(?<![a-z]){re.escape(broad.lower())}(?![a-z])", joined_low):
                matches.append((-1, 999999, -location_specificity_score(loc), loc))

    if not matches:
        return ""

    matches.sort(key=lambda x: (x[0], x[1], x[2]))
    return matches[0][3]


def detect_remote_preferences_rule_based(text: str) -> str:
    lines = gather_location_lines(text)
    low = normalize_quotes("\n".join(lines)).lower()
    found = []

    remote_signal = bool(re.search(
        r"\bhome based\b|\bhome-based\b|\buk remote\b|\bfully remote\b|\bremote enabled\b|\bremote-enabled\b|\bremote working\b|\bwork from home\b|\bwfh\b",
        low
    ))
    hybrid_signal = bool(re.search(
        r"\bhybrid\b|\bagile working\b|\boffice attendance requirement\b|\b\d+\s*-\s*\d+\s+days?\s+per week\b.*\boffice\b|\b\d+\s+days?\s+per week\b.*\boffice\b|\b\d{1,3}%\s+of your working week must be office based\b",
        low
    ))
    onsite_signal = bool(re.search(r"\bonsite\b|\bon-site\b|\bon site\b|\bin office\b|\bin-office\b", low))

    pct_match = re.search(r"\b(\d{1,3})%\s+of your working week must be office based\b", low)
    if pct_match:
        pct = int(pct_match.group(1))
        if pct >= 100:
            onsite_signal = True
        elif pct > 0:
            hybrid_signal = True

    if remote_signal:
        found.append("remote")
    if hybrid_signal:
        found.append("hybrid")
    if onsite_signal:
        found.append("onsite")

    if re.search(r"\bwork location\b.*\bremote\b", low) or re.search(r"\bremote working within the united kingdom\b", low):
        return "remote"

    ordered = [x for x in ["onsite", "hybrid", "remote"] if x in found]

    if "hybrid" in ordered and "remote" in ordered:
        return "hybrid, remote"
    return ", ".join(ordered)


def has_explicit_remote_days_evidence(text: str) -> bool:
    low = normalize_quotes(text or "").lower()
    low = low.replace("approximately", "approx")
    low = low.replace("a week", "per week")
    low = re.sub(r"\s+", " ", low)

    patterns = [
        r"\boffice attendance requirement(?: of)?(?: approx\.?)?\s*\d+\s*-\s*\d+\s+days?\s+per week\b",
        r"\boffice attendance requirement(?: of)?(?: approx\.?)?\s*\d+\s+days?\s+per week\b",
        r"\b\d+\s*-\s*\d+\s+days?\s+(?:per week\s+)?(?:in the office|from the office|office|on site|onsite|on-site)\b",
        r"\b\d+\s+days?\s+(?:per week\s+)?(?:in the office|from the office|office|on site|onsite|on-site)\b",
        r"\b\d+\s*-\s*\d+\s+days?\s+(?:per week\s+)?(?:from home|remote|wfh)\b",
        r"\b\d+\s+days?\s+(?:per week\s+)?(?:from home|remote|wfh)\b",
        r"\b\d{1,3}%\s+of your working week must be office based\b",
    ]
    return any(re.search(p, low) for p in patterns)


def _extract_remote_days_from_text(text: str) -> str:
    low = normalize_quotes(text).lower()
    low = low.replace("approximately", "approx")
    low = low.replace("a week", "per week")
    low = re.sub(r"\s+", " ", low)

    if re.search(r"\bfully remote\b", low):
        return ""
    if re.search(r"\bremote working within the united kingdom\b", low):
        return ""
    if re.search(r"\buk remote\b", low):
        return ""

    office_patterns = [
        r"\boffice attendance requirement(?: of)?(?: approx\.?)?\s*(\d)\s*-\s*(\d)\s+days?\s+per week\b",
        r"\boffice attendance requirement(?: of)?(?: approx\.?)?\s*(\d)\s+days?\s+per week\b",
        r"\b(?:approx\.?\s*)?(\d)\s*-\s*(\d)\s+days?\s+(?:per week\s+)?(?:in the office|from the office|office|on site|onsite|on-site)\b",
        r"\b(?:approx\.?\s*)?(\d)\s+days?\s+(?:per week\s+)?(?:in the office|from the office|office|on site|onsite|on-site)\b",
        r"\bthis role has .*?office attendance requirement .*?(\d)\s*-\s*(\d)\s+days?\s+per week\b",
        r"\bthis role has .*?office attendance requirement .*?(\d)\s+days?\s+per week\b",
        r"\b(\d)\s*-\s*(\d)\s+days?\s+per week\b.*?\boffice\b",
        r"\b(\d)\s+days?\s+per week\b.*?\boffice\b",
    ]

    for pat in office_patterns:
        m = re.search(pat, low)
        if not m:
            continue
        groups = [int(g) for g in m.groups() if g is not None]
        if len(groups) == 2:
            office_max = max(groups)
            return str(max(0, 5 - office_max))
        if len(groups) == 1:
            return str(max(0, 5 - groups[0]))

    pct_match = re.search(r"\b(\d{1,3})%\s+of your working week must be office based\b", low)
    if pct_match:
        pct = int(pct_match.group(1))
        if 0 < pct < 100:
            office_days = round((pct / 100.0) * 5)
            office_days = max(1, min(5, office_days))
            return str(max(0, 5 - office_days))

    remote_patterns = [
        r"\b(\d)\s*-\s*(\d)\s+days?\s+(?:per week\s+)?(?:from home|remote|wfh)\b",
        r"\b(\d)\s+days?\s+(?:per week\s+)?(?:from home|remote|wfh)\b",
        r"\b(?:work|working)\s+(\d)\s+days?\s+(?:from home|remote|wfh)\b",
    ]

    for pat in remote_patterns:
        m = re.search(pat, low)
        if not m:
            continue
        groups = [int(g) for g in m.groups() if g is not None]
        if len(groups) == 2:
            return str(min(groups))
        if len(groups) == 1:
            return str(groups[0])

    return ""


def detect_remote_days_rule_based(text: str, remote_prefs: str = "") -> str:
    from fetch_extract import normalize_common_location_aliases

    candidate_text = "\n".join([
        "\n".join(gather_location_lines(text)),
        normalize_common_location_aliases(text),
    ])

    if not has_explicit_remote_days_evidence(candidate_text):
        return ""

    days = _extract_remote_days_from_text(candidate_text)
    if days:
        return days

    prefs_low = (remote_prefs or "").lower()
    if prefs_low == "onsite":
        return ""

    return ""


def normalize_currency_symbol(sym: str) -> str:
    return {"£": "GBP", "$": "USD", "€": "EUR"}.get(sym, "")


def looks_like_non_salary_line(line: str) -> bool:
    low = line.lower()
    bad_tokens = [
        "reference:",
        "posted on",
        "posted",
        "apply by",
        "closing date",
        "interview",
        "q1",
        "q2",
        "q3",
        "q4",
        "roi",
        "employees",
        "revenue",
        "years of experience",
        "iso",
        "nist",
        "800-53",
        "2024",
        "2025",
        "2026",
        "2027",
        "2028",
        "m365",
    ]
    return any(tok in low for tok in bad_tokens)


def line_has_compensation_anchor(line: str) -> bool:
    low = line.lower()
    anchors = [
        "salary",
        "pay",
        "salary band",
        "salary location band",
        "compensation",
        "package",
        "rate",
        "per annum",
        "annum",
        "annual",
        "day rate",
        "daily rate",
        "hourly",
        "per day",
        "per hour",
        "p/d",
        "p.a",
    ]
    return any(a in low for a in anchors) or bool(re.search(r"[£$€]", line))


def parse_money_range_from_line(line: str) -> tuple:
    line = normalize_quotes(line)
    low = line.lower()

    if looks_like_non_salary_line(line) or not line_has_compensation_anchor(line):
        return "", "", "", ""

    period = ""
    if re.search(r"\b(p/d|per day|day rate|daily rate)\b", low):
        period = "day"
    elif re.search(r"\b(per hour|hourly)\b", low):
        period = "hour"
    elif re.search(r"\b(per month|monthly)\b", low):
        period = "month"
    elif re.search(r"\b(per annum|annum|annual|annually|salary)\b", low):
        period = "year"

    range_pat = re.compile(
        r"(£|\$|€)\s?(\d{1,3}(?:,\d{3})+|\d{4,6})(?:\.\d+)?\s*(?:-|–|to)\s*(?:\1\s?)?(\d{1,3}(?:,\d{3})+|\d{4,6})(?:\.\d+)?",
        flags=re.I,
    )
    m = range_pat.search(line)
    if m:
        currency = normalize_currency_symbol(m.group(1))
        min_raw = re.sub(r"[^\d]", "", m.group(2))
        max_raw = re.sub(r"[^\d]", "", m.group(3))
        if min_raw and max_raw:
            return min_raw, max_raw, currency, period

    single_pat = re.compile(r"(£|\$|€)\s?(\d{1,3}(?:,\d{3})+|\d{4,6})(?:\.\d+)?", flags=re.I)
    singles = list(single_pat.finditer(line))
    if len(singles) == 1:
        currency = normalize_currency_symbol(singles[0].group(1))
        raw = re.sub(r"[^\d]", "", singles[0].group(2))
        if raw:
            return raw, raw, currency, period

    return "", "", "", ""


def parse_explicit_salary(text: str, _allowed_salaries_unused: List[int]) -> tuple:
    lines = split_lines(text)
    candidate_lines = [line for line in lines[:220] if line_has_compensation_anchor(line)]

    for line in candidate_lines:
        parsed = parse_money_range_from_line(line)
        if parsed != ("", "", "", ""):
            return parsed

    return "", "", "", ""


def snap_salary_value(value: str, allowed_salaries: List[int]) -> str:
    if not value or not allowed_salaries:
        return value
    try:
        num = int(float(str(value).replace(",", "").strip()))
    except Exception:
        return value
    closest = min(allowed_salaries, key=lambda x: (abs(x - num), x))
    return str(closest)


def detect_visa_rule_based(text: str) -> str:
    low = normalize_quotes(text).lower()

    yes_patterns = [
        r"visa sponsorship available",
        r"sponsorship available",
        r"we can sponsor",
        r"offers visa sponsorship",
        r"skilled worker visa",
        r"will sponsor visa",
    ]
    no_patterns = [
        r"no visa sponsorship",
        r"unable to sponsor",
        r"cannot sponsor",
        r"do not sponsor",
        r"without sponsorship",
        r"right to work in the uk is mandatory",
        r"must have the right to work",
        r"subject to .*right to work",
        r"applicants must have the right to work in the united kingdom",
        r"unable to consider candidates who require visa sponsorship",
    ]

    if any(re.search(p, low) for p in no_patterns):
        return "no"
    if any(re.search(p, low) for p in yes_patterns):
        return "yes"
    return ""


def detect_job_type_rule_based(text: str, structured_employment_type: str = "") -> str:
    low = f"{structured_employment_type} {text}".lower()

    permanent_patterns = [r"\bpermanent\b", r"\bfull[- ]time\b", r"\bstandard\b"]
    ftc_patterns = [r"\btemporary\b", r"\bfixed[- ]term\b", r"\bmaternity cover\b", r"\bftc\b"]
    part_time_patterns = [r"\bpart[- ]time\b", r"\bjob share\b", r"\bjob-share\b"]
    freelance_patterns = [r"\bfreelance\b", r"\bcontract\b", r"\bcontracting\b"]

    if any(re.search(p, low) for p in permanent_patterns):
        return "Permanent"
    if any(re.search(p, low) for p in ftc_patterns):
        return "FTC"
    if any(re.search(p, low) for p in part_time_patterns):
        return "Part Time"
    if any(re.search(p, low) for p in freelance_patterns):
        return "Freelance/Contract"
    return ""


def _extract_years(text_low: str) -> List[int]:
    years = []
    for m in re.finditer(r"\b(\d)\s*-\s*(\d)\s+years?\b", text_low):
        years.extend([int(m.group(1)), int(m.group(2))])
    for m in re.finditer(r"\b(\d)\+?\s+years?\b", text_low):
        years.append(int(m.group(1)))
    return years


def _manager_like_title(title_low: str) -> bool:
    return bool(re.search(r"\b(manager|mgr)\b", title_low))


def _leadership_title(title_low: str) -> bool:
    return bool(re.search(
        r"\b(head of|director|technical director|vp|vice president|chief|cfo|cto|cio|coo|cmo|cro|cpo|cso|engineering manager)\b",
        title_low
    ))


def _juniorish_title(title_low: str) -> bool:
    return bool(re.search(r"\b(junior|jr|assistant|associate|entry)\b", title_low))


def _strong_lead_signals(text_low: str) -> bool:
    patterns = [
        r"\bmanage(?:s|d|ing)? team\b",
        r"\bmanage(?:s|d|ing)? team members\b",
        r"\bcoaching team members\b",
        r"\bmanage and coach\b",
        r"\blead end-to-end\b",
        r"\bact as an escalation point\b",
        r"\bescalation point\b",
        r"\bline manager\b",
        r"\bdirect report\b",
        r"\bpeople management\b",
        r"\bteam management\b",
        r"\bmanage(?:s|d|ing)? a third-party provider\b",
        r"\bown(?:s|ing)? end-to-end\b",
        r"\blead .* delivery\b",
        r"\blead .* team\b",
        r"\bprogramme governance\b",
        r"\bprogram governance\b",
        r"\bgovernance reviews\b",
        r"\bstakeholder management\b",
        r"\bresource management\b",
        r"\bplanning and forecasting\b",
        r"\bplanning & forecasting\b",
        r"\boversight and reporting\b",
        r"\bfull contract lifecycle\b",
        r"\bfull project lifecycle\b",
    ]
    return any(re.search(p, text_low) for p in patterns)


def _strategic_senior_signals(title_low: str, text_low: str) -> bool:
    title_patterns = [
        r"\bpmo manager\b",
        r"\bprogramme manager\b",
        r"\bprogram manager\b",
        r"\bproject manager\b",
        r"\baccount manager\b",
        r"\bnational account manager\b",
        r"\bkey account manager\b",
        r"\boperations manager\b",
        r"\bproduct manager\b",
    ]
    text_patterns = [
        r"\bfull contract lifecycle\b",
        r"\bfull project lifecycle\b",
        r"\bbidding\b",
        r"\binitiating\b",
        r"\bexecuting\b",
        r"\bmonitoring/?controlling\b",
        r"\bclosing\b",
        r"\bprogramme governance\b",
        r"\bgovernance aware culture\b",
        r"\bkey stakeholders\b",
        r"\bcommunicating complex information\b",
        r"\bresource management\b",
        r"\bplanning and forecasting\b",
        r"\bearned value\b",
        r"\brisk management\b",
        r"\bcost control\b",
        r"\bconfiguration control\b",
        r"\bkpis\b",
        r"\boversight\b",
        r"\bteam assigned to projects\b",
    ]
    return any(re.search(p, title_low) for p in title_patterns) or any(re.search(p, text_low) for p in text_patterns)


def normalize_seniority_list(values: List[str]) -> List[str]:
    order = ["entry", "junior", "mid", "senior", "lead", "leadership"]
    found = []
    for v in values:
        low = str(v).strip().lower()
        if low in order and low not in found:
            found.append(low)
    return [x for x in order if x in found][:3]


def fallback_seniorities(job_title: str, role_text: str) -> List[str]:
    title_low = (job_title or "").lower()
    text_low = (role_text or "").lower()

    if _leadership_title(title_low):
        return ["leadership"]

    if _manager_like_title(title_low):
        if _juniorish_title(title_low):
            return ["mid", "lead"]
        if _strategic_senior_signals(title_low, text_low) or _strong_lead_signals(text_low):
            return ["senior", "lead"]
        return ["senior", "lead"]

    if any(x in title_low for x in ["senior", "sr ", "sr."]):
        return ["senior"]

    if any(x in title_low for x in ["lead ", "principal"]):
        return ["lead"]

    if any(x in title_low for x in ["junior", "jr "]):
        return ["junior", "mid"]

    if any(x in title_low for x in ["associate", "mid weight", "mid-weight"]):
        return ["mid"]

    if _strong_lead_signals(text_low):
        return ["lead"]

    years = _extract_years(text_low)
    if years:
        max_y = max(years)
        if max_y <= 1:
            return ["entry", "junior"]
        if max_y == 2:
            return ["junior", "mid"]
        if 3 <= max_y <= 5:
            return ["senior"]
        if max_y > 5:
            return ["senior", "lead"]

    return []


def refine_seniorities_rule_based(job_title: str, role_text: str, seniorities: List[str]) -> List[str]:
    title_low = (job_title or "").lower()
    text_low = (role_text or "").lower()
    current = normalize_seniority_list(seniorities)

    leadership_title = _leadership_title(title_low)
    manager_title = _manager_like_title(title_low)
    junior_signal = _juniorish_title(title_low)
    strong_lead = _strong_lead_signals(text_low)
    strategic_senior = _strategic_senior_signals(title_low, text_low)

    if leadership_title:
        return ["leadership"]

    if manager_title:
        current = [s for s in current if s not in {"entry", "junior", "mid"}]
        if "senior" not in current and not junior_signal:
            current.append("senior")
        if "lead" not in current:
            current.append("lead")
        return normalize_seniority_list(current)

    if strong_lead:
        current = [s for s in current if s != "mid"]
        if strategic_senior and "senior" not in current:
            current.append("senior")
        if "lead" not in current:
            current.append("lead")
        return normalize_seniority_list(current)

    years = _extract_years(text_low)
    if years and max(years) >= 5 and "mid" in current and "senior" in current:
        current = [s for s in current if s != "mid"]

    return normalize_seniority_list(current)


def is_tp_by_rules(job_title: str, role_text: str) -> bool:
    text = f"{job_title}\n{role_text}".lower()
    tp_patterns = [
        r"\bengineer\b", r"\bdeveloper\b", r"\bsoftware\b", r"\bdata\b",
        r"\bmachine learning\b", r"\bml\b", r"\bai\b", r"\bproduct\b",
        r"\bdesigner\b", r"\bux\b", r"\bui\b", r"\bqa\b", r"\bdevops\b",
        r"\bsite reliability\b", r"\bsre\b", r"\barchitect\b", r"\bcloud\b",
        r"\bplatform\b", r"\binfrastructure\b", r"\bsystem administrator\b",
        r"\bsystems administrator\b", r"\bsupport engineer\b", r"\btechnical support\b",
        r"\bnetwork engineer\b", r"\bsolutions engineer\b", r"\bit support\b",
        r"\b2nd line\b", r"\bsecond line\b", r"\bmsp\b", r"\bwindows server\b",
        r"\bactive directory\b", r"\bexchange\b", r"\bhyper-v\b", r"\bvmware\b",
        r"\bcitrix\b", r"\brouter\b", r"\bfirewall\b", r"\bvpn\b",
        r"\bremote desktop\b", r"\bvoip\b", r"\brust\b", r"\bkubernetes\b",
        r"\bopenshift\b", r"\bgrpc\b", r"\bprotocol buffers\b",
    ]
    return any(re.search(p, text) for p in tp_patterns)


def is_relevant_by_rules(job_title: str, role_text: str, header_text: str = "") -> bool:
    text = f"{job_title}\n{header_text}\n{role_text}".lower()

    allowed_patterns = [
        r"\btalent acquisition\b", r"\brecruiter\b", r"\brecruitment\b",
        r"\bhuman resources\b", r"\bhead of hr\b", r"\bhr manager\b",
        r"\bpeople ops\b", r"\bpeople operations\b", r"\bpeople partner\b",
        r"\baccount manager\b", r"\baccount executive\b", r"\baccount director\b",
        r"\bcustomer success\b", r"\bcsm\b", r"\brenewals\b", r"\bclient services\b",
        r"\bbusiness analyst\b", r"\bbusiness operations\b", r"\boperations\b",
        r"\bchange manager\b", r"\btransformation\b", r"\bpmo\b",
        r"\bprogramme manager\b", r"\bprogram manager\b", r"\bproject manager\b",
        r"\brisk\b", r"\bcompliance\b", r"\blegal\b", r"\bfinance\b",
        r"\baccounting\b", r"\bfp&a\b", r"\brevops\b", r"\bsales operations\b",
        r"\bsdr\b", r"\bbdr\b", r"\bmarketing\b", r"\bseo\b", r"\bpr\b",
        r"\bcommunications\b", r"\bengineer\b", r"\bdeveloper\b",
        r"\barchitect\b", r"\bdevops\b", r"\bqa\b", r"\bproduct\b",
        r"\bdesigner\b", r"\bux\b", r"\bui\b", r"\bdata\b",
        r"\bmachine learning\b", r"\bai\b", r"\bsecurity\b", r"\bcloud\b",
        r"\bnetwork\b", r"\binfrastructure\b", r"\bsystems\b",
        r"\bsupport engineer\b", r"\bsystem administrator\b", r"\bsystem engineer\b",
        r"\bsolutions engineer\b", r"\bit support\b", r"\b2nd line\b", r"\bsecond line\b",
    ]
    excluded_patterns = [
        r"\bteacher\b", r"\bnurse\b", r"\bwaiter\b", r"\bchef\b",
        r"\bconstruction\b", r"\bcivil engineer\b", r"\belectrician\b",
        r"\bmechanical engineer\b", r"\bmanufacturing\b", r"\bmaritime\b",
        r"\bmicrobiology\b", r"\binjection molding\b", r"\bwarehouse\b",
        r"\bdriver\b", r"\bcleaner\b",
    ]

    if any(re.search(p, text) for p in allowed_patterns):
        return True
    if any(re.search(p, text) for p in excluded_patterns):
        return False
    return False


def normalize_job_title_from_list(value: str, allowed_job_titles: List[str]) -> str:
    value_clean = clean_whitespace(value)
    if not value_clean:
        return ""

    for jt in allowed_job_titles:
        if value_clean.lower() == jt.lower():
            return jt

    canon_value = canonical_label(value_clean)
    for jt in allowed_job_titles:
        if canon_value == canonical_label(jt):
            return jt

    return ""


def find_allowed_title_case_insensitive(target: str, allowed_job_titles: List[str]) -> str:
    for jt in allowed_job_titles:
        if jt.strip().lower() == target.strip().lower():
            return jt
    return ""


def postprocess_job_titles(job_title: str, description: str, predicted_titles: List[str], allowed_job_titles: List[str]) -> List[str]:
    title_low = normalize_quotes(job_title or "").lower()
    desc_low = normalize_quotes(description or "").lower()

    out = []
    for t in predicted_titles:
        exact = normalize_job_title_from_list(t, allowed_job_titles)
        if exact and exact not in out:
            out.append(exact)

    csm_account_manager = find_allowed_title_case_insensitive("CSM / Account Manager", allowed_job_titles)
    account_executive = find_allowed_title_case_insensitive("Account Executive", allowed_job_titles)
    system_engineer = find_allowed_title_case_insensitive("System Engineer", allowed_job_titles)
    system_admin = find_allowed_title_case_insensitive("System Administrator", allowed_job_titles)
    devops_engineer = find_allowed_title_case_insensitive("DevOps Engineer", allowed_job_titles)
    solutions_engineer = find_allowed_title_case_insensitive("Solutions Engineer", allowed_job_titles)
    full_stack = find_allowed_title_case_insensitive("Full Stack", allowed_job_titles)
    cloud_engineer = find_allowed_title_case_insensitive("Cloud Engineer", allowed_job_titles)
    marketing_analyst = find_allowed_title_case_insensitive("Marketing Analyst", allowed_job_titles)
    data_insight_analyst = find_allowed_title_case_insensitive("Data / Insight Analyst", allowed_job_titles)
    data_scientist = find_allowed_title_case_insensitive("Data Scientist", allowed_job_titles)
    business_analyst = find_allowed_title_case_insensitive("Business Analyst", allowed_job_titles)

    account_manager_signals = [
        r"\bnational account manager\b", r"\bkey account manager\b", r"\baccount manager\b",
        r"\bcustomer success manager\b", r"\bcustomer success\b", r"\bcsm\b",
        r"\bclient success manager\b", r"\brenewals manager\b", r"\bsales account management\b",
        r"\baccount management\b",
    ]
    account_exec_signals = [
        r"\baccount executive\b", r"\bnew business\b", r"\bhunter\b",
        r"\bpipeline generation\b", r"\bprospecting\b", r"\bquota\b",
    ]

    sales_account_role_signal = any(re.search(p, title_low) for p in account_manager_signals) or any(
        re.search(p, desc_low) for p in [
            r"\bwholesale\b", r"\bbuyers\b", r"\baccounts\b", r"\btrade terms\b",
            r"\bpromotional plans\b", r"\bretailers\b", r"\bbrands\b",
            r"\bcommercial conversations\b", r"\bdistributor partners\b",
            r"\bnetwork development\b", r"\bvalue, gross margin, and volume targets\b",
        ]
    )

    if csm_account_manager and sales_account_role_signal:
        if csm_account_manager in out:
            out.remove(csm_account_manager)
        out.insert(0, csm_account_manager)

    if account_executive and any(re.search(p, title_low) for p in account_exec_signals):
        if account_executive not in out:
            out.append(account_executive)

    if sales_account_role_signal:
        technical_titles_to_remove = {
            system_engineer, system_admin, devops_engineer,
            solutions_engineer, full_stack, cloud_engineer,
        }
        out = [x for x in out if x not in technical_titles_to_remove and x]

    system_engineer_signal = any(re.search(p, title_low) for p in [
        r"\bsenior systems engineer\b", r"\bsystems engineer\b",
        r"\bsystem engineer\b", r"\bdtn software engineer\b",
    ]) or any(re.search(p, desc_low) for p in [
        r"\bvirtuali[sz]ation\b", r"\bvmware\b", r"\bopenshift\b",
        r"\bkubernetes\b", r"\blinux\b", r"\benterprise infrastructure\b",
        r"\bdelay-tolerant networking\b", r"\bnetworking\b", r"\bstorage\b",
        r"\bqueueing\b", r"\bgrpc\b", r"\bprotocol buffers\b", r"\brust\b",
        r"\bday 2 operations\b",
    ])

    if not sales_account_role_signal:
        if system_engineer and system_engineer_signal:
            if system_engineer in out:
                out.remove(system_engineer)
            out.insert(0, system_engineer)

        if devops_engineer and any(re.search(p, desc_low) for p in [
            r"\bkubernetes\b", r"\bopenshift\b", r"\bcontaineri[sz]ing\b",
            r"\bdistributed system\b", r"\bcloud platforms?\b", r"\baws\b",
            r"\bgoogle cloud platform\b", r"\bgcp\b", r"\bmicroservice\b",
            r"\bevent-driven\b", r"\bscalable and distributed\b",
        ]):
            if devops_engineer not in out:
                out.append(devops_engineer)

        if full_stack and full_stack in out:
            if not any(re.search(p, desc_low) for p in [r"\bfront-end\b", r"\bfrontend\b", r"\breact\b", r"\bjavascript\b", r"\btypescript\b", r"\bui\b"]):
                out = [x for x in out if x != full_stack]

        if solutions_engineer and solutions_engineer in out:
            if not any(re.search(p, title_low + "\n" + desc_low) for p in [r"\bsolutions engineer\b", r"\bpre-sales\b", r"\bpresales\b", r"\bsales engineer\b"]):
                out = [x for x in out if x != solutions_engineer]

        if system_admin and system_admin in out and system_engineer_signal:
            out = [x for x in out if x != system_admin]
            if system_engineer and system_engineer not in out:
                out.insert(0, system_engineer)

        if cloud_engineer and cloud_engineer in out and system_engineer_signal and any(re.search(p, title_low) for p in [r"\bsystems engineer\b", r"\bsystem engineer\b"]):
            out = [x for x in out if x != cloud_engineer]

    marketing_data_signal = any(re.search(p, desc_low) for p in [
        r"\bcampaign performance\b", r"\bcustomer behaviour\b", r"\bcustomer behavior\b",
        r"\bloyalty scheme\b", r"\bcrm analytics\b", r"\bmarketing optimisation\b",
        r"\bmarketing optimization\b", r"\brfm\b", r"\bltv\b", r"\bbasket analysis\b",
        r"\bchurn analysis\b", r"\bpromotional performance\b", r"\baudience counts\b",
        r"\bclient database\b", r"\besp\b",
    ])

    if marketing_data_signal:
        if marketing_analyst and marketing_analyst not in out:
            out.insert(0, marketing_analyst)
        if data_insight_analyst and data_insight_analyst not in out:
            out.append(data_insight_analyst)

        if business_analyst and business_analyst in out:
            out = [x for x in out if x != business_analyst]

        data_scientist_strong = any(re.search(p, desc_low) for p in [
            r"\bmachine learning\b", r"\bpredictive\b", r"\bstatistical model",
            r"\bmodelling\b", r"\bmodeling\b", r"\bclassification\b",
            r"\bregression\b", r"\bdata science\b",
        ])
        if data_scientist and data_scientist in out and not data_scientist_strong:
            out = [x for x in out if x != data_scientist]

    return dedupe_keep_order(out)[:3]


def html_escape(text: str) -> str:
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def plain_text_to_html_preserve_structure(job_description_text: str) -> str:
    text = clean_whitespace(job_description_text)
    if not text:
        return ""

    lines = split_lines(text)
    if not lines:
        return ""

    html_parts: List[str] = []
    bullet_buffer: List[str] = []

    def flush_bullets():
        nonlocal bullet_buffer, html_parts
        if bullet_buffer:
            html_parts.append("<ul>")
            for item in bullet_buffer:
                html_parts.append(f"<li>{html_escape(item)}</li>")
            html_parts.append("</ul>")
            bullet_buffer = []

    def is_heading_line(line: str) -> bool:
        raw = normalize_quotes(line).strip()
        norm = raw.lower().strip(" :-•\t")
        if raw.endswith(":") and len(raw) <= 120:
            return True
        if norm in {
            "role and responsibilities", "key responsibilities", "skills, knowledge and expertise",
            "person specification", "required qualifications", "preferred qualifications",
            "responsibilities", "about you", "experience you'll bring", "experience you’ll bring",
            "main responsibilities and accountabilities", "other responsibilities",
            "skills and competencies", "strategic and growth focused activities",
            "recruitment delivery", "recruitment coordination and administration",
        }:
            return True
        if len(raw) <= 90 and raw == raw.title() and not re.search(r"[.!?]$", raw):
            return True
        return False

    def is_bullet_like(line: str) -> bool:
        raw = normalize_quotes(line).strip()
        if re.match(r"^[-*•]\s+", raw):
            return True
        if len(raw) <= 240 and not raw.endswith(".") and not raw.endswith(":") and raw[:1].isupper():
            return True
        return False

    for line in lines:
        raw = normalize_quotes(line).strip()
        if not raw:
            continue

        if is_heading_line(raw):
            flush_bullets()
            heading = raw[:-1].strip() if raw.endswith(":") else raw
            html_parts.append(f"<b>{html_escape(heading)}</b>")
            continue

        cleaned_bullet = re.sub(r"^[-*•]\s+", "", raw).strip()
        if is_bullet_like(raw):
            bullet_buffer.append(cleaned_bullet)
            continue

        flush_bullets()
        html_parts.append(f"<p>{html_escape(raw)}</p>")

    flush_bullets()
    return "\n".join(html_parts).strip()


def build_skill_regex(skill: str) -> re.Pattern:
    normalized = skill.strip().lower()

    special_map = {
        "c++": r"(?<![A-Za-z0-9])c\+\+(?![A-Za-z0-9])",
        "c#": r"(?<![A-Za-z0-9])c#(?![A-Za-z0-9])",
        ".net": r"(?<![A-Za-z0-9])(?:\.net|dotnet|dot net)(?![A-Za-z0-9])",
        "node.js": r"(?<![A-Za-z0-9])(?:node\.js|nodejs|node js)(?![A-Za-z0-9])",
        "next.js": r"(?<![A-Za-z0-9])(?:next\.js|nextjs|next js)(?![A-Za-z0-9])",
        "vue.js": r"(?<![A-Za-z0-9])(?:vue\.js|vuejs|vue js)(?![A-Za-z0-9])",
        "nuxt.js": r"(?<![A-Za-z0-9])(?:nuxt\.js|nuxtjs|nuxt js)(?![A-Za-z0-9])",
        "git": r"(?<![A-Za-z0-9])git(?![A-Za-z0-9])",
    }
    if normalized in special_map:
        return re.compile(special_map[normalized], flags=re.I)

    escaped = re.escape(skill)
    return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", flags=re.I)


def build_skills_source_text(role_body_text: str) -> str:
    plain = strip_html(role_body_text)
    lines = split_lines(plain)
    out = []

    disallow = [
        "smartrecruiters", "workday, inc.", "privacy policy", "read more", "follow us",
        "recruitment agencies", "why version 1?", "about the company", "company description",
        "benefits", "what we offer", "additional information", "equality, diversity and inclusion",
        "reasonable adjustments", "contact", "©", "uk & ireland's premier aws",
        "microsoft & oracle partner", "great place to work", "employee wellbeing",
        "annual excellence awards",
    ]

    for line in lines:
        low = line.lower()
        if any(x in low for x in disallow):
            continue
        out.append(line)

    return clean_whitespace("\n".join(out))


def exact_match_skills_in_order(description: str, skill_list: List[str], limit: int = 10) -> List[str]:
    if not description or not skill_list:
        return []

    found = []
    for skill in skill_list:
        regex = build_skill_regex(skill)
        match = regex.search(description)
        if match:
            index = match.start()
            if not any(e["skill"].lower() == skill.lower() for e in found):
                found.append({"skill": skill, "index": index})

    found.sort(key=lambda x: x["index"])
    return [x["skill"] for x in found[:limit]]


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
        "step2_classifiers_modularized",
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
