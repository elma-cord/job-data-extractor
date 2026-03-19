import re
from typing import Dict, List

from fetch_extract import normalize_common_location_aliases


def clean_whitespace(text: str) -> str:
    text = text or ""
    text = text.replace("\xa0", " ")
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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
    )


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


def canonical_label(text: str) -> str:
    s = normalize_quotes(text or "").lower().strip()
    s = s.replace("&", "and")
    s = s.replace("/", "")
    s = s.replace("\\", "")
    s = s.replace(",", "")
    s = s.replace("-", "")
    s = re.sub(r"\s+", "", s)
    return s


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
    if not allowed_locations or not text:
        return ""

    city_lookup = build_location_lookup(allowed_locations)
    candidate_lines = gather_location_lines(text)
    if not candidate_lines:
        return ""

    matches = []

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


def parse_money_range_from_line(line: str) -> tuple[str, str, str, str]:
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


def parse_explicit_salary(text: str, _allowed_salaries_unused: List[int]) -> tuple[str, str, str, str]:
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
