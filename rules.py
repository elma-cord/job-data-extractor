import csv
import re
from pathlib import Path
from typing import Optional


CURRENCY_SIGNS = {
    "£": "GBP",
    "$": "USD",
    "€": "EUR",
}

REMOTE_ORDER = ["onsite", "hybrid", "remote"]

LEADERSHIP_TERMS = [
    "head of",
    "director",
    "vice president",
    "vp ",
    "chief ",
    "cfo",
    "cto",
    "coo",
    "cio",
    "cmo",
    "cro",
    "cpo",
    "engineering manager",
]

MID_TERMS = ["mid", "mid-level", "intermediate"]
JUNIOR_TERMS = ["junior", "jr", "entry level", "graduate", "associate"]
SENIOR_TERMS = ["senior", "sr", "staff", "principal"]
LEAD_TERMS = ["lead", "team lead"]

IRRELEVANT_ROLE_HINTS = [
    "teacher", "nurse", "waiter", "chef", "warehouse", "cleaner", "receptionist",
    "construction", "civil engineer", "mechanical", "electrical", "manufacturing",
    "beauty therapist", "maritime", "microbiology", "injection moulding", "retail assistant",
]

TP_HINTS = [
    "engineer", "developer", "software", "data", "product", "ux", "ui", "devops",
    "site reliability", "security", "qa", "automation", "backend", "back end",
    "front end", "frontend", "full stack", "it support", "infrastructure",
    "architect", "technical", "support engineer", "line support", "systems engineer",
    "network engineer", "platform", "cloud", "sre", "helpdesk",
]

BUSINESS_RELEVANT_HINTS = [
    "account executive", "account manager", "account director",
    "sales", "business development", "customer success",
    "recruiter", "recruitment", "talent acquisition", "talent partner",
    "operations", "finance", "marketing", "consultant",
]

LOCATION_LABEL_PATTERNS = [
    r"(?im)^\s*location\s*[:\-]?\s*(.+?)\s*$",
    r"(?im)^\s*job location\s*[:\-]?\s*(.+?)\s*$",
    r"(?im)^\s*based in\s*[:\-]?\s*(.+?)\s*$",
]

WORKPLACE_LABEL_PATTERNS = [
    r"(?im)^\s*workplace type\s*[:\-]?\s*(.+?)\s*$",
    r"(?im)^\s*working pattern\s*[:\-]?\s*(.+?)\s*$",
    r"(?im)^\s*work type\s*[:\-]?\s*(.+?)\s*$",
    r"(?im)^\s*location type\s*[:\-]?\s*(.+?)\s*$",
    r"(?im)^\s*remote type\s*[:\-]?\s*(.+?)\s*$",
]

CONTRACT_PATTERNS = {
    "Permanent": [
        r"\bpermanent\b",
        r"\bfull[\s-]?time\b",
        r"\bjob type\s*[:\-]?\s*full[\s-]?time\b",
        r"\bemployment type\s*[:\-]?\s*permanent\b",
        r"\bemployment type\s*[:\-]?\s*permanent\s*-\s*full[\s-]?time\b",
    ],
    "FTC": [
        r"\bfixed[\s-]?term\b",
        r"\btemporary\b",
        r"\bmaternity cover\b",
        r"\bmaternity leave\b",
    ],
    "Part Time": [
        r"\bpart[\s-]?time\b",
        r"\bjob[\s-]?share\b",
    ],
    "Freelance/Contract": [
        r"\bfreelance\b",
        r"\bcontract role\b",
        r"\bcontract position\b",
        r"\bcontractor\b",
        r"\bcontracting\b",
        r"\b(?:3|6|12)[\s-]?month contract\b",
    ],
}


def load_single_column_csv(path: Path) -> list[str]:
    values = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            value = row[0].strip()
            if value and value.lower() not in {"job title", "location", "skill"}:
                values.append(value)
    return values


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def lower_text(text: str) -> str:
    return normalize_text(text).lower()


def split_lines(text: str) -> list[str]:
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def get_primary_text_window(text: str, max_chars: int = 4500) -> str:
    text = text or ""

    cut_markers = [
        r"(?i)\bour hiring process\b",
        r"(?i)\bother jobs\b",
        r"(?i)\bsimilar jobs\b",
        r"(?i)\byou may also like\b",
        r"(?i)\bmore jobs\b",
        r"(?i)\brelated jobs\b",
        r'(?i)apply for this role',
        r'(?i)indicates required fields',
        r'(?i)we offer a free, no obligation consultation',
    ]
    for pattern in cut_markers:
        m = re.search(pattern, text)
        if m:
            text = text[:m.start()]
            break

    if len(text) > max_chars:
        text = text[:max_chars]

    return text.strip()


def _extract_labeled_values(text: str, patterns: list[str]) -> list[str]:
    values = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            val = (match.group(1) or "").strip()
            if val:
                values.append(val)
    return dedupe_keep_order(values)


def _clean_location_value(value: str) -> str:
    value = value or ""
    value = re.sub(r"\s*\|\s*", ", ", value)
    value = re.sub(r"\s*/\s*", " / ", value)
    value = re.sub(r"\s+", " ", value).strip(" -,:;")
    return value.strip()


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", lower_text(text))


def _country_alias(country: str) -> list[str]:
    c = lower_text(country)
    if c == "united kingdom":
        return ["united kingdom", "uk", "great britain"]
    return [country]


def _build_location_alias_map(predefined_locations: list[str]) -> dict[str, list[str]]:
    alias_map: dict[str, list[str]] = {}

    def add(alias: str, target: str) -> None:
        key = _compact(alias)
        if not key:
            return
        alias_map.setdefault(key, [])
        if target not in alias_map[key]:
            alias_map[key].append(target)

    for loc in predefined_locations:
        add(loc, loc)

        parts = [p.strip() for p in loc.split(",") if p.strip()]
        if not parts:
            continue

        first = parts[0]
        add(first, loc)

        if len(parts) >= 2:
            country = parts[-1]
            add(f"{first}, {country}", loc)
            for ca in _country_alias(country):
                add(f"{first}, {ca}", loc)

        if len(parts) >= 3:
            mid = ", ".join(parts[:-1])
            add(mid, loc)

    return alias_map


def _score_location_match(raw_value: str, target: str) -> tuple[int, int, int]:
    raw_l = lower_text(raw_value)
    target_l = lower_text(target)
    raw_compact = _compact(raw_value)
    target_compact = _compact(target)

    exact = 1 if raw_l == target_l else 0
    starts = 1 if target_l.startswith(raw_l + ",") else 0
    compact_exact = 1 if raw_compact == target_compact else 0

    parts = [p.strip() for p in target.split(",") if p.strip()]
    specificity = len(parts)

    return (exact, compact_exact + starts, specificity)


def match_best_predefined_location(raw_value: str, predefined_locations: list[str]) -> str:
    raw_value = _clean_location_value(raw_value)
    if not raw_value:
        return ""

    alias_map = _build_location_alias_map(predefined_locations)
    raw_key = _compact(raw_value)

    candidates = alias_map.get(raw_key, []).copy()

    if not candidates:
        raw_l = lower_text(raw_value)
        first_token = raw_l.split(",")[0].strip()
        for loc in predefined_locations:
            loc_l = lower_text(loc)
            if loc_l == raw_l or loc_l.startswith(raw_l + ","):
                candidates.append(loc)
            elif first_token and (loc_l == first_token or loc_l.startswith(first_token + ",")):
                candidates.append(loc)

    candidates = dedupe_keep_order(candidates)
    if not candidates:
        return ""

    ranked = sorted(
        candidates,
        key=lambda loc: _score_location_match(raw_value, loc),
        reverse=True,
    )
    return ranked[0]


def extract_location_candidates(text: str, predefined_locations: list[str]) -> list[str]:
    """
    Return ONLY predefined-list locations.
    Strong priority:
    1. first labeled Location field
    2. other labeled location fields
    3. top header lines
    4. broader primary text scan
    """
    primary = get_primary_text_window(text)

    labeled_values = _extract_labeled_values(primary, LOCATION_LABEL_PATTERNS)
    labeled_values = [_clean_location_value(v) for v in labeled_values if _clean_location_value(v)]

    strong_matches = []
    for value in labeled_values:
        best = match_best_predefined_location(value, predefined_locations)
        if best:
            strong_matches.append(best)

    if strong_matches:
        return dedupe_keep_order(strong_matches)

    header_lines = split_lines(primary)[:18]
    header_blob = "\n".join(header_lines)

    header_matches = []
    for line in header_lines:
        best = match_best_predefined_location(line, predefined_locations)
        if best:
            header_matches.append(best)

    if header_matches:
        return dedupe_keep_order(header_matches)

    broad_matches = []
    for loc in predefined_locations:
        loc_l = lower_text(loc)
        first = loc_l.split(",")[0].strip()
        if not first:
            continue
        if re.search(rf"(?<![a-z]){re.escape(first)}(?![a-z])", lower_text(header_blob)):
            broad_matches.append(loc)

    if broad_matches:
        ranked = sorted(
            dedupe_keep_order(broad_matches),
            key=lambda loc: ("," in loc, len(loc)),
            reverse=True,
        )
        return ranked

    return []


def select_best_location(location_candidates: list[str]) -> str:
    if not location_candidates:
        return ""

    def score(loc: str) -> tuple[int, int]:
        parts = [p.strip() for p in loc.split(",") if p.strip()]
        broad_exact = {
            "united kingdom", "uk", "england", "scotland",
            "wales", "northern ireland", "ireland", "europe", "emea"
        }
        if lower_text(loc) in broad_exact:
            return (0, len(parts))
        return (1, len(parts))

    return sorted(location_candidates, key=score, reverse=True)[0]


def extract_remote_preferences(text: str) -> list[str]:
    primary = get_primary_text_window(text)
    found = set()

    labeled_values = _extract_labeled_values(primary, WORKPLACE_LABEL_PATTERNS)
    labeled_blob = " | ".join(labeled_values).lower()

    if labeled_blob:
        if "hybrid" in labeled_blob:
            found.add("hybrid")
        if re.search(r"\bremote\b|\bwork from home\b|\bwfh\b|\bhome\b", labeled_blob):
            found.add("remote")
        if re.search(r"\bonsite\b|\bon-site\b", labeled_blob):
            found.add("onsite")
        if found:
            return [x for x in REMOTE_ORDER if x in found]

    top = lower_text("\n".join(split_lines(primary)[:12]))

    if re.search(r"\bhybrid\b", top):
        found.add("hybrid")
    if re.search(r"\bfully remote\b|\b100% remote\b|\bremote\b|\bwork from home\b|\bwfh\b", top):
        found.add("remote")

    onsite_patterns = [
        r"\bthis is an onsite role\b",
        r"\bthis role is onsite\b",
        r"\boffice[- ]based\b",
        r"\bfully office based\b",
        r"\b5 days (?:a week|per week)? in the office\b",
    ]
    if not found.intersection({"hybrid", "remote"}):
        if any(re.search(p, top) for p in onsite_patterns):
            found.add("onsite")

    return [x for x in REMOTE_ORDER if x in found]


def extract_remote_days(text: str) -> str:
    primary = lower_text(get_primary_text_window(text))

    if re.search(r"\bfully remote\b|\b100% remote\b|\bremote every day\b", primary):
        return "not specified"

    office_patterns = [
        r"\b(\d)(?:\s*-\s*(\d))?\s+days?\s+(?:in|from)\s+the\s+office\b",
        r"\b(\d)(?:\s*-\s*(\d))?\s+days?\s+in office\b",
    ]
    for pattern in office_patterns:
        m = re.search(pattern, primary)
        if m:
            start = int(m.group(1))
            end = int(m.group(2) or m.group(1))
            return str(max(5 - start, 5 - end))

    remote_patterns = [
        r"\b(\d)(?:\s*-\s*(\d))?\s+days?\s+(?:remote|from home|wfh)\b",
        r"\b(\d)(?:\s*-\s*(\d))?\s+days?\s+working from home\b",
    ]
    for pattern in remote_patterns:
        m = re.search(pattern, primary)
        if m:
            start = int(m.group(1))
            end = int(m.group(2) or m.group(1))
            return str(max(start, end))

    return "not specified"


def detect_contract_type(text: str) -> str:
    t = lower_text(get_primary_text_window(text))

    if re.search(r"\bpermanent\b", t) or re.search(r"\bfull[\s-]?time\b", t):
        return "Permanent"

    if re.search(r"\bpart[\s-]?time\b|\bjob[\s-]?share\b", t):
        return "Part Time"

    if re.search(r"\bfixed[\s-]?term\b|\btemporary\b|\bmaternity cover\b|\bmaternity leave\b", t):
        return "FTC"

    if re.search(r"\bfreelance\b|\bcontract role\b|\bcontract position\b|\bcontractor\b|\bcontracting\b", t):
        return "Freelance/Contract"

    if re.search(r"\b(?:3|6|12)[\s-]?month contract\b", t):
        return "Freelance/Contract"

    return ""


def detect_seniority_from_title_and_description(position_name: str, description: str) -> list[str]:
    title = lower_text(position_name)
    desc = lower_text(description)
    full = f"{title} {desc}"

    if any(term in title for term in LEADERSHIP_TERMS):
        return ["leadership"]

    out = []

    if any(term in title for term in JUNIOR_TERMS):
        out.append("junior")
    if any(term in title for term in MID_TERMS):
        out.append("mid")
    if any(term in title for term in SENIOR_TERMS):
        out.append("senior")
    if any(term in title for term in LEAD_TERMS):
        out.append("lead")

    exp_match = re.search(r"\b(\d+)\s*(?:\+|plus)?\s+years?\b", full)
    if exp_match:
        years = int(exp_match.group(1))
        if years <= 1:
            out.extend(["entry", "junior"])
        elif years == 2:
            out.extend(["junior", "mid"])
        elif 3 <= years <= 5:
            out.extend(["senior"])
        elif years > 5:
            out.extend(["senior", "lead"])

    if re.search(r"\bmanage\b|\bmanagement\b|\bline manager\b|\bteam leadership\b|\bmentor\b|\bownership\b", full):
        out.append("lead")

    if not out:
        if "manager" in title:
            out.extend(["senior", "lead"])
        elif "consultant" in title:
            out.append("mid")
        elif "engineer" in title or "developer" in title or "analyst" in title:
            out.append("mid")
        elif "executive" in title:
            out.append("mid")

    order = ["entry", "junior", "mid", "senior", "lead", "leadership"]
    out = [x for x in order if x in out]
    return dedupe_keep_order(out)[:3]


def detect_basic_relevance_from_title(position_name: str) -> tuple[str, str]:
    title = lower_text(position_name)

    if any(bad in title for bad in IRRELEVANT_ROLE_HINTS):
        return "Not Relevant", "Title clearly points to an excluded non-target function."

    if any(tp in title for tp in TP_HINTS):
        return "Relevant", "Title appears to match target tech/product scope."

    if any(h in title for h in BUSINESS_RELEVANT_HINTS):
        return "Relevant", "Title appears to match target business scope."

    return "", ""


def detect_tp_from_title(position_name: str) -> str:
    title = lower_text(position_name)
    if any(tp in title for tp in TP_HINTS):
        return "T&P job"
    if title:
        return "Not T&P"
    return ""


def extract_salary(text: str, allowed_salaries: list[int]) -> dict:
    t = text or ""
    lower = lower_text(t)

    salary_context_patterns = [
        r"\bsalary\b",
        r"\bcompensation\b",
        r"\bpay\b",
        r"\bpackage\b",
        r"£\s*\d",
        r"\bGBP\b",
        r"\bUSD\b",
        r"\bEUR\b",
        r"\bCAD\b",
        r"\bAUD\b",
        r"\bCHF\b",
        r"\bper year\b",
        r"\bper annum\b",
        r"\bannually\b",
        r"\b\d+\s*k\b",
    ]
    if not any(re.search(p, lower, re.I) for p in salary_context_patterns):
        return {"salary_min": "", "salary_max": "", "salary_currency": ""}

    currency = ""
    for sign, code in CURRENCY_SIGNS.items():
        if sign in t:
            currency = code
            break
    if not currency:
        m = re.search(r"\b(GBP|USD|EUR|CAD|AUD|CHF)\b", t, re.I)
        if m:
            currency = m.group(1).upper()

    candidates = []

    patterns = [
        r"(?i)(?:salary|pay|package)?\s*[:\-]?\s*[£$€]\s*(\d{2,3}(?:,\d{3})+|\d{2,3})\s*(?:-\s*[£$€]?\s*(\d{2,3}(?:,\d{3})+|\d{2,3}))?\s*(k)?",
        r"(?i)(\d{2,3}(?:,\d{3})+|\d{2,3})\s*(?:-\s*(\d{2,3}(?:,\d{3})+|\d{2,3}))?\s*(k)\b",
        r"(?i)[£$€]\s*(\d{4,6})(?:\s*-\s*[£$€]?\s*(\d{4,6}))?",
    ]

    def parse_amount(raw: str, has_k: bool) -> int:
        val = int(raw.replace(",", ""))
        if has_k and val < 1000:
            val *= 1000
        return val

    for pattern in patterns:
        for m in re.finditer(pattern, t):
            g1 = m.group(1)
            g2 = m.group(2) if len(m.groups()) >= 2 else None
            g3 = m.group(3) if len(m.groups()) >= 3 else None

            if g1:
                v1 = parse_amount(g1, bool(g3))
                if 10000 <= v1 <= 500000:
                    candidates.append(v1)
            if g2:
                v2 = parse_amount(g2, bool(g3))
                if 10000 <= v2 <= 500000:
                    candidates.append(v2)

    candidates = sorted(set(candidates))
    if not candidates:
        return {"salary_min": "", "salary_max": "", "salary_currency": currency}

    min_val = candidates[0]
    max_val = candidates[1] if len(candidates) > 1 else candidates[0]

    def closest(val: int) -> int:
        return min(allowed_salaries, key=lambda x: abs(x - val))

    return {
        "salary_min": str(closest(min_val)),
        "salary_max": str(closest(max_val)),
        "salary_currency": currency,
    }


def extract_visa_status(text: str) -> str:
    t = lower_text(text)
    if re.search(r"\bvisa sponsorship\b|\bsponsor(?:ship)?\b", t):
        if re.search(r"\bno visa sponsorship\b|\bwithout sponsorship\b|\bcannot sponsor\b", t):
            return "no"
        return "yes"
    return ""


def location_or_remote_missing(description: str, predefined_locations: Optional[list[str]] = None) -> bool:
    locations = extract_location_candidates(description, predefined_locations or [])
    remote = extract_remote_preferences(description)
    return not (bool(locations) and bool(remote))


def infer_job_titles_from_position_name(position_name: str, allowed_job_titles: list[str]) -> list[str]:
    title = lower_text(position_name)
    allowed_lookup = {lower_text(x): x for x in allowed_job_titles}

    def existing(options: list[str]) -> list[str]:
        out = []
        for opt in options:
            key = lower_text(opt)
            if key in allowed_lookup:
                out.append(allowed_lookup[key])
        return dedupe_keep_order(out)

    if title in allowed_lookup:
        return [allowed_lookup[title]]

    alias_groups = [
        (
            ["recruitment consultant", "recruiter", "resource consultant", "talent acquisition", "talent partner"],
            ["Talent Acquisition", "Recruiter", "Talent Partner", "Recruitment Consultant"],
        ),
        (
            ["3rd line support engineer", "third line support engineer", "2nd line engineer", "2nd line support engineer", "support engineer", "technical support engineer"],
            ["Technical Support Engineer", "Support Engineer", "IT Support", "Infrastructure Engineer", "Systems Engineer"],
        ),
        (
            ["account director"],
            ["Account Director", "Account Executive", "Account Manager"],
        ),
        (
            ["account executive"],
            ["Account Executive", "Sales Executive", "Account Manager"],
        ),
        (
            ["back end", "backend"],
            ["Back End", "Software Engineer", "Back End Engineer"],
        ),
    ]

    for triggers, outputs in alias_groups:
        if any(trigger in title for trigger in triggers):
            mapped = existing(outputs)
            if mapped:
                return mapped[:3]

    contains_matches = []
    for allowed in allowed_job_titles:
        if lower_text(allowed) in title:
            contains_matches.append(allowed)

    return dedupe_keep_order(contains_matches)[:3]


def is_location_allowed(locations: list[str], remote_preferences: list[str]) -> bool:
    if not locations:
        return True

    remote_set = set(remote_preferences)

    uk_markers = [
        "United Kingdom", "UK", "England", "Scotland", "Wales", "Northern Ireland"
    ]
    for loc in locations:
        if any(marker in loc for marker in uk_markers):
            return True

    for loc in locations:
        if "Ireland" in loc and "remote" in remote_set:
            return True

    for loc in locations:
        if ("Europe" in loc or "EMEA" in loc) and "remote" in remote_set:
            return True

    return False
