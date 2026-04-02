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

CONTRACT_PATTERNS = {
    "Permanent": [
        r"\bpermanent\b",
        r"\bfull[\s-]?time\b",
        r"\bfull time\b",
        r"\bemployment type\b.{0,20}\bpermanent\b",
        r"\bemployment type\b.{0,40}\bfull[\s-]?time\b",
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
        r"\b12[\s-]?month contract\b",
        r"\b6[\s-]?month contract\b",
    ],
}

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
    r"(?im)^\s*locations\s*[:\-]?\s*(.+?)\s*$",
    r"(?im)^\s*all locations\s*[:\-]?\s*(.+?)\s*$",
    r"(?im)^\s*office location\s*[:\-]?\s*(.+?)\s*$",
]

WORKPLACE_LABEL_PATTERNS = [
    r"(?im)^\s*workplace type\s*[:\-]?\s*(.+?)\s*$",
    r"(?im)^\s*working pattern\s*[:\-]?\s*(.+?)\s*$",
    r"(?im)^\s*work type\s*[:\-]?\s*(.+?)\s*$",
    r"(?im)^\s*location type\s*[:\-]?\s*(.+?)\s*$",
    r"(?im)^\s*remote type\s*[:\-]?\s*(.+?)\s*$",
    r"(?im)^\s*remote status\s*[:\-]?\s*(.+?)\s*$",
    r"(?im)^\s*workplace\s*[:\-]?\s*(.+?)\s*$",
]

NOISE_LOCATION_PATTERNS = [
    r"reporting to",
    r"department",
    r"employment type",
    r"salary",
    r"key responsibilities",
    r"skills, knowledge",
    r"about the role",
    r"internal advertising end date",
    r"careers site advertising end date",
    r"required field",
]

LOCATION_FIELD_LABELS = {
    "location",
    "job location",
    "based in",
    "locations",
    "all locations",
    "office location",
}

WORKPLACE_FIELD_LABELS = {
    "workplace type",
    "working pattern",
    "work type",
    "location type",
    "remote type",
    "remote status",
    "workplace",
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


def get_primary_text_window(text: str, max_chars: int = 7000) -> str:
    text = text or ""

    cut_markers = [
        r"(?i)\bour hiring process\b",
        r"(?i)\bother jobs\b",
        r"(?i)\bsimilar jobs\b",
        r"(?i)\byou may also like\b",
        r"(?i)\bmore jobs\b",
        r"(?i)\brelated jobs\b",
    ]
    for pattern in cut_markers:
        m = re.search(pattern, text)
        if m:
            text = text[:m.start()]
            break

    if len(text) > max_chars:
        text = text[:max_chars]

    return text.strip()


def _normalize_field_label(line: str) -> str:
    line = (line or "").strip()
    line = re.sub(r"[*#`_]+", "", line)
    line = line.replace("：", ":")
    line = re.sub(r"\s+", " ", line).strip(" :-")
    return line.lower()


def _extract_labeled_values(text: str, patterns: list[str]) -> list[str]:
    values = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            val = (match.group(1) or "").strip()
            if val:
                values.append(val)
    return dedupe_keep_order(values)


def _extract_field_values_from_lines(text: str, allowed_labels: set[str]) -> list[str]:
    values = []
    lines = split_lines(text)

    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        normalized = _normalize_field_label(line)

        for label in allowed_labels:
            if normalized == label:
                for next_idx in range(idx + 1, min(idx + 5, len(lines))):
                    nxt = lines[next_idx].strip()
                    nxt_norm = _normalize_field_label(nxt)
                    if not nxt:
                        continue
                    if nxt_norm in LOCATION_FIELD_LABELS or nxt_norm in WORKPLACE_FIELD_LABELS:
                        break
                    if _looks_like_noise_location_value(nxt):
                        continue
                    values.append(nxt)
                    break
                continue

            m = re.match(rf"(?i)^\**\s*{re.escape(label)}\s*[:\-]\s*(.+?)\s*\**$", line)
            if m:
                val = (m.group(1) or "").strip()
                if val:
                    values.append(val)

    return dedupe_keep_order(values)


def _clean_location_value(value: str) -> str:
    value = value or ""
    value = re.sub(r"[*#`_]+", "", value)
    value = value.replace("：", ":")

    value = re.split(
        r"(?i)\b(?:salary|job type|employment type|workplace type|remote status|reporting to|department)\b\s*[:\-]",
        value,
        maxsplit=1,
    )[0]

    value = re.sub(
        r"(?i)\((?:[^)]*\bhybrid\b[^)]*|[^)]*\bremote\b[^)]*|[^)]*\bonsite\b[^)]*|[^)]*\bhome\b[^)]*|[^)]*\bclient sites?\b[^)]*)\)",
        "",
        value,
    )

    value = re.sub(r"(?i)\bhybrid\s*:\s*home\s*,?\s*client sites?\b", "", value)
    value = re.sub(r"(?i)\bremote status\s*:\s*.+$", "", value)
    value = re.sub(r"(?i)\bworkplace type\s*:\s*.+$", "", value)

    value = re.sub(r"^\s*:\s*", "", value)
    value = re.sub(r"\s*\|\s*", ", ", value)
    value = re.sub(r"\s+", " ", value).strip(" -,:;")
    return value.strip()


def _looks_like_noise_location_value(value: str) -> bool:
    v = lower_text(value)
    if any(re.search(p, v) for p in NOISE_LOCATION_PATTERNS):
        return True
    if len(v.split()) > 10:
        return True
    if re.search(r"[.!?]", value):
        return True
    return False


def _split_possible_locations(raw: str) -> list[str]:
    raw = _clean_location_value(raw)
    if not raw:
        return []

    pieces = [raw]

    for part in re.split(r"[|;/]", raw):
        part = _clean_location_value(part)
        if part:
            pieces.append(part)

    comma_parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(comma_parts) >= 2:
        pieces.append(", ".join(comma_parts[:2]))
        pieces.append(", ".join(comma_parts[-2:]))
        pieces.append(comma_parts[0])
        pieces.append(comma_parts[-1])

    return dedupe_keep_order([p for p in pieces if p and not _looks_like_noise_location_value(p)])


def extract_location_candidates(text: str, predefined_locations: list[str]) -> list[str]:
    primary = get_primary_text_window(text)

    labeled_values = []
    labeled_values.extend(_extract_labeled_values(primary, LOCATION_LABEL_PATTERNS))
    labeled_values.extend(_extract_field_values_from_lines(primary, LOCATION_FIELD_LABELS))
    labeled_values = dedupe_keep_order(labeled_values)

    matches = []
    for value in labeled_values:
        matches.extend(_split_possible_locations(value))

    return dedupe_keep_order(matches)


def select_best_location(location_candidates: list[str]) -> str:
    if not location_candidates:
        return ""

    def score(loc: str) -> tuple[int, int]:
        loc_l = lower_text(loc)
        broad_exact = {
            "united kingdom", "uk", "england", "scotland",
            "wales", "northern ireland", "ireland", "europe", "emea"
        }
        if loc_l in broad_exact:
            return (0, len(loc))
        if "," in loc:
            return (2, len(loc))
        return (1, len(loc))

    ranked = sorted(location_candidates, key=score, reverse=True)
    return ranked[0]


def extract_remote_preferences(text: str) -> list[str]:
    primary = get_primary_text_window(text)
    found = set()

    labeled_values = []
    labeled_values.extend(_extract_labeled_values(primary, WORKPLACE_LABEL_PATTERNS))
    labeled_values.extend(_extract_field_values_from_lines(primary, WORKPLACE_FIELD_LABELS))

    location_values = []
    location_values.extend(_extract_labeled_values(primary, LOCATION_LABEL_PATTERNS))
    location_values.extend(_extract_field_values_from_lines(primary, LOCATION_FIELD_LABELS))

    combined_labeled_blob = " | ".join(labeled_values + location_values).lower()
    t = lower_text(primary)

    explicit_hybrid_patterns = [
        r"\bworkplace type\s*[:\-]?\s*hybrid\b",
        r"\bworking pattern\s*[:\-]?\s*hybrid\b",
        r"\bremote status\s*[:\-]?\s*hybrid\b",
        r"\bhybrid working\b",
        r"\bhybrid role\b",
        r"\bhybrid:\s*home\b",
        r"\bhybrid\b",
    ]
    explicit_remote_patterns = [
        r"\bworkplace type\s*[:\-]?\s*remote\b",
        r"\bremote status\s*[:\-]?\s*remote\b",
        r"\bfully remote\b",
        r"\b100% remote\b",
        r"\bremote[- ]first\b",
        r"\bthis is a remote role\b",
        r"\bthis role is remote\b",
        r"\bhome[- ]based\b",
        r"\bwork from home\b",
        r"\bwfh\b",
    ]
    explicit_onsite_patterns = [
        r"\bworkplace type\s*[:\-]?\s*onsite\b",
        r"\bremote status\s*[:\-]?\s*onsite\b",
        r"\bthis is an onsite role\b",
        r"\bthis role is onsite\b",
        r"\boffice[- ]based\b",
        r"\bfully office based\b",
        r"\bon[- ]site\b",
        r"\bonsite\b",
    ]

    if any(re.search(p, combined_labeled_blob) for p in explicit_hybrid_patterns):
        found.add("hybrid")

    if any(re.search(p, combined_labeled_blob) for p in explicit_remote_patterns):
        found.add("remote")

    if any(re.search(p, combined_labeled_blob) for p in explicit_onsite_patterns):
        found.add("onsite")

    if re.search(r"\bhome\b", combined_labeled_blob) and re.search(r"\bclient sites?\b|\boffice\b", combined_labeled_blob):
        found.add("hybrid")

    if any(re.search(p, t) for p in explicit_hybrid_patterns):
        found.add("hybrid")

    if any(re.search(p, t) for p in explicit_remote_patterns):
        found.add("remote")

    if any(re.search(p, t) for p in explicit_onsite_patterns):
        found.add("onsite")

    if re.search(r"\b\d(?:\s*-\s*\d)?\s+days?\s+(?:in|from)\s+the\s+office\b", t) or re.search(r"\b\d(?:\s*-\s*\d)?\s+days?\s+in office\b", t):
        found.add("hybrid")

    if "hybrid" in found and "remote" in found:
        if not re.search(r"\bfully remote\b|\b100% remote\b|\bremote[- ]first\b", t):
            found.discard("remote")

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

    if re.search(r"\bpart[\s-]?time\b", t) or re.search(r"\bjob[\s-]?share\b", t):
        return "Part Time"

    if re.search(r"\bfixed[\s-]?term\b|\btemporary\b|\bmaternity cover\b|\bmaternity leave\b", t):
        return "FTC"

    if re.search(r"\bfreelance\b|\bcontract role\b|\bcontract position\b|\bcontractor\b|\bcontracting\b", t):
        return "Freelance/Contract"

    if re.search(r"\b(?:12|6|3)[\s-]?month contract\b", t):
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
        "united kingdom", "uk", "england", "scotland", "wales", "northern ireland"
    ]
    broad_remote_markers = ["europe", "emea", "ireland"]

    disallowed_country_markers = [
        "spain", "france", "germany", "italy", "portugal", "poland", "latvia",
        "lithuania", "estonia", "romania", "bulgaria", "croatia", "serbia",
        "bosnia", "netherlands", "belgium", "sweden", "norway", "denmark",
        "finland", "switzerland", "austria", "czech", "slovakia", "hungary",
        "greece", "turkey", "india", "singapore", "japan", "china", "australia",
        "canada", "united states", "usa"
    ]

    for loc in locations:
        loc_l = lower_text(loc)

        if any(marker in loc_l for marker in uk_markers):
            return True

        if any(marker in loc_l for marker in broad_remote_markers) and "remote" in remote_set:
            return True

        if "," not in loc and 1 <= len(loc.split()) <= 4:
            if not any(marker in loc_l for marker in disallowed_country_markers):
                return True

        if "ireland" in loc_l and "remote" in remote_set:
            return True

    return False
