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
]

WORKPLACE_LABEL_PATTERNS = [
    r"(?im)^\s*workplace type\s*[:\-]?\s*(.+?)\s*$",
    r"(?im)^\s*working pattern\s*[:\-]?\s*(.+?)\s*$",
    r"(?im)^\s*work type\s*[:\-]?\s*(.+?)\s*$",
    r"(?im)^\s*location type\s*[:\-]?\s*(.+?)\s*$",
    r"(?im)^\s*remote type\s*[:\-]?\s*(.+?)\s*$",
]

NOISE_LOCATION_PATTERNS = [
    r"reporting to",
    r"department",
    r"employment type",
    r"salary",
    r"key responsibilities",
    r"skills, knowledge",
    r"about the role",
]

# ---------------------------------------------------------------------------
# Built-in location knowledge — replaces predefined_locations CSV.
# Keys are lowercase canonical names; values are their display form.
# Covers UK cities/regions, Ireland, major EU cities, and broad regions.
# Longer / more specific keys are preferred during matching.
# ---------------------------------------------------------------------------
_KNOWN_LOCATIONS: dict[str, str] = {
    # ── UK regions / countries ───────────────────────────────────────────────
    "united kingdom": "United Kingdom",
    "uk": "UK",
    "england": "England",
    "scotland": "Scotland",
    "wales": "Wales",
    "northern ireland": "Northern Ireland",
    # ── UK cities ────────────────────────────────────────────────────────────
    "london, united kingdom": "London, UK",
    "london, uk": "London, UK",
    "london": "London, UK",
    "manchester, united kingdom": "Manchester, UK",
    "manchester, uk": "Manchester, UK",
    "manchester": "Manchester, UK",
    "birmingham, united kingdom": "Birmingham, UK",
    "birmingham, uk": "Birmingham, UK",
    "birmingham": "Birmingham, UK",
    "leeds, united kingdom": "Leeds, UK",
    "leeds, uk": "Leeds, UK",
    "leeds": "Leeds, UK",
    "bristol, united kingdom": "Bristol, UK",
    "bristol, uk": "Bristol, UK",
    "bristol": "Bristol, UK",
    "edinburgh, united kingdom": "Edinburgh, UK",
    "edinburgh, uk": "Edinburgh, UK",
    "edinburgh": "Edinburgh, UK",
    "glasgow, united kingdom": "Glasgow, UK",
    "glasgow, uk": "Glasgow, UK",
    "glasgow": "Glasgow, UK",
    "sheffield, united kingdom": "Sheffield, UK",
    "sheffield, uk": "Sheffield, UK",
    "sheffield": "Sheffield, UK",
    "liverpool, united kingdom": "Liverpool, UK",
    "liverpool, uk": "Liverpool, UK",
    "liverpool": "Liverpool, UK",
    "nottingham, united kingdom": "Nottingham, UK",
    "nottingham, uk": "Nottingham, UK",
    "nottingham": "Nottingham, UK",
    "oxford, united kingdom": "Oxford, UK",
    "oxford, uk": "Oxford, UK",
    "oxford": "Oxford, UK",
    "cambridge, united kingdom": "Cambridge, UK",
    "cambridge, uk": "Cambridge, UK",
    "cambridge": "Cambridge, UK",
    "reading, united kingdom": "Reading, UK",
    "reading, uk": "Reading, UK",
    "reading": "Reading, UK",
    "brighton, united kingdom": "Brighton, UK",
    "brighton, uk": "Brighton, UK",
    "brighton": "Brighton, UK",
    "cardiff, united kingdom": "Cardiff, UK",
    "cardiff, uk": "Cardiff, UK",
    "cardiff": "Cardiff, UK",
    "belfast, united kingdom": "Belfast, UK",
    "belfast, uk": "Belfast, UK",
    "belfast": "Belfast, UK",
    "newcastle upon tyne": "Newcastle, UK",
    "newcastle, united kingdom": "Newcastle, UK",
    "newcastle, uk": "Newcastle, UK",
    "newcastle": "Newcastle, UK",
    "coventry, united kingdom": "Coventry, UK",
    "coventry, uk": "Coventry, UK",
    "coventry": "Coventry, UK",
    "leicester, united kingdom": "Leicester, UK",
    "leicester, uk": "Leicester, UK",
    "leicester": "Leicester, UK",
    "southampton, united kingdom": "Southampton, UK",
    "southampton, uk": "Southampton, UK",
    "southampton": "Southampton, UK",
    "portsmouth, united kingdom": "Portsmouth, UK",
    "portsmouth, uk": "Portsmouth, UK",
    "portsmouth": "Portsmouth, UK",
    "milton keynes, united kingdom": "Milton Keynes, UK",
    "milton keynes, uk": "Milton Keynes, UK",
    "milton keynes": "Milton Keynes, UK",
    "guildford, united kingdom": "Guildford, UK",
    "guildford, uk": "Guildford, UK",
    "guildford": "Guildford, UK",
    "watford, united kingdom": "Watford, UK",
    "watford, uk": "Watford, UK",
    "watford": "Watford, UK",
    "exeter, united kingdom": "Exeter, UK",
    "exeter, uk": "Exeter, UK",
    "exeter": "Exeter, UK",
    "york, united kingdom": "York, UK",
    "york, uk": "York, UK",
    "york": "York, UK",
    "hull, united kingdom": "Hull, UK",
    "hull, uk": "Hull, UK",
    "hull": "Hull, UK",
    "stoke-on-trent, united kingdom": "Stoke-on-Trent, UK",
    "stoke-on-trent, uk": "Stoke-on-Trent, UK",
    "stoke-on-trent": "Stoke-on-Trent, UK",
    "stoke": "Stoke-on-Trent, UK",
    "derby, united kingdom": "Derby, UK",
    "derby, uk": "Derby, UK",
    "derby": "Derby, UK",
    "swansea, united kingdom": "Swansea, UK",
    "swansea, uk": "Swansea, UK",
    "swansea": "Swansea, UK",
    "aberdeen, united kingdom": "Aberdeen, UK",
    "aberdeen, uk": "Aberdeen, UK",
    "aberdeen": "Aberdeen, UK",
    "dundee, united kingdom": "Dundee, UK",
    "dundee, uk": "Dundee, UK",
    "dundee": "Dundee, UK",
    "bath, united kingdom": "Bath, UK",
    "bath, uk": "Bath, UK",
    "bath": "Bath, UK",
    "worcester, united kingdom": "Worcester, UK",
    "worcester, uk": "Worcester, UK",
    "worcester": "Worcester, UK",
    "norwich, united kingdom": "Norwich, UK",
    "norwich, uk": "Norwich, UK",
    "norwich": "Norwich, UK",
    "ipswich, united kingdom": "Ipswich, UK",
    "ipswich, uk": "Ipswich, UK",
    "ipswich": "Ipswich, UK",
    "peterborough, united kingdom": "Peterborough, UK",
    "peterborough, uk": "Peterborough, UK",
    "peterborough": "Peterborough, UK",
    "luton, united kingdom": "Luton, UK",
    "luton, uk": "Luton, UK",
    "luton": "Luton, UK",
    "swindon, united kingdom": "Swindon, UK",
    "swindon, uk": "Swindon, UK",
    "swindon": "Swindon, UK",
    "slough, united kingdom": "Slough, UK",
    "slough, uk": "Slough, UK",
    "slough": "Slough, UK",
    "warrington, united kingdom": "Warrington, UK",
    "warrington, uk": "Warrington, UK",
    "warrington": "Warrington, UK",
    "wigan, united kingdom": "Wigan, UK",
    "wigan, uk": "Wigan, UK",
    "wigan": "Wigan, UK",
    "bolton, united kingdom": "Bolton, UK",
    "bolton, uk": "Bolton, UK",
    "bolton": "Bolton, UK",
    # ── Ireland ───────────────────────────────────────────────────────────────
    "republic of ireland": "Ireland",
    "ireland": "Ireland",
    "dublin, ireland": "Dublin, Ireland",
    "dublin": "Dublin, Ireland",
    "cork, ireland": "Cork, Ireland",
    "cork": "Cork, Ireland",
    "galway, ireland": "Galway, Ireland",
    "galway": "Galway, Ireland",
    # ── Broad regions ─────────────────────────────────────────────────────────
    "europe": "Europe",
    "emea": "EMEA",
    "remote": "Remote",
    "worldwide": "Worldwide",
    "global": "Global",
    # ── Major EU / international cities frequently seen in UK job ads ──────────
    "amsterdam, netherlands": "Amsterdam, Netherlands",
    "amsterdam": "Amsterdam, Netherlands",
    "berlin, germany": "Berlin, Germany",
    "berlin": "Berlin, Germany",
    "paris, france": "Paris, France",
    "paris": "Paris, France",
    "madrid, spain": "Madrid, Spain",
    "madrid": "Madrid, Spain",
    "barcelona, spain": "Barcelona, Spain",
    "barcelona": "Barcelona, Spain",
    "lisbon, portugal": "Lisbon, Portugal",
    "lisbon": "Lisbon, Portugal",
    "new york, usa": "New York, USA",
    "new york": "New York, USA",
    "san francisco, usa": "San Francisco, USA",
    "san francisco": "San Francisco, USA",
    "toronto, canada": "Toronto, Canada",
    "toronto": "Toronto, Canada",
    "sydney, australia": "Sydney, Australia",
    "sydney": "Sydney, Australia",
}

# Pre-build a sorted list of keys from longest to shortest so that more
# specific aliases (e.g. "london, united kingdom") are tried before shorter
# ones (e.g. "london").
_KNOWN_LOCATION_KEYS_SORTED: list[str] = sorted(
    _KNOWN_LOCATIONS.keys(), key=len, reverse=True
)


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


def _looks_like_noise_location_value(value: str) -> bool:
    v = lower_text(value)
    return any(re.search(p, v) for p in NOISE_LOCATION_PATTERNS)


def _extract_locations_from_text(text: str) -> list[str]:
    """
    Extract location names from *text* without requiring a predefined CSV.

    Strategy:
    1. Match against the built-in _KNOWN_LOCATIONS dictionary using whole-word
       regex, longest alias first so specific forms beat bare city names.
    2. Additionally capture freeform title-case "City, Country" patterns not
       already covered by the dictionary.

    Returns a deduplicated list of display-form location strings.
    """
    t_lower = lower_text(text)
    found: list[tuple[int, str]] = []  # (alias_length, display_value)

    # ── Step 1: match against built-in dictionary ─────────────────────────
    for key in _KNOWN_LOCATION_KEYS_SORTED:
        escaped = re.escape(key)
        pattern = r"(?<![a-z])" + escaped + r"(?![a-z])"
        if re.search(pattern, t_lower):
            display = _KNOWN_LOCATIONS[key]
            found.append((len(key), display))

    # ── Step 2: capture freeform "Word(s), Word(s)" city/country patterns ─
    # Matches things like "Austin, Texas" or "Cape Town, South Africa" that
    # are not in the dictionary. We require title-case to reduce false hits.
    freeform_pattern = (
        r"\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?)"
        r",\s*"
        r"([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?)\b"
    )
    for m in re.finditer(freeform_pattern, text):
        candidate = f"{m.group(1)}, {m.group(2)}"
        candidate_lower = lower_text(candidate)
        if candidate_lower not in _KNOWN_LOCATIONS:
            if not _looks_like_noise_location_value(candidate):
                found.append((len(candidate_lower), candidate))

    # Sort: longest (most specific) first; stable so dict entries beat
    # freeform ones of equal length.
    found.sort(key=lambda x: -x[0])
    return dedupe_keep_order([v for _, v in found])


def extract_location_candidates(text: str) -> list[str]:
    """
    Return location strings extracted directly from *text*.
    No predefined CSV is required.

    Search priority (stops as soon as results are found):
    1. Explicit label fields  (e.g. "Location: London")
    2. First 20 lines / header block of the primary section
    3. Full primary section scan  ← last resort
    """
    primary = get_primary_text_window(text)
    matches: list[str] = []

    # ── Priority 1: labeled location fields ──────────────────────────────
    labeled_values = _extract_labeled_values(primary, LOCATION_LABEL_PATTERNS)
    for value in labeled_values:
        cleaned = _clean_location_value(value)
        if cleaned and not _looks_like_noise_location_value(cleaned):
            hits = _extract_locations_from_text(cleaned)
            if hits:
                matches.extend(hits)
            else:
                # Labeled field found but not in dictionary — keep raw value.
                matches.append(cleaned)

    if matches:
        return dedupe_keep_order(matches)

    # ── Priority 2: top / header lines only (first 20 lines) ─────────────
    top_lines = split_lines(primary)[:20]
    header_blob = "\n".join(top_lines)
    matches.extend(_extract_locations_from_text(header_blob))
    if matches:
        return dedupe_keep_order(matches)

    # ── Priority 3: full primary section scan ─────────────────────────────
    matches.extend(_extract_locations_from_text(primary))
    return dedupe_keep_order(matches)


def select_best_location(location_candidates: list[str]) -> str:
    """
    Pick the single most specific / trustworthy location from the candidates.

    Scoring (higher = better):
      2 + length  → city + country form  e.g. "London, UK"
      1 + length  → city-only or region  e.g. "London"
      0 + length  → broad region only    e.g. "United Kingdom", "Europe"
    """
    if not location_candidates:
        return ""

    broad_exact = {
        "united kingdom", "uk", "england", "scotland",
        "wales", "northern ireland", "ireland", "europe", "emea",
        "remote", "worldwide", "global",
    }

    def score(loc: str) -> tuple[int, int]:
        loc_l = lower_text(loc)
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

    labeled_values = _extract_labeled_values(primary, WORKPLACE_LABEL_PATTERNS)
    labeled_blob = " | ".join(labeled_values).lower()

    if labeled_blob:
        if "hybrid" in labeled_blob:
            found.add("hybrid")
        if re.search(r"\bremote\b|\bwork from home\b|\bwfh\b", labeled_blob):
            found.add("remote")
        if re.search(r"\bonsite\b|\bon-site\b", labeled_blob):
            found.add("onsite")
        if found:
            return [x for x in REMOTE_ORDER if x in found]

    t = lower_text(primary)

    if re.search(r"\bhybrid\b|\bflexible hybrid\b", t):
        found.add("hybrid")
    if re.search(r"\bfully remote\b|\b100% remote\b|\bremote\b|\bwork from home\b|\bwfh\b|\bhome[- ]based\b", t):
        found.add("remote")

    onsite_patterns = [
        r"\bthis is an onsite role\b",
        r"\bthis role is onsite\b",
        r"\bworkplace type\s*:\s*onsite\b",
        r"\boffice[- ]based\b",
        r"\bfully office based\b",
        r"\b5 days (?:a week|per week)? in the office\b",
    ]
    if not found.intersection({"hybrid", "remote"}):
        if any(re.search(p, t) for p in onsite_patterns):
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

    # Strong priority: if full time/permanent exists anywhere in main section, return Permanent.
    if re.search(r"\bpermanent\b", t) or re.search(r"\bfull[\s-]?time\b", t):
        return "Permanent"

    if re.search(r"\bpart[\s-]?time\b", t) or re.search(r"\bjob[\s-]?share\b", t):
        return "Part Time"

    if re.search(r"\bfixed[\s-]?term\b|\btemporary\b|\bmaternity cover\b|\bmaternity leave\b", t):
        return "FTC"

    if re.search(r"\bfreelance\b|\bcontract role\b|\bcontract position\b|\bcontractor\b|\bcontracting\b", t):
        return "Freelance/Contract"

    # Only very explicit standalone contract-employment phrasing should fallback to contract.
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


def location_or_remote_missing(description: str) -> bool:
    """Returns True if either location or remote preference could not be found."""
    locations = extract_location_candidates(description)
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
