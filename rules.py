import csv
import re
from pathlib import Path


CURRENCY_SIGNS = {
    "£": "GBP",
    "$": "USD",
    "€": "EUR",
}

CURRENCY_CODES = {"GBP", "USD", "EUR", "CAD", "AUD", "CHF"}

REMOTE_ORDER = ["onsite", "hybrid", "remote"]

CONTRACT_PATTERNS = {
    "Permanent": [
        r"\bpermanent\b",
        r"\bfull[\s-]?time\b",
        r"\bstandard\b",
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
        r"\bcontract(?:or|ing)?\b",
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
JUNIOR_TERMS = ["junior", "jr", "entry level", "graduate"]
SENIOR_TERMS = ["senior", "sr", "staff", "principal"]
LEAD_TERMS = ["lead", "team lead"]

IRRELEVANT_ROLE_HINTS = [
    "teacher", "nurse", "waiter", "chef", "warehouse", "cleaner", "receptionist",
    "construction", "civil engineer", "mechanical", "electrical", "manufacturing",
    "beauty therapist", "maritime", "microbiology", "injection moulding", "retail assistant",
]

TP_HINTS = [
    "engineer", "developer", "software", "data", "product", "ux", "ui", "devops",
    "site reliability", "security", "qa", "automation", "backend", "front end",
    "full stack", "it support", "infrastructure", "architect", "technical"
]


def load_single_column_csv(path: Path) -> list[str]:
    values = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            value = row[0].strip()
            if value and value.lower() != "job title" and value.lower() != "location" and value.lower() != "skill":
                values.append(value)
    return values


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def lower_text(text: str) -> str:
    return normalize_text(text).lower()


def extract_remote_preferences(text: str) -> list[str]:
    t = lower_text(text)
    found = set()

    if re.search(r"\bon[\s-]?site\b|\bin office\b|\bfrom the office\b|\boffice based\b", t):
        found.add("onsite")
    if re.search(r"\bhybrid\b|\bflexible hybrid\b|\bblend of home and office\b", t):
        found.add("hybrid")
    if re.search(r"\bremote\b|\bwork from home\b|\bwfh\b|\bhome[- ]based\b", t):
        found.add("remote")

    return [x for x in REMOTE_ORDER if x in found]


def extract_remote_days(text: str) -> str:
    t = lower_text(text)

    if re.search(r"\bfully remote\b|\b100% remote\b|\bremote every day\b", t):
        return "not specified"

    office_match = re.search(r"\b(\d)(?:\s*-\s*(\d))?\s+days?\s+(?:in|from)\s+the\s+office\b", t)
    if office_match:
        start = int(office_match.group(1))
        end = int(office_match.group(2) or office_match.group(1))
        remote_options = [5 - start, 5 - end]
        return str(max(remote_options))

    remote_match = re.search(r"\b(\d)(?:\s*-\s*(\d))?\s+days?\s+(?:remote|from home|wfh)\b", t)
    if remote_match:
        start = int(remote_match.group(1))
        end = int(remote_match.group(2) or remote_match.group(1))
        return str(max(start, end))

    return "not specified"


def detect_contract_type(text: str) -> str:
    t = lower_text(text)
    for label in ["Permanent", "FTC", "Part Time", "Freelance/Contract"]:
        for pattern in CONTRACT_PATTERNS[label]:
            if re.search(pattern, t):
                return label
    return ""


def detect_seniority_from_title_and_description(position_name: str, description: str) -> list[str]:
    title = lower_text(position_name)
    desc = lower_text(description)
    full = f"{title} {desc}"

    if any(term in title for term in LEADERSHIP_TERMS):
        return ["leadership"]

    out = []

    if any(term in title for term in JUNIOR_TERMS):
        out.extend(["junior"])
    if any(term in title for term in MID_TERMS):
        out.extend(["mid"])
    if any(term in title for term in SENIOR_TERMS):
        out.extend(["senior"])
    if any(term in title for term in LEAD_TERMS):
        out.extend(["lead"])

    exp_match = re.search(r"\b(\d+)\s*(?:\+|plus)?\s+years?\b", full)
    if exp_match:
        years = int(exp_match.group(1))
        if years <= 1:
            out.extend(["entry", "junior"])
        elif years == 2:
            out.extend(["junior", "mid"])
        elif 3 <= years <= 5:
            out.extend(["senior", "lead"])
        elif years > 5:
            out.extend(["senior", "lead"])

    if re.search(r"\bmanage\b|\bmanagement\b|\bline manager\b|\bteam leadership\b|\bmentor\b", full):
        out.append("lead")

    order = ["entry", "junior", "mid", "senior", "lead", "leadership"]
    out = [x for x in order if x in out]
    deduped = []
    for item in out:
        if item not in deduped:
            deduped.append(item)
    return deduped[:3]


def detect_basic_relevance_from_title(position_name: str) -> tuple[str, str]:
    title = lower_text(position_name)

    if any(bad in title for bad in IRRELEVANT_ROLE_HINTS):
        return "Not Relevant", "Title clearly points to an excluded non-target function."

    if any(tp in title for tp in TP_HINTS):
        return "Relevant", "Title appears to match target tech/product scope."

    return "", ""


def detect_tp_from_title(position_name: str) -> str:
    title = lower_text(position_name)
    if any(tp in title for tp in TP_HINTS):
        return "T&P job"
    return ""


def extract_salary(text: str, allowed_salaries: list[int]) -> dict:
    t = text or ""

    currency = ""
    for sign, code in CURRENCY_SIGNS.items():
        if sign in t:
            currency = code
            break
    if not currency:
        code_match = re.search(r"\b(GBP|USD|EUR|CAD|AUD|CHF)\b", t, re.I)
        if code_match:
            currency = code_match.group(1).upper()

    nums = []
    for m in re.findall(r"(?<!\d)(\d{2,3}(?:,\d{3})+|\d{4,6})(?!\d)", t):
        clean = int(m.replace(",", ""))
        if clean >= 5000:
            nums.append(clean)

    if not nums:
        return {"salary_min": "", "salary_max": "", "salary_currency": currency}

    nums = sorted(nums[:2]) if len(nums) >= 2 else [nums[0], nums[0]]

    def closest(val: int) -> int:
        return min(allowed_salaries, key=lambda x: abs(x - val))

    return {
        "salary_min": str(closest(nums[0])),
        "salary_max": str(closest(nums[1])),
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
    t = lower_text(description)
    has_remote = bool(extract_remote_preferences(t))
    has_location_hint = bool(
        re.search(
            r"\blondon\b|\buk\b|\bunited kingdom\b|\bengland\b|\bscotland\b|\bwales\b|\bnorthern ireland\b|\bireland\b|\bemea\b|\beurope\b|\bremote\b",
            t,
        )
    )
    return not (has_remote and has_location_hint)
