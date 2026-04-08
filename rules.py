import csv
import re
from pathlib import Path


def load_single_column_csv(path: Path) -> list[str]:
    values = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            value = (row[0] or "").strip()
            if value and value.lower() not in {"job title", "location", "skill", "salary"}:
                values.append(value)
    return values


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def lower_text(text: str) -> str:
    return normalize_text(text).lower()


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def get_primary_text_window(text: str, max_chars: int = 10000) -> str:
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


def looks_like_non_job_content(position_name: str, description: str) -> bool:
    text = lower_text(f"{position_name}\n{description}")
    if not text:
        return True

    hard_non_job_terms = [
        "course module",
        "learning module",
        "training module",
        "lesson",
        "curriculum",
        "course content",
        "study guide",
        "career guide",
        "salary guide",
        "news article",
        "blog post",
        "learning objectives",
        "administrative checklist",
        "checklist only",
    ]
    job_signals = [
        "responsibilities",
        "requirements",
        "experience",
        "apply",
        "job type",
        "salary",
        "location",
        "benefits",
        "we are looking for",
        "candidate",
        "team",
        "reporting to",
        "full time",
        "part time",
    ]

    if any(term in text for term in hard_non_job_terms) and not any(term in text for term in job_signals):
        return True

    if "learning objectives" in text and "apply" not in text and "responsibilities" not in text:
        return True

    if "administrative checklist" in text and "apply" not in text and "responsibilities" not in text:
        return True

    return False


def obvious_excluded_role(position_name: str, description: str) -> tuple[bool, str]:
    text = lower_text(f"{position_name}\n{description}")

    exclusion_patterns = [
        (r"\bteacher\b|\bteaching assistant\b", "Teaching role is outside allowed tech/business scope."),
        (r"\bnurse\b|\bregistered nurse\b|\bhealthcare assistant\b", "Medical role is outside allowed tech/business scope."),
        (r"\bwaiter\b|\bwaitress\b|\bchef\b|\bkitchen\b", "Hospitality role is outside allowed tech/business scope."),
        (r"\bcashier\b|\bretail assistant\b|\bsales associate\b|\bshop assistant\b", "Retail store role is outside allowed tech/business scope."),
        (r"\bconstruction\b|\bcivil engineer(?:ing)?\b", "Construction / civil engineering role is outside allowed scope."),
        (r"\bmanufacturing technician\b|\bproduction operator\b|\bassembly technician\b|\bshop floor\b|\bproduction line\b|\bplant operator\b", "Manufacturing / shop-floor role is outside allowed scope."),
        (r"\bpsychiatrist\b|\bphysician\b|\bsurgeon\b|\btherapist\b|\bpatient care\b|\bhospital\b", "Medical / clinical role is outside allowed scope."),
        (r"\brf test engineer\b|\bradio frequency test engineer\b", "RF test engineering role is outside allowed target scope."),
    ]

    for pattern, reason in exclusion_patterns:
        if re.search(pattern, text):
            return True, reason

    return False, ""


def detect_quick_tp_from_title(position_name: str) -> str:
    title = lower_text(position_name)

    tp_terms = [
        "engineer", "developer", "software", "data", "product", "ux", "ui",
        "devops", "site reliability", "security", "qa", "automation",
        "backend", "back end", "front end", "frontend", "full stack",
        "it support", "infrastructure", "architect", "technical", "support engineer",
        "systems engineer", "system engineer", "network engineer", "platform",
        "cloud", "sre", "machine learning", "ai engineer",
    ]

    if any(term in title for term in tp_terms):
        return "T&P job"
    return "Not T&P" if title else ""


def title_has_leadership_signal(position_name: str) -> bool:
    title = lower_text(position_name)
    leadership_terms = [
        "head of",
        "director",
        "vice president",
        "vp ",
        "chief ",
        "engineering manager",
        "technical director",
        "cto",
        "cfo",
        "coo",
        "cio",
        "cmo",
        "cro",
        "cpo",
    ]
    return any(term in title for term in leadership_terms)


def _normalize_for_matching(text: str) -> str:
    text = text.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[+/]", " ", text)
    text = re.sub(r"[^a-z0-9#.\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def skill_is_supported(skill: str, text: str) -> bool:
    if not skill or not text:
        return False

    text_n = _normalize_for_matching(text)
    skill_n = _normalize_for_matching(skill)

    if not skill_n:
        return False

    if re.search(rf"(?<![a-z0-9]){re.escape(skill_n)}(?![a-z0-9])", text_n):
        return True

    parts = [p for p in skill_n.split() if p]
    if len(parts) >= 2:
        return all(re.search(rf"(?<![a-z0-9]){re.escape(p)}(?![a-z0-9])", text_n) for p in parts)

    return False


def extract_deterministic_skills(position_name: str, description: str, allowed_skills: list[str], max_items: int = 10) -> list[str]:
    full_text = f"{position_name}\n{description}"
    out = []

    for skill in allowed_skills:
        if skill_is_supported(skill, full_text):
            out.append(skill)

    return dedupe_keep_order(out)[:max_items]


def closest_salary_value(value: int, allowed_salaries: list[int]) -> str:
    if not allowed_salaries:
        return ""
    return str(min(allowed_salaries, key=lambda x: abs(x - value)))


def salary_context_exists(text: str) -> bool:
    t = lower_text(text)
    patterns = [
        r"\bsalary\b",
        r"\bcompensation\b",
        r"\bpay\b",
        r"\bpackage\b",
        r"£\s*\d",
        r"\$\s*\d",
        r"€\s*\d",
        r"\bgbp\b",
        r"\busd\b",
        r"\beur\b",
        r"\bcad\b",
        r"\bper annum\b",
        r"\bper year\b",
        r"\bannually\b",
    ]
    return any(re.search(p, t) for p in patterns)


def normalize_location_match(value: str, allowed_locations: list[str]) -> str:
    value_n = lower_text(value)
    allowed_map = {lower_text(x): x for x in allowed_locations}

    if value_n in allowed_map:
        return allowed_map[value_n]

    value_n_simple = re.sub(r"[^a-z0-9 ]+", " ", value_n)
    value_n_simple = re.sub(r"\s+", " ", value_n_simple).strip()

    best = ""
    best_score = -1

    for loc in allowed_locations:
        loc_n = lower_text(loc)
        loc_n_simple = re.sub(r"[^a-z0-9 ]+", " ", loc_n)
        loc_n_simple = re.sub(r"\s+", " ", loc_n_simple).strip()

        score = 0
        if value_n_simple == loc_n_simple:
            score += 1000

        value_parts = set(value_n_simple.split())
        loc_parts = set(loc_n_simple.split())
        overlap = len(value_parts & loc_parts)
        score += overlap * 25

        if value_parts and value_parts.issubset(loc_parts):
            score += 80

        if score > best_score:
            best_score = score
            best = loc

    if best_score >= 50:
        return best

    return ""


def is_location_allowed(job_location: str, remote_preferences: list[str], source_text: str) -> bool:
    if not job_location or lower_text(job_location) == "unknown":
        return True

    loc = lower_text(job_location)
    text = lower_text(source_text)
    remote_set = set(remote_preferences)

    uk_markers = ["uk", "united kingdom", "england", "scotland", "wales", "northern ireland"]
    if any(m in loc for m in uk_markers):
        return True

    if "ireland" in loc:
        return "remote" in remote_set

    if "europe" in loc or "emea" in loc:
        return "remote" in remote_set

    if "global" in loc or "worldwide" in loc:
        if "remote" not in remote_set:
            return False
        blocked_regions = [
            "remote apac", "apac only", "asia only", "remote asia",
            "latam only", "remote latam", "africa only", "remote africa",
            "usa only", "us only", "canada only",
        ]
        return not any(term in text for term in blocked_regions)

    hard_disallowed = [
        "united states", "usa", "canada", "germany", "france", "spain", "italy",
        "netherlands", "belgium", "sweden", "norway", "denmark", "finland",
        "switzerland", "austria", "poland", "portugal", "india", "singapore",
        "japan", "china", "australia",
    ]
    if any(term in loc for term in hard_disallowed):
        return False

    return True


def reason_strongly_says_not_relevant(reason: str) -> bool:
    r = lower_text(reason)
    if not r:
        return False

    signals = [
        "outside allowed",
        "not a real job posting",
        "educational",
        "informational",
        "checklist",
        "medical",
        "clinical",
        "retail",
        "shop-floor",
        "manufacturing",
        "construction",
        "civil engineering",
        "rf test",
        "language other than english",
        "location is not allowed",
        "outside allowed regions",
        "usa only",
        "canada only",
    ]
    return any(s in r for s in signals)
