import html
import json
import re
from typing import Any, Iterable


# Digits observed standing in for a lost apostrophe (CORD-6921). The apostrophe
# most often becomes a "7" ("We 7re" -> "We're"), but 2/9/27/39/92/99 have also
# been seen. We only touch a digit that sits where an apostrophe grammatically
# belongs - directly before a contraction ending or a possessive - so real
# numbers ("7 years", "70,000", "24/7", "7am") are never altered.
_APOSTROPHE_DIGIT_RE = re.compile(
    r"\b([A-Za-z]+)\s*(?:2|7|9|27|39|92|99)\s*(ll|re|ve|s|d|m|t)\b",
    re.IGNORECASE,
)

# Curly / non-standard punctuation -> plain ASCII.
_PUNCT_MAP = {
    "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-",
    "\u00a0": " ",
}


def repair_text(text: str) -> str:
    """Fix character-encoding damage in job text.

    - Decodes HTML entities (&amp;, &#39;, &#x27;, &rsquo;, ...).
    - Restores apostrophes corrupted into stray digits ("We 7re" -> "We're",
      "don7t" -> "don't", "company7s" -> "company's"), but only where the digit
      stands in for an apostrophe. Real numbers are left untouched.
    - Normalises curly quotes/dashes and non-breaking spaces to plain ASCII.

    Idempotent and safe on both plain text and simple HTML.
    """
    if not text:
        return text

    s = str(text)
    s = html.unescape(s)
    s = _APOSTROPHE_DIGIT_RE.sub(r"\1'\2", s)
    for bad, good in _PUNCT_MAP.items():
        s = s.replace(bad, good)
    return s


def normalize_whitespace(value: str) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def clean_description(text: str) -> str:
    # Repair encoding damage FIRST (decode entities, fix apostrophe-as-digit
    # artifacts) so every downstream consumer - classification, matching, and
    # the trimmed description - works on clean text.
    text = repair_text(text or "")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def extract_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}

    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}

    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def normalize_relevance_label(value: str) -> str:
    v = normalize_whitespace(value).lower()
    if v == "relevant":
        return "Relevant"
    if v == "not relevant":
        return "Not Relevant"
    return ""


def normalize_tp_label(value: str) -> str:
    v = normalize_whitespace(value).lower()
    if v in {"t&p", "tp", "t&p job", "tp job"}:
        return "T&P job"
    if v in {"not t&p", "not tp", "non-tp", "non tp", "not t&p job"}:
        return "Not T&P"
    return ""


def normalize_remote_preferences(value: Any) -> list[str]:
    allowed = {"onsite", "hybrid", "remote"}

    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str):
        raw_items = [x.strip() for x in value.split(",")]
    else:
        raw_items = []

    cleaned = []
    for item in raw_items:
        item_l = normalize_whitespace(str(item)).lower()
        if item_l in allowed and item_l not in cleaned:
            cleaned.append(item_l)

    # Keep every supported work pattern (e.g. ["onsite", "hybrid"]) in canonical
    # order. Combinations are allowed so a role surfaces for candidates filtering
    # on either pattern; the one contradictory pair (onsite + remote) is handled
    # downstream in the classifier.
    ordered = [x for x in ["onsite", "hybrid", "remote"] if x in cleaned]

    return ordered


def normalize_remote_days(value: Any) -> str:
    value = normalize_whitespace("" if value is None else str(value)).lower()
    if not value:
        return "not specified"
    if value == "not specified":
        return "not specified"
    if re.fullmatch(r"[0-5]", value):
        return value
    return "not specified"


def validate_contract_type(value: str, allowed_contract_types: list[str]) -> str:
    v = normalize_whitespace(value)
    return v if v in set(allowed_contract_types) else ""


def validate_job_titles(value: Any, allowed_job_titles: list[str], max_items: int = 3) -> list[str]:
    allowed_set = {normalize_whitespace(x).lower(): x for x in allowed_job_titles}

    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str):
        raw_items = [x.strip() for x in value.split(",")]
    else:
        raw_items = []

    out = []
    for item in raw_items:
        key = normalize_whitespace(item).lower()
        if key in allowed_set:
            out.append(allowed_set[key])

    return dedupe_preserve_order(out)[:max_items]


def validate_seniorities(value: Any, allowed_seniorities: list[str], max_items: int = 3) -> list[str]:
    allowed = {x.lower(): x.lower() for x in allowed_seniorities}
    ordered_map = {x.lower(): i for i, x in enumerate(allowed_seniorities)}

    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str):
        raw_items = [x.strip() for x in value.split(",")]
    else:
        raw_items = []

    out = []
    for item in raw_items:
        item_l = normalize_whitespace(item).lower()
        if item_l in allowed:
            out.append(item_l)

    out = dedupe_preserve_order(out)
    out.sort(key=lambda x: ordered_map.get(x, 999))
    return out[:max_items]


def validate_skills(value: Any, allowed_skills: list[str], max_items: int = 10) -> list[str]:
    allowed_set = {normalize_whitespace(x).lower(): x for x in allowed_skills}

    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str):
        raw_items = [x.strip() for x in value.split(",")]
    else:
        raw_items = []

    out = []
    for item in raw_items:
        key = normalize_whitespace(item).lower()
        if key in allowed_set:
            out.append(allowed_set[key])

    return dedupe_preserve_order(out)[:max_items]


def safe_str(value: Any) -> str:
    return normalize_whitespace("" if value is None else str(value))
