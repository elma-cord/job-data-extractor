import json
import re
from typing import Any, Iterable


def normalize_whitespace(value: str) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def clean_description(text: str) -> str:
    text = text or ""
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
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}

    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
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
    if v in {"not t&p", "not tp", "non tp", "non-tp", "not t&p job", "not t&p role"}:
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

    ordered = [x for x in ["onsite", "hybrid", "remote"] if x in cleaned]

    if "hybrid" in ordered and "remote" in ordered:
        ordered = [x for x in ordered if x != "remote"]

    return ordered


def normalize_remote_days(value: Any) -> str:
    v = normalize_whitespace(value).lower()
    if v in {"", "unknown"}:
        return "not specified"
    if v == "not specified":
        return "not specified"
    if re.fullmatch(r"[0-5]", v):
        return v
    return "not specified"


def validate_contract_type(value: str, allowed_contract_types: list[str]) -> str:
    v = normalize_whitespace(value)
    return v if v in set(allowed_contract_types) else ""


def validate_job_titles(value: Any, allowed_job_titles: list[str], max_items: int = 3) -> list[str]:
    allowed_map = {normalize_whitespace(x).lower(): x for x in allowed_job_titles}

    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str):
        raw_items = [x.strip() for x in value.split(",")]
    else:
        raw_items = []

    out = []
    for item in raw_items:
        key = normalize_whitespace(item).lower()
        if key in allowed_map:
            out.append(allowed_map[key])

    return dedupe_preserve_order(out)[:max_items]


def validate_seniorities(value: Any, allowed_seniorities: list[str], max_items: int = 3) -> list[str]:
    allowed = {x.lower(): x.lower() for x in allowed_seniorities}
    order_map = {x.lower(): i for i, x in enumerate(allowed_seniorities)}

    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str):
        raw_items = [x.strip() for x in value.split(",")]
    else:
        raw_items = []

    cleaned = []
    for item in raw_items:
        item_l = normalize_whitespace(item).lower()
        if item_l in allowed:
            cleaned.append(item_l)

    cleaned = dedupe_preserve_order(cleaned)
    cleaned.sort(key=lambda x: order_map.get(x, 999))
    return cleaned[:max_items]


def validate_skills(value: Any, allowed_skills: list[str], max_items: int = 10) -> list[str]:
    allowed_map = {normalize_whitespace(x).lower(): x for x in allowed_skills}

    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str):
        raw_items = [x.strip() for x in value.split(",")]
    else:
        raw_items = []

    out = []
    for item in raw_items:
        key = normalize_whitespace(item).lower()
        if key in allowed_map:
            out.append(allowed_map[key])

    return dedupe_preserve_order(out)[:max_items]


def safe_str(value: Any) -> str:
    return normalize_whitespace("" if value is None else str(value))


def safe_int(value: Any):
    try:
        return int(value)
    except Exception:
        return None
