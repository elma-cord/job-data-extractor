import re
from typing import Iterable


def normalize_whitespace(value: str) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def split_csv_like_list(value: str) -> list[str]:
    value = normalize_whitespace(value)
    if not value:
        return []
    parts = [p.strip() for p in value.split(",")]
    return [p for p in parts if p]


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def normalize_relevance_label(value: str) -> str:
    v = normalize_whitespace(value).lower()
    if "not relevant" in v:
        return "Not Relevant"
    if "relevant" in v:
        return "Relevant"
    return ""


def normalize_tp_label(value: str) -> str:
    v = normalize_whitespace(value).lower()
    if "not t&p" in v or "not tp" in v:
        return "Not T&P"
    if "t&p" in v or "tp job" in v:
        return "T&P job"
    return ""


def parse_role_relevance_response(response_text: str) -> dict:
    text = normalize_whitespace(response_text)
    parts = [p.strip() for p in text.split(" | ")]
    while len(parts) < 3:
        parts.append("")
    return {
        "role_relevance": normalize_relevance_label(parts[0]),
        "job_category": normalize_tp_label(parts[1]),
        "role_relevance_reason": parts[2],
    }


def validate_job_titles(value: str, allowed_job_titles: list[str], max_items: int = 3) -> list[str]:
    allowed_set = {normalize_whitespace(x).lower(): x for x in allowed_job_titles}
    out = []
    for item in split_csv_like_list(value):
        key = normalize_whitespace(item).lower()
        if key in allowed_set:
            out.append(allowed_set[key])
    return dedupe_preserve_order(out)[:max_items]


def validate_seniorities(value: str, allowed_seniorities: list[str], max_items: int = 3) -> list[str]:
    value = value or ""
    allowed = {x.lower(): x.lower() for x in allowed_seniorities}
    ordered = []
    for item in split_csv_like_list(value.lower()):
        item = item.strip().lower()
        if item in allowed:
            ordered.append(item)

    ordered = dedupe_preserve_order(ordered)
    sort_order = {name: idx for idx, name in enumerate(allowed_seniorities)}
    ordered.sort(key=lambda x: sort_order.get(x, 999))
    return ordered[:max_items]


def validate_contract_type(value: str) -> str:
    v = normalize_whitespace(value)
    allowed = {"Permanent", "FTC", "Part Time", "Freelance/Contract"}
    return v if v in allowed else ""


def validate_skills(value: str, allowed_skills: list[str], max_items: int = 10) -> list[str]:
    allowed_set = {normalize_whitespace(x).lower(): x for x in allowed_skills}
    out = []
    for item in split_csv_like_list(value):
        key = normalize_whitespace(item).lower()
        if key in allowed_set:
            out.append(allowed_set[key])
    return dedupe_preserve_order(out)[:max_items]


def clean_description(text: str) -> str:
    text = text or ""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_int(value):
    try:
        return int(value)
    except Exception:
        return None
