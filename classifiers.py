import hashlib
import json
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from prompts import (
    build_core_fields_prompt,
    build_job_titles_prompt,
    build_relevance_prompt,
    build_salary_prompt,
    build_seniority_prompt,
    build_skills_additional_prompt,
    build_skills_full_prompt,
)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

OPENAI_TIMEOUT_S = float(os.getenv("OPENAI_TIMEOUT_S", "35"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
OPENAI_RETRY_SLEEP_S = float(os.getenv("OPENAI_RETRY_SLEEP_S", "1.2"))

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

_cache_lock = threading.Lock()
_response_cache: Dict[str, Any] = {}


def clean_whitespace(text: str) -> str:
    text = text or ""
    text = text.replace("\xa0", " ")
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_json_loads(text: str) -> Dict[str, Any]:
    text = (text or "").strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json|html)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    return json.loads(text)


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


def normalize_job_title_from_list(value: str, allowed_job_titles: List[str]) -> str:
    value_clean = clean_whitespace(value)
    if not value_clean:
        return ""

    value_lower = value_clean.lower()
    for jt in allowed_job_titles:
        if value_lower == jt.lower():
            return jt

    canon_value = canonical_label(value_clean)
    for jt in allowed_job_titles:
        if canon_value == canonical_label(jt):
            return jt

    return ""


def normalize_seniority_list(values: List[str]) -> List[str]:
    order = ["entry", "junior", "mid", "senior", "lead", "leadership"]
    found = []
    for v in values:
        low = str(v).strip().lower()
        if low in order and low not in found:
            found.append(low)
    return [x for x in order if x in found][:3]


def normalize_remote_preferences_value(value: str) -> str:
    value = clean_whitespace(value)
    if not value:
        return ""

    low = normalize_quotes(value).lower()
    if low in {"not specified", "not_specified"}:
        return "not specified"

    found = []
    for pref in ["onsite", "hybrid", "remote"]:
        if re.search(rf"(?<![a-z]){re.escape(pref)}(?![a-z])", low):
            found.append(pref)

    if not found:
        return ""

    return ", ".join(found)


def _cache_key(namespace: str, payload: str) -> str:
    return f"{namespace}:{hashlib.sha1(payload.encode('utf-8')).hexdigest()}"


def _cache_get(key: str) -> Optional[Any]:
    with _cache_lock:
        return _response_cache.get(key)


def _cache_set(key: str, value: Any) -> None:
    with _cache_lock:
        _response_cache[key] = value


def _call_openai_text(prompt: str) -> str:
    if not client:
        raise RuntimeError("OPENAI_API_KEY missing")

    last_err = None
    for attempt in range(OPENAI_MAX_RETRIES + 1):
        try:
            response = client.with_options(timeout=OPENAI_TIMEOUT_S).responses.create(
                model=OPENAI_MODEL,
                input=prompt,
            )
            return clean_whitespace(response.output_text)
        except Exception as e:
            last_err = e
            if attempt < OPENAI_MAX_RETRIES:
                time.sleep(OPENAI_RETRY_SLEEP_S * (attempt + 1))
            else:
                raise last_err


def _call_openai_json_cached(namespace: str, prompt: str) -> Dict[str, Any]:
    key = _cache_key(namespace, prompt)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    text = _call_openai_text(prompt)
    data = safe_json_loads(text)
    _cache_set(key, data)
    return data


def ai_check_relevance(
    job_title: str,
    role_context_text: str,
    fallback_role_relevance: str,
    fallback_reason: str,
    fallback_job_category: str = "",
) -> Dict[str, str]:
    if not client:
        return {
            "role_relevance": fallback_role_relevance,
            "job_category": fallback_job_category,
            "role_relevance_reason": fallback_reason or "OPENAI_API_KEY missing",
        }

    prompt = build_relevance_prompt(job_title, role_context_text)

    try:
        data = _call_openai_json_cached("relevance", prompt)
        role_relevance = str(data.get("role_relevance", "") or "").strip()
        reason = str(data.get("role_relevance_reason", "") or "").strip()
        job_category = normalize_category_for_skills(str(data.get("job_category", "") or "").strip())

        if role_relevance not in {"Relevant", "Not relevant"}:
            role_relevance = fallback_role_relevance
        if not reason:
            reason = fallback_reason
        if not job_category:
            job_category = fallback_job_category

        return {
            "role_relevance": role_relevance,
            "job_category": job_category,
            "role_relevance_reason": reason,
        }
    except Exception as e:
        return {
            "role_relevance": fallback_role_relevance,
            "job_category": fallback_job_category,
            "role_relevance_reason": f"AI error: {e}",
        }


def ai_extract_core_fields(
    job_title: str,
    header_text: str,
    role_body_text: str,
    allowed_locations: List[str],
    fallback_job_category: str,
    fallback_location: str,
    fallback_remote_preferences: str,
    fallback_remote_days: str,
    fallback_salary_min: str,
    fallback_salary_max: str,
    fallback_salary_currency: str,
    fallback_salary_period: str,
    fallback_visa: str,
    fallback_job_type: str,
) -> Dict[str, str]:
    if not client:
        return {
            "job_category": fallback_job_category,
            "job_location": fallback_location,
            "remote_preferences": fallback_remote_preferences,
            "remote_days": fallback_remote_days,
            "salary_min": fallback_salary_min,
            "salary_max": fallback_salary_max,
            "salary_currency": fallback_salary_currency,
            "salary_period": fallback_salary_period,
            "visa_sponsorship": fallback_visa,
            "job_type": fallback_job_type,
        }

    prompt = build_core_fields_prompt(
        job_title=job_title,
        header_text=header_text,
        role_body_text=role_body_text,
        allowed_locations=allowed_locations,
    )

    try:
        data = _call_openai_json_cached("core_fields", prompt)

        job_category = (
            normalize_category_for_skills(str(data.get("job_category", "") or "").strip())
            or fallback_job_category
        )
        job_location = str(data.get("job_location", "") or "").strip() or fallback_location

        remote_preferences = normalize_remote_preferences_value(
            str(data.get("remote_preferences", "") or "").strip()
        ) or fallback_remote_preferences

        remote_days = str(data.get("remote_days", "") or "").strip() or fallback_remote_days
        if remote_days.lower() == "not specified":
            remote_days = "not specified"

        salary_min = str(data.get("salary_min", "") or "").strip() or fallback_salary_min
        salary_max = str(data.get("salary_max", "") or "").strip() or fallback_salary_max
        salary_currency = str(data.get("salary_currency", "") or "").strip() or fallback_salary_currency
        salary_period = str(data.get("salary_period", "") or "").strip() or fallback_salary_period

        visa_sponsorship = str(data.get("visa_sponsorship", "") or "").strip()
        job_type = str(data.get("job_type", "") or "").strip()

        if visa_sponsorship not in {"yes", "no", ""}:
            visa_sponsorship = fallback_visa

        if job_type not in {"Permanent", "FTC", "Part Time", "Freelance/Contract", ""}:
            job_type = fallback_job_type

        if salary_period not in {"year", "day", "hour", "month", ""}:
            salary_period = fallback_salary_period

        if salary_currency not in {"GBP", "USD", "EUR", "CAD", ""}:
            salary_currency = fallback_salary_currency

        return {
            "job_category": job_category,
            "job_location": job_location,
            "remote_preferences": remote_preferences,
            "remote_days": remote_days,
            "salary_min": salary_min,
            "salary_max": salary_max,
            "salary_currency": salary_currency,
            "salary_period": salary_period,
            "visa_sponsorship": visa_sponsorship,
            "job_type": job_type,
        }
    except Exception:
        return {
            "job_category": fallback_job_category,
            "job_location": fallback_location,
            "remote_preferences": fallback_remote_preferences,
            "remote_days": fallback_remote_days,
            "salary_min": fallback_salary_min,
            "salary_max": fallback_salary_max,
            "salary_currency": fallback_salary_currency,
            "salary_period": fallback_salary_period,
            "visa_sponsorship": fallback_visa,
            "job_type": fallback_job_type,
        }


def ai_extract_salary_only(
    job_title: str,
    header_text: str,
    role_body_text: str,
    fallback_salary_min: str,
    fallback_salary_max: str,
    fallback_salary_currency: str,
    fallback_salary_period: str,
) -> Dict[str, str]:
    if not client:
        return {
            "salary_min": fallback_salary_min,
            "salary_max": fallback_salary_max,
            "salary_currency": fallback_salary_currency,
            "salary_period": fallback_salary_period,
        }

    prompt = build_salary_prompt(job_title, header_text, role_body_text)

    try:
        data = _call_openai_json_cached("salary_only", prompt)

        salary_min = str(data.get("salary_min", "") or "").strip() or fallback_salary_min
        salary_max = str(data.get("salary_max", "") or "").strip() or fallback_salary_max
        salary_currency = str(data.get("salary_currency", "") or "").strip() or fallback_salary_currency
        salary_period = str(data.get("salary_period", "") or "").strip() or fallback_salary_period

        if salary_period not in {"year", "day", "hour", "month", ""}:
            salary_period = fallback_salary_period

        if salary_currency not in {"GBP", "USD", "EUR", "CAD", ""}:
            salary_currency = fallback_salary_currency

        return {
            "salary_min": salary_min,
            "salary_max": salary_max,
            "salary_currency": salary_currency,
            "salary_period": salary_period,
        }
    except Exception:
        return {
            "salary_min": fallback_salary_min,
            "salary_max": fallback_salary_max,
            "salary_currency": fallback_salary_currency,
            "salary_period": fallback_salary_period,
        }


def ai_map_job_titles_only(position_name: str, description: str, allowed_job_titles: List[str]) -> List[str]:
    if not client:
        return []

    prompt = build_job_titles_prompt(position_name, description, allowed_job_titles)

    try:
        data = _call_openai_json_cached("job_titles_only", prompt)
        raw_titles = data.get("job_titles", [])
        if not isinstance(raw_titles, list):
            return []

        out = []
        for t in raw_titles:
            exact = normalize_job_title_from_list(str(t), allowed_job_titles)
            if exact and exact not in out:
                out.append(exact)

        return out[:3]
    except Exception:
        return []


def ai_map_seniority_only(position_name: str, description: str) -> List[str]:
    if not client:
        return []

    prompt = build_seniority_prompt(position_name, description)

    try:
        data = _call_openai_json_cached("seniority_only", prompt)
        raw = data.get("seniorities", [])
        if not isinstance(raw, list):
            return []
        return normalize_seniority_list(raw)
    except Exception:
        return []


def ai_generate_skills_full(
    role_category: str,
    description: str,
    candidate_skills: List[str],
    allowed_skills: List[str],
) -> List[str]:
    if not client or not role_category or not description or not allowed_skills:
        return []

    role_category = normalize_category_for_skills(role_category)
    if role_category not in {"T&P", "NonT&P"}:
        return []

    prompt = build_skills_full_prompt(
        role_category=role_category,
        description=description,
        candidate_skills=candidate_skills,
        allowed_skills=allowed_skills,
    )

    try:
        data = _call_openai_json_cached("skills_full", prompt)

        out_category = normalize_category_for_skills(str(data.get("role_category", "") or "").strip())
        if out_category != role_category:
            return []

        raw_skills = data.get("skills", [])
        if not isinstance(raw_skills, list):
            return []

        allowed_lower = {s.lower(): s for s in allowed_skills}
        out = []
        for sk in raw_skills:
            exact = allowed_lower.get(str(sk).strip().lower())
            if exact and exact not in out:
                out.append(exact)

        return out[:10]
    except Exception:
        return []


def ai_generate_additional_skills(
    role_category: str,
    description: str,
    existing_skills: List[str],
    candidate_skills: List[str],
    allowed_skills: List[str],
) -> List[str]:
    if not client or not role_category or not description or not allowed_skills:
        return []

    role_category = normalize_category_for_skills(role_category)
    if role_category not in {"T&P", "NonT&P"}:
        return []

    prompt = build_skills_additional_prompt(
        role_category=role_category,
        description=description,
        existing_skills=existing_skills,
        candidate_skills=candidate_skills,
        allowed_skills=allowed_skills,
    )

    try:
        data = _call_openai_json_cached("skills_additional", prompt)

        out_category = normalize_category_for_skills(str(data.get("role_category", "") or "").strip())
        if out_category != role_category:
            return []

        raw_skills = data.get("additional_skills", [])
        if not isinstance(raw_skills, list):
            return []

        allowed_lower = {s.lower(): s for s in allowed_skills}
        existing_lower = {s.lower() for s in existing_skills}

        out = []
        for sk in raw_skills:
            exact = allowed_lower.get(str(sk).strip().lower())
            if exact and exact.lower() not in existing_lower and exact not in out:
                out.append(exact)

        return out[:5]
    except Exception:
        return []
