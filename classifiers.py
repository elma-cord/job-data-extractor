import json
import os
import re
from typing import Any, Dict, List

from openai import OpenAI

from prompts import (
    build_job_titles_prompt,
    build_relevance_prompt,
    build_seniority_prompt,
    build_skills_prompt,
    build_tag_relevant_job_prompt,
)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


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
        .replace("-", "-")
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

    for jt in allowed_job_titles:
        if value_clean.lower() == jt.lower():
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


def ai_check_relevance(
    job_title: str,
    role_context_text: str,
    fallback_role_relevance: str,
    fallback_reason: str,
) -> Dict[str, str]:
    if not client:
        return {
            "role_relevance": fallback_role_relevance,
            "role_relevance_reason": fallback_reason or "OPENAI_API_KEY missing",
        }

    prompt = build_relevance_prompt(job_title, role_context_text)

    try:
        response = client.responses.create(model=OPENAI_MODEL, input=prompt)
        data = safe_json_loads(response.output_text)
        role_relevance = str(data.get("role_relevance", "") or "").strip()
        reason = str(data.get("role_relevance_reason", "") or "").strip()

        if role_relevance not in {"Relevant", "Not relevant"}:
            role_relevance = fallback_role_relevance
        if not reason:
            reason = fallback_reason

        return {
            "role_relevance": role_relevance,
            "role_relevance_reason": reason,
        }
    except Exception as e:
        return {
            "role_relevance": fallback_role_relevance,
            "role_relevance_reason": f"AI error: {e}",
        }


def ai_tag_relevant_job(
    job_title: str,
    header_text: str,
    role_body_text: str,
    allowed_locations: List[str],
    allowed_salaries: List[int],
    allowed_job_titles: List[str],
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
    fallback_job_description: str,
) -> Dict[str, Any]:
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
            "job_description": fallback_job_description,
            "job_titles": [],
            "seniorities": [],
        }

    prompt = build_tag_relevant_job_prompt(
        job_title=job_title,
        header_text=header_text,
        role_body_text=role_body_text,
        allowed_locations=allowed_locations,
        allowed_salaries=allowed_salaries,
        allowed_job_titles=allowed_job_titles,
    )

    try:
        response = client.responses.create(model=OPENAI_MODEL, input=prompt)
        data = safe_json_loads(response.output_text)

        job_category = normalize_category_for_skills(str(data.get("job_category", "") or "").strip()) or fallback_job_category
        job_location = str(data.get("job_location", "") or "").strip() or fallback_location
        remote_preferences = str(data.get("remote_preferences", "") or "").strip() or fallback_remote_preferences
        remote_days = str(data.get("remote_days", "") or "").strip() or fallback_remote_days
        salary_min = str(data.get("salary_min", "") or "").strip() or fallback_salary_min
        salary_max = str(data.get("salary_max", "") or "").strip() or fallback_salary_max
        salary_currency = str(data.get("salary_currency", "") or "").strip() or fallback_salary_currency
        salary_period = str(data.get("salary_period", "") or "").strip() or fallback_salary_period
        visa_sponsorship = str(data.get("visa_sponsorship", "") or "").strip()
        job_type = str(data.get("job_type", "") or "").strip()
        job_description = str(data.get("job_description", "") or "").strip() or fallback_job_description

        if visa_sponsorship not in {"yes", "no", ""}:
            visa_sponsorship = fallback_visa
        if job_type not in {"Permanent", "FTC", "Part Time", "Freelance/Contract", ""}:
            job_type = fallback_job_type
        if salary_period not in {"year", "day", "hour", "month", ""}:
            salary_period = fallback_salary_period

        if remote_preferences:
            parts = [p.strip() for p in remote_preferences.split(",") if p.strip()]
            valid_order = ["onsite", "hybrid", "remote"]
            parts = [p for p in valid_order if p in parts]
            if "hybrid" in parts and "remote" in parts:
                remote_preferences = "hybrid, remote"
            else:
                remote_preferences = ", ".join(parts)

        raw_titles = data.get("job_titles", [])
        if not isinstance(raw_titles, list):
            raw_titles = []
        normalized_titles = []
        for t in raw_titles:
            exact = normalize_job_title_from_list(str(t), allowed_job_titles)
            if exact and exact not in normalized_titles:
                normalized_titles.append(exact)
        normalized_titles = normalized_titles[:3]

        raw_seniorities = data.get("seniorities", [])
        if not isinstance(raw_seniorities, list):
            raw_seniorities = []
        normalized_seniorities = normalize_seniority_list(raw_seniorities)

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
            "job_description": job_description,
            "job_titles": normalized_titles,
            "seniorities": normalized_seniorities,
        }
    except Exception as e:
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
            "job_description": fallback_job_description,
            "job_titles": [],
            "seniorities": [],
            "_ai_error": str(e),
        }


def ai_map_job_titles_only(position_name: str, description: str, allowed_job_titles: List[str]) -> List[str]:
    if not client:
        return []

    prompt = build_job_titles_prompt(position_name, description, allowed_job_titles)

    try:
        response = client.responses.create(model=OPENAI_MODEL, input=prompt)
        data = safe_json_loads(response.output_text)
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
        response = client.responses.create(model=OPENAI_MODEL, input=prompt)
        data = safe_json_loads(response.output_text)
        raw = data.get("seniorities", [])
        if not isinstance(raw, list):
            return []
        return normalize_seniority_list(raw)
    except Exception:
        return []


def ai_enrich_skills(
    role_category: str,
    description: str,
    exact_skills: List[str],
    allowed_skills: List[str],
) -> List[str]:
    if not client or not role_category or not description or not allowed_skills:
        return exact_skills[:10]

    role_category = normalize_category_for_skills(role_category)
    if role_category not in {"T&P", "NonT&P"}:
        return exact_skills[:10]

    prompt = build_skills_prompt(role_category, description, exact_skills, allowed_skills)

    try:
        response = client.responses.create(model=OPENAI_MODEL, input=prompt)
        data = safe_json_loads(response.output_text)

        out_category = normalize_category_for_skills(str(data.get("role_category", "") or "").strip())
        if out_category != role_category:
            return exact_skills[:10]

        raw_skills = data.get("skills", [])
        if not isinstance(raw_skills, list):
            return exact_skills[:10]

        merged = []
        for sk in exact_skills:
            exact = next((s for s in allowed_skills if s.lower() == sk.lower()), "")
            if exact and exact not in merged:
                merged.append(exact)

        for sk in raw_skills:
            sk_clean = str(sk).strip()
            exact = next((s for s in allowed_skills if s.lower() == sk_clean.lower()), "")
            if exact and exact not in merged:
                merged.append(exact)

        filtered = []
        allowed_lower = {s.lower(): s for s in allowed_skills}
        for sk in merged:
            exact = allowed_lower.get(sk.lower())
            if exact and exact not in filtered:
                filtered.append(exact)

        return filtered[:10]
    except Exception:
        return exact_skills[:10]
