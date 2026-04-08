import csv
import json
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from config import (
    ALLOWED_CONTRACT_TYPES,
    ALLOWED_SENIORITIES,
    LOCATION_UNKNOWN,
    MAIN_MODEL,
    MAX_JOB_TITLES,
    MAX_OUTPUT_TOKENS,
    MAX_RETRIES,
    MAX_SKILLS,
    MIN_DESCRIPTION_LENGTH_FOR_NO_FETCH,
    NOT_RELEVANT_LABEL,
    NOT_TP_LABEL,
    OPENAI_TIMEOUT_SECONDS,
    PREDEFINED_JOB_TITLES_CSV,
    PREDEFINED_LOCATIONS_CSV,
    PREDEFINED_NONTP_SKILLS_CSV,
    PREDEFINED_SALARIES_CSV,
    PREDEFINED_TP_SKILLS_CSV,
    RELEVANT_LABEL,
    TP_LABEL,
)
from fetch_extract import fetch_job_page_text
from prompts import build_unified_job_extraction_prompt
from rules import (
    closest_salary_value,
    dedupe_keep_order,
    detect_quick_tp_from_title,
    extract_deterministic_skills,
    is_location_allowed,
    load_single_column_csv,
    looks_like_non_job_content,
    normalize_location_match,
    obvious_excluded_role,
    reason_strongly_says_not_relevant,
    salary_context_exists,
    skill_is_supported,
    title_has_leadership_signal,
)
from validators import (
    clean_description,
    extract_json_object,
    normalize_relevance_label,
    normalize_remote_days,
    normalize_remote_preferences,
    normalize_tp_label,
    safe_str,
    validate_contract_type,
    validate_job_titles,
    validate_seniorities,
    validate_skills,
)


class JobClassifier:
    def __init__(self):
        self.client = OpenAI(timeout=OPENAI_TIMEOUT_SECONDS)

        self.predefined_job_titles = load_single_column_csv(Path(PREDEFINED_JOB_TITLES_CSV))
        self.predefined_locations = load_single_column_csv(Path(PREDEFINED_LOCATIONS_CSV))
        self.predefined_tp_skills = load_single_column_csv(Path(PREDEFINED_TP_SKILLS_CSV))
        self.predefined_nontp_skills = load_single_column_csv(Path(PREDEFINED_NONTP_SKILLS_CSV))
        self.predefined_salaries = self._load_salary_values(Path(PREDEFINED_SALARIES_CSV))

        self.job_title_lookup = {x.lower(): x for x in self.predefined_job_titles}
        self.location_lookup = {x.lower(): x for x in self.predefined_locations}

    @staticmethod
    def _load_salary_values(path: Path) -> list[int]:
        values = []
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                value = (row[0] or "").strip()
                if value.isdigit():
                    values.append(int(value))
        return sorted(set(values))

    def _call_model(self, prompt: str) -> str:
        last_error = None
        for _ in range(MAX_RETRIES + 1):
            try:
                response = self.client.responses.create(
                    model=MAIN_MODEL,
                    input=prompt,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                )
                text = response.output_text.strip()
                if text:
                    return text
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"model_call_failed: {last_error}")

    @staticmethod
    def _blank_result(
        role_relevance: str = "",
        role_relevance_reason: str = "",
        job_category: str = "",
        notes: str = "",
    ) -> dict[str, Any]:
        return {
            "role_relevance": role_relevance,
            "role_relevance_reason": role_relevance_reason,
            "job_category": job_category,
            "job_location": "",
            "remote_preferences": "",
            "remote_days": "",
            "salary_min": "",
            "salary_max": "",
            "salary_currency": "",
            "visa_sponsorship": "",
            "contract_type": "",
            "job_titles": [],
            "seniorities": [],
            "skills": [],
            "notes": notes,
        }

    def _choose_allowed_skills(self, quick_category: str) -> list[str]:
        return self.predefined_tp_skills if quick_category == TP_LABEL else self.predefined_nontp_skills

    def _should_fetch_url(self, description: str) -> bool:
        description = clean_description(description)
        if len(description) < MIN_DESCRIPTION_LENGTH_FOR_NO_FETCH:
            return True

        desc_l = description.lower()
        has_location_signal = any(
            x in desc_l for x in [
                "location:", "job location", "based in", "workplace type",
                "remote", "hybrid", "onsite", "office", "work from home",
            ]
        )
        has_role_signal = any(
            x in desc_l for x in [
                "responsibilities", "requirements", "experience", "about the role",
                "we are looking for", "job type", "salary",
            ]
        )
        return not (has_location_signal and has_role_signal)

    def _build_source_text(self, position_name: str, description: str, job_url: str) -> tuple[str, str]:
        description = clean_description(description)
        notes = []

        source_text = description

        if job_url and self._should_fetch_url(description):
            fetched = fetch_job_page_text(job_url)
            if fetched.ok and fetched.text:
                page_text = clean_description(fetched.text)
                if page_text:
                    if description:
                        source_text = f"{description}\n\n--- JOB PAGE TEXT ---\n\n{page_text}"
                        notes.append("used current description + page text")
                    else:
                        source_text = page_text
                        notes.append("used page text")
            else:
                notes.append(f"url_fetch_failed: {fetched.error or fetched.status_code or 'unknown'}")
        else:
            if description:
                notes.append("used current description")

        return source_text, "; ".join(notes)

    def _coerce_location(self, value: Any) -> str:
        value_s = safe_str(value)
        if not value_s:
            return ""
        if value_s.lower() == "unknown":
            return LOCATION_UNKNOWN

        direct = self.location_lookup.get(value_s.lower())
        if direct:
            return direct

        normalized = normalize_location_match(value_s, self.predefined_locations)
        return normalized or LOCATION_UNKNOWN

    def _coerce_salary_value(self, value: Any) -> str:
        value_s = safe_str(value)
        if not value_s:
            return ""
        value_s = value_s.replace(",", "")
        if not value_s.isdigit():
            return ""
        return closest_salary_value(int(value_s), self.predefined_salaries)

    @staticmethod
    def _coerce_salary_currency(value: Any) -> str:
        cur = safe_str(value).upper()
        return cur if cur in {"GBP", "USD", "EUR", "CAD", "AUD", "CHF"} else ""

    @staticmethod
    def _coerce_visa_status(value: Any, source_text: str) -> str:
        v = safe_str(value).lower()
        text = source_text.lower()

        if "visa" not in text and "sponsorship" not in text:
            return ""

        if v in {"yes", "no"}:
            return v
        return ""

    def _parse_ai_payload(
        self,
        payload: dict[str, Any],
        quick_category: str,
        source_text: str,
        deterministic_skills: list[str],
    ) -> dict[str, Any]:
        role_relevance = normalize_relevance_label(payload.get("role_relevance", ""))
        job_category = normalize_tp_label(payload.get("job_category", "")) or quick_category or NOT_TP_LABEL
        role_relevance_reason = safe_str(payload.get("role_relevance_reason", ""))

        job_location = self._coerce_location(payload.get("job_location", ""))
        remote_preferences_list = normalize_remote_preferences(payload.get("remote_preferences", []))
        remote_preferences = ", ".join(remote_preferences_list) if remote_preferences_list else ""
        remote_days = normalize_remote_days(payload.get("remote_days", ""))

        salary_min = self._coerce_salary_value(payload.get("salary_min", ""))
        salary_max = self._coerce_salary_value(payload.get("salary_max", ""))
        salary_currency = self._coerce_salary_currency(payload.get("salary_currency", ""))

        if not salary_context_exists(source_text):
            salary_min = ""
            salary_max = ""
            salary_currency = ""
        elif salary_min and not salary_max:
            salary_max = salary_min
        elif salary_max and not salary_min:
            salary_min = salary_max

        visa_sponsorship = self._coerce_visa_status(payload.get("visa_sponsorship", ""), source_text)
        contract_type = validate_contract_type(payload.get("contract_type", ""), ALLOWED_CONTRACT_TYPES)

        job_titles = validate_job_titles(payload.get("job_titles", []), self.predefined_job_titles, max_items=MAX_JOB_TITLES)
        seniorities = validate_seniorities(payload.get("seniorities", []), ALLOWED_SENIORITIES, max_items=3)

        allowed_skills = self.predefined_tp_skills if job_category == TP_LABEL else self.predefined_nontp_skills
        ai_skills = validate_skills(payload.get("skills", []), allowed_skills, max_items=MAX_SKILLS)

        supported_ai_skills = [s for s in ai_skills if skill_is_supported(s, source_text)]
        skills = deterministic_skills[:]
        for skill in supported_ai_skills:
            if skill not in skills:
                skills.append(skill)
        skills = skills[:MAX_SKILLS]

        notes = safe_str(payload.get("notes", ""))

        return {
            "role_relevance": role_relevance,
            "role_relevance_reason": role_relevance_reason,
            "job_category": job_category,
            "job_location": job_location,
            "remote_preferences_list": remote_preferences_list,
            "remote_preferences": remote_preferences,
            "remote_days": remote_days,
            "salary_min": salary_min,
            "salary_max": salary_max,
            "salary_currency": salary_currency,
            "visa_sponsorship": visa_sponsorship,
            "contract_type": contract_type,
            "job_titles": job_titles,
            "seniorities": seniorities,
            "skills": skills,
            "notes": notes,
        }

    def _apply_final_consistency(self, result: dict[str, Any], source_text: str) -> dict[str, Any]:
        reason = result.get("role_relevance_reason", "")
        role_relevance = result.get("role_relevance", "")
        job_location = result.get("job_location", "")
        remote_preferences_list = result.get("remote_preferences_list", [])
        position_is_relevant = role_relevance == RELEVANT_LABEL

        if reason_strongly_says_not_relevant(reason):
            result["role_relevance"] = NOT_RELEVANT_LABEL

        if result.get("role_relevance") == RELEVANT_LABEL:
            if not is_location_allowed(job_location, remote_preferences_list, source_text):
                result["role_relevance"] = NOT_RELEVANT_LABEL
                result["job_category"] = result.get("job_category") or NOT_TP_LABEL
                result["role_relevance_reason"] = "Location is outside allowed regions or work-pattern rules."

        if result.get("role_relevance") == NOT_RELEVANT_LABEL:
            return self._blank_result(
                role_relevance=NOT_RELEVANT_LABEL,
                role_relevance_reason=result.get("role_relevance_reason", "") or "Role is outside allowed scope.",
                job_category=result.get("job_category", "") or NOT_TP_LABEL,
                notes=result.get("notes", ""),
            )

        if not position_is_relevant and result.get("role_relevance") == RELEVANT_LABEL and not result.get("role_relevance_reason"):
            result["role_relevance_reason"] = "Role matches allowed location rules and target role scope."

        result["remote_preferences"] = ", ".join(result.get("remote_preferences_list", [])) if result.get("remote_preferences_list") else ""
        return result

    def classify_job(self, row: dict[str, Any]) -> dict[str, Any]:
        position_name = clean_description(row.get("job_title", row.get("position_name", "")))
        job_description = clean_description(row.get("job_description", ""))
        job_url = safe_str(row.get("job_url", ""))

        if looks_like_non_job_content(position_name, job_description):
            return self._blank_result(
                role_relevance=NOT_RELEVANT_LABEL,
                role_relevance_reason="Content is educational, informational, checklist-based, or not a real job posting.",
                job_category=NOT_TP_LABEL,
            )

        excluded, excluded_reason = obvious_excluded_role(position_name, job_description)
        if excluded:
            return self._blank_result(
                role_relevance=NOT_RELEVANT_LABEL,
                role_relevance_reason=excluded_reason,
                job_category=detect_quick_tp_from_title(position_name) or NOT_TP_LABEL,
            )

        quick_category = detect_quick_tp_from_title(position_name) or NOT_TP_LABEL
        source_text, source_note = self._build_source_text(position_name, job_description, job_url)

        allowed_skills = self._choose_allowed_skills(quick_category)
        deterministic_skills = extract_deterministic_skills(
            position_name=position_name,
            description=source_text,
            allowed_skills=allowed_skills,
            max_items=MAX_SKILLS,
        )

        prompt = build_unified_job_extraction_prompt(
            position_name=position_name,
            source_text=source_text,
            predefined_job_titles=self.predefined_job_titles,
            predefined_locations=self.predefined_locations,
            predefined_salaries=self.predefined_salaries,
            allowed_skills=allowed_skills,
        )

        raw = self._call_model(prompt)
        payload = extract_json_object(raw)

        parsed = self._parse_ai_payload(
            payload=payload,
            quick_category=quick_category,
            source_text=source_text,
            deterministic_skills=deterministic_skills,
        )

        if source_note:
            parsed["notes"] = "; ".join([x for x in [parsed.get("notes", ""), source_note] if x])

        if title_has_leadership_signal(position_name):
            parsed["seniorities"] = ["leadership"]

        final_result = self._apply_final_consistency(parsed, source_text)
        return final_result
