import csv
import json
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from config import (
    MAIN_MODEL,
    MAX_JOB_TITLES,
    MAX_RETRIES,
    MAX_SKILLS,
    OPENAI_TIMEOUT_SECONDS,
    PREDEFINED_JOB_TITLES_CSV,
    PREDEFINED_LOCATIONS_CSV,
    PREDEFINED_NONTP_SKILLS_CSV,
    PREDEFINED_SALARIES_CSV,
    PREDEFINED_TP_SKILLS_CSV,
)
from fetch_extract import fetch_job_page_text
from prompts import (
    build_contract_type_prompt,
    build_job_titles_prompt,
    build_role_relevance_prompt,
    build_seniority_prompt,
    build_skills_prompt,
)
from rules import (
    detect_basic_relevance_from_title,
    detect_contract_type,
    detect_seniority_from_title_and_description,
    detect_tp_from_title,
    extract_remote_days,
    extract_remote_preferences,
    extract_salary,
    extract_visa_status,
    load_single_column_csv,
    location_or_remote_missing,
    normalize_text,
)
from validators import (
    clean_description,
    parse_role_relevance_response,
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

        self.location_lookup = self._build_location_lookup(self.predefined_locations)
        self.job_title_lookup = {x.lower(): x for x in self.predefined_job_titles}

    @staticmethod
    def _load_salary_values(path: Path) -> list[int]:
        values = []
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                first = (row[0] or "").strip()
                if not first:
                    continue
                if not re.fullmatch(r"\d+", first):
                    continue
                values.append(int(first))
        return sorted(set(values))

    @staticmethod
    def _build_location_lookup(locations: list[str]) -> dict[str, str]:
        lookup = {}
        for loc in locations:
            norm = loc.strip().lower()
            if norm:
                lookup[norm] = loc
        return lookup

    def _call_model(self, prompt: str) -> str:
        last_error = None

        for _ in range(MAX_RETRIES + 1):
            try:
                response = self.client.responses.create(
                    model=MAIN_MODEL,
                    input=prompt,
                    max_output_tokens=300,
                )
                text = response.output_text.strip()
                if text:
                    return text
            except Exception as exc:
                last_error = exc

        raise RuntimeError(f"model_call_failed: {last_error}")

    def _extract_location_from_text(self, text: str) -> str:
        t = (text or "").lower()

        # exact/predefined location matching first, longest match first
        for key in sorted(self.location_lookup.keys(), key=len, reverse=True):
            if key and key in t:
                return self.location_lookup[key]

        # useful broad fallbacks
        broad_patterns = [
            (r"\blondon\b", "London"),
            (r"\bunited kingdom\b|\buk\b", "United Kingdom"),
            (r"\bengland\b", "England"),
            (r"\bscotland\b", "Scotland"),
            (r"\bwales\b", "Wales"),
            (r"\bnorthern ireland\b", "Northern Ireland"),
            (r"\bireland\b", "Ireland"),
            (r"\bemea\b", "EMEA"),
            (r"\beurope\b", "Europe"),
        ]
        for pattern, label in broad_patterns:
            if re.search(pattern, t):
                return label

        return ""

    def _extract_location_remote_from_description_or_url(
        self,
        description: str,
        job_url: str,
    ) -> dict[str, str]:
        desc = clean_description(description)

        job_location = self._extract_location_from_text(desc)
        remote_preferences_list = extract_remote_preferences(desc)
        remote_days = extract_remote_days(desc)

        notes = []

        if job_location:
            notes.append("location from current job description")
        if remote_preferences_list:
            notes.append("remote preferences from current job description")

        should_fetch = (not job_location) or (not remote_preferences_list)

        if should_fetch and job_url:
            fetched = fetch_job_page_text(job_url)

            if fetched.ok and fetched.text:
                page_text = clean_description(fetched.text)

                if not job_location:
                    fetched_location = self._extract_location_from_text(page_text)
                    if fetched_location:
                        job_location = fetched_location
                        notes.append("location via link")

                if not remote_preferences_list:
                    fetched_remote = extract_remote_preferences(page_text)
                    if fetched_remote:
                        remote_preferences_list = fetched_remote
                        notes.append("remote preferences via link")

                if remote_days == "not specified":
                    fetched_remote_days = extract_remote_days(page_text)
                    if fetched_remote_days != "not specified":
                        remote_days = fetched_remote_days
                        notes.append("remote days via link")
            else:
                notes.append(f"url_fetch_failed: {fetched.error or fetched.status_code or 'unknown'}")

        return {
            "job_location": job_location,
            "remote_preferences": ", ".join(remote_preferences_list) if remote_preferences_list else "",
            "remote_days": remote_days,
            "source_notes": "; ".join(notes),
        }

    def _classify_relevance_and_category(self, position_name: str, job_description: str) -> dict[str, str]:
        quick_rel, quick_reason = detect_basic_relevance_from_title(position_name)
        quick_tp = detect_tp_from_title(position_name)

        # Fast reject only when very obvious
        if quick_rel == "Not Relevant":
            return {
                "role_relevance": "Not Relevant",
                "job_category": quick_tp or "Not T&P",
                "role_relevance_reason": quick_reason,
            }

        prompt = build_role_relevance_prompt(
            position_name=position_name,
            job_description=job_description,
            predefined_job_titles=self.predefined_job_titles,
        )
        raw = self._call_model(prompt)
        parsed = parse_role_relevance_response(raw)

        # fallback if model response is imperfect
        if not parsed["role_relevance"]:
            parsed["role_relevance"] = quick_rel or ""
        if not parsed["job_category"]:
            parsed["job_category"] = quick_tp or ""

        return parsed

    def _classify_job_titles(self, position_name: str, job_description: str) -> list[str]:
        title_clean = normalize_text(position_name)
        exact = self.job_title_lookup.get(title_clean.lower())
        if exact:
            return [exact]

        prompt = build_job_titles_prompt(
            position_name=position_name,
            job_description=job_description,
            predefined_job_titles=self.predefined_job_titles,
        )
        raw = self._call_model(prompt)
        return validate_job_titles(raw, self.predefined_job_titles, max_items=MAX_JOB_TITLES)

    def _classify_seniority(self, position_name: str, job_description: str) -> list[str]:
        rule_result = detect_seniority_from_title_and_description(position_name, job_description)

        # If rule is decisive enough, keep it
        if rule_result in (["leadership"], ["lead"], ["senior"], ["junior"], ["mid"], ["entry"]):
            return rule_result

        prompt = build_seniority_prompt(position_name=position_name, job_description=job_description)
        raw = self._call_model(prompt)
        validated = validate_seniorities(raw, ["entry", "junior", "mid", "senior", "lead", "leadership"], max_items=3)

        if validated:
            return validated
        return rule_result

    def _classify_skills(self, position_name: str, job_description: str, job_category: str) -> list[str]:
        allowed_skills = (
            self.predefined_tp_skills
            if job_category == "T&P job"
            else self.predefined_nontp_skills
        )

        if not allowed_skills:
            return []

        prompt = build_skills_prompt(
            position_name=position_name,
            job_description=job_description,
            role_category_label=job_category,
            allowed_skills=allowed_skills,
        )
        raw = self._call_model(prompt)
        return validate_skills(raw, allowed_skills, max_items=MAX_SKILLS)

    def classify_job(self, row: dict[str, Any]) -> dict[str, Any]:
        position_name = clean_description(row.get("job_title", row.get("position_name", "")))
        job_description = clean_description(row.get("job_description", ""))
        job_url = (row.get("job_url", "") or "").strip()

        notes = []

        # 1) description-only rule-based fields
        location_remote = self._extract_location_remote_from_description_or_url(
            description=job_description,
            job_url=job_url,
        )
        notes.append(location_remote.get("source_notes", ""))

        salary = extract_salary(job_description, self.predefined_salaries)
        visa_sponsorship = extract_visa_status(job_description)

        contract_type = detect_contract_type(job_description)
        if not contract_type:
            # AI fallback only if needed
            raw_contract = self._call_model(build_contract_type_prompt(job_description))
            contract_type = validate_contract_type(raw_contract)

        # 2) relevance/category
        relevance_data = self._classify_relevance_and_category(position_name, job_description)

        result = {
            "role_relevance": relevance_data.get("role_relevance", ""),
            "role_relevance_reason": relevance_data.get("role_relevance_reason", ""),
            "job_category": relevance_data.get("job_category", ""),

            "job_location": location_remote.get("job_location", ""),
            "remote_preferences": location_remote.get("remote_preferences", ""),
            "remote_days": location_remote.get("remote_days", ""),

            "salary_min": salary.get("salary_min", ""),
            "salary_max": salary.get("salary_max", ""),
            "salary_currency": salary.get("salary_currency", ""),

            "visa_sponsorship": visa_sponsorship,
            "contract_type": contract_type,

            "job_titles": [],
            "seniorities": [],
            "skills": [],
            "notes": "",
        }

        # 3) early stop
        if result["role_relevance"] == "Not Relevant":
            result["notes"] = "; ".join([x for x in notes if x])
            return result

        # 4) semantic fields only for relevant jobs
        result["job_titles"] = self._classify_job_titles(position_name, job_description)
        result["seniorities"] = self._classify_seniority(position_name, job_description)

        if result["job_category"] in {"T&P job", "Not T&P"}:
            result["skills"] = self._classify_skills(
                position_name=position_name,
                job_description=job_description,
                job_category=result["job_category"],
            )

        result["notes"] = "; ".join([x for x in notes if x])
        return result
