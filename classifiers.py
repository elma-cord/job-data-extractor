import csv
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
    detect_quick_tp_from_title,
    extract_deterministic_skills,
    infer_job_titles_from_position_name,
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

        self.location_lookup = {x.lower(): x for x in self.predefined_locations}
        self.job_title_lookup = {x.lower(): x for x in self.predefined_job_titles}

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

    @staticmethod
    def _is_broad_location(value: str) -> bool:
        value_l = (value or "").strip().lower()
        return value_l in {
            "",
            "unknown",
            "uk",
            "united kingdom",
            "england",
            "scotland",
            "wales",
            "northern ireland",
            "ireland",
            "europe",
            "emea",
            "global",
            "worldwide",
        }

    @staticmethod
    def _split_location_variants(raw: str) -> list[str]:
        raw = (raw or "").strip()
        if not raw:
            return []

        raw = re.sub(r"<[^>]+>", " ", raw)
        raw = raw.replace("|", ",")
        raw = raw.replace(";", ",")
        raw = re.sub(r"\s+", " ", raw).strip()

        items = [raw]

        for part in re.split(r"\s+or\s+|/|,|\||;", raw, flags=re.IGNORECASE):
            part = part.strip(" :-")
            if part:
                items.append(part)

        cleaned = []
        for item in items:
            if item and item not in cleaned:
                cleaned.append(item)

        return cleaned

    def _extract_weighted_location_candidates(self, text: str) -> list[tuple[int, int, str]]:
        text = clean_description(text)
        if not text:
            return []

        candidates: list[tuple[int, int, str]] = []

        patterns = [
            (240, r"(?im)^\s*location\s*$\n^\s*(.+)$"),
            (230, r"(?im)^\s*####\s*location\s*$\n^\s*(.+)$"),
            (230, r"(?im)^\s*location\s*[:\-]\s*(.+)$"),
            (225, r"(?im)^\s*job location\s*[:\-]\s*(.+)$"),
            (220, r"(?im)^\s*office location\s*[:\-]\s*(.+)$"),
            (210, r"(?im)^\s*city\s*[:\-]\s*(.+)$"),
            (205, r"(?im)^\s*based in\s+(.+)$"),
            (200, r"(?im)\brole is based at (?:our )?(.+?)(?: office|\.)"),
            (140, r"(?im)^\s*where you[’']ll work\s*[:\-]?\s*(.+)$"),
            (110, r"(?im)\bhub based\s*\((.+?)\)"),
        ]

        text_len = max(len(text), 1)

        for base_weight, pattern in patterns:
            for match in re.finditer(pattern, text):
                value = (match.group(1) or "").strip()
                if not value:
                    continue

                value = re.split(
                    r"(?i)\b(hybrid|remote|onsite|salary|schedule|travel required|shift|clearance|required|benefits|reporting to)\b",
                    value,
                    maxsplit=1,
                )[0].strip(" ,:-")

                recency_bonus = int((match.start() / text_len) * 80)

                for variant in self._split_location_variants(value):
                    candidates.append((base_weight + recency_bonus, match.start(), variant))

        return candidates

    def _deterministic_location_from_text(self, text: str) -> str:
        weighted = self._extract_weighted_location_candidates(text)
        if not weighted:
            return ""

        best_value = ""
        best_score = -10**9
        best_pos = -1

        for weight, pos, raw in weighted:
            normalized = normalize_location_match(raw, self.predefined_locations)
            if not normalized:
                continue

            score = weight

            if not self._is_broad_location(normalized):
                score += 70

            if "," in normalized:
                score += 15

            if self._is_broad_location(normalized):
                score -= 120

            if score > best_score or (score == best_score and pos > best_pos):
                best_score = score
                best_value = normalized
                best_pos = pos

        return best_value

    def _description_has_ambiguous_location(self, description: str) -> bool:
        d = clean_description(description).lower()
        ambiguous_markers = [
            " hub based ",
            "all our roles are hub based",
            "multiple locations",
            "bristol, glasgow or london",
            "kelso or bathgate",
        ]
        return any(marker in d for marker in ambiguous_markers)

    def _should_fetch_url(self, description: str) -> bool:
        description = clean_description(description)
        if len(description) < MIN_DESCRIPTION_LENGTH_FOR_NO_FETCH:
            return True

        if self._description_has_ambiguous_location(description):
            return True

        desc_l = description.lower()
        has_location_signal = any(
            x in desc_l for x in [
                "location:", "job location", "based in", "where you’ll work", "where you'll work",
                "remote", "hybrid", "onsite", "office", "work from home",
            ]
        )
        has_role_signal = any(
            x in desc_l for x in [
                "responsibilities", "requirements", "experience", "about the role",
                "we are looking for", "job type", "salary", "role purpose",
            ]
        )
        return not (has_location_signal and has_role_signal)

    def _build_source_text(self, description: str, job_url: str) -> tuple[str, str]:
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

    def _coerce_location(self, ai_location: Any, source_text: str) -> str:
        ai_location_s = safe_str(ai_location)
        direct = ""
        if ai_location_s:
            if ai_location_s.lower() == "unknown":
                direct = LOCATION_UNKNOWN
            else:
                direct = normalize_location_match(ai_location_s, self.predefined_locations) or LOCATION_UNKNOWN

        deterministic = self._deterministic_location_from_text(source_text)

        if deterministic and (self._is_broad_location(direct) or not direct):
            return deterministic

        if deterministic and direct and self._is_broad_location(direct) and not self._is_broad_location(deterministic):
            return deterministic

        return direct or deterministic or LOCATION_UNKNOWN

    def _coerce_salary_value(self, value: Any) -> str:
        value_s = safe_str(value).replace(",", "")
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

    @staticmethod
    def _clean_skill_list(skills: list[str], source_text: str, max_items: int) -> list[str]:
        cleaned = []
        for skill in skills:
            if skill and skill not in cleaned and skill_is_supported(skill, source_text):
                cleaned.append(skill)
        return cleaned[:max_items]

    @staticmethod
    def _finalize_seniorities(position_name: str, seniorities: list[str]) -> list[str]:
        title = clean_description(position_name).lower()

        if title_has_leadership_signal(position_name):
            return ["leadership"]

        if "manager" in title:
            return ["senior", "lead"]

        if "assistant" in title:
            return ["entry", "junior"]

        if "administrator" in title and not seniorities:
            return ["mid"]

        if not seniorities:
            if "engineer" in title or "developer" in title or "analyst" in title:
                return ["mid"]

        return seniorities[:3]

    def _filter_job_titles(self, position_name: str, job_titles: list[str]) -> list[str]:
        title_l = clean_description(position_name).lower()
        titles = job_titles[:]

        strongly_technical = any(
            term in title_l
            for term in [
                "engineer",
                "it ",
                "it-",
                "support",
                "application engineer",
                "1st line",
                "2nd line",
                "3rd line",
                "systems",
                "network",
                "infrastructure",
                "administrator",
            ]
        )

        if strongly_technical:
            banned_for_technical = {
                "Customer Service Representative",
                "Customer Support",
            }
            titles = [t for t in titles if t not in banned_for_technical]

        return titles[:MAX_JOB_TITLES]

    def _fallback_job_titles(self, position_name: str, source_text: str, existing_titles: list[str]) -> list[str]:
        out = existing_titles[:]

        if out:
            out = self._filter_job_titles(position_name, out)

        if not out:
            inferred = infer_job_titles_from_position_name(position_name, self.predefined_job_titles)
            if inferred:
                out = inferred[:MAX_JOB_TITLES]

        title_l = clean_description(position_name).lower()
        text_l = clean_description(source_text).lower()

        if not out:
            if "1st line" in title_l or "first line" in title_l:
                for candidate in ["Support Engineer", "System Administrator", "System Engineer"]:
                    if candidate in self.predefined_job_titles and candidate not in out:
                        out.append(candidate)
                out = out[:MAX_JOB_TITLES]

        if not out:
            if "application engineer" in title_l:
                for candidate in ["System Engineer", "Support Engineer", "Solutions Engineer"]:
                    if candidate in self.predefined_job_titles and candidate not in out:
                        out.append(candidate)
                out = out[:MAX_JOB_TITLES]

        if "analytics manager" in title_l:
            if "Business Analyst" in self.predefined_job_titles and "Business Analyst" not in out:
                out.append("Business Analyst")
            if "Data/Insight Analyst" in self.predefined_job_titles and "Data/Insight Analyst" not in out:
                out.append("Data/Insight Analyst")
            out = out[:MAX_JOB_TITLES]

        if not out and "accounts payable" in title_l:
            for candidate in ["Finance/Accounting", "Operations"]:
                if candidate in self.predefined_job_titles and candidate not in out:
                    out.append(candidate)
            out = out[:MAX_JOB_TITLES]

        out = self._filter_job_titles(position_name, out)
        return out[:MAX_JOB_TITLES]

    def _parse_ai_payload(self, payload: dict[str, Any], source_text: str) -> dict[str, Any]:
        role_relevance = normalize_relevance_label(payload.get("role_relevance", ""))
        job_category = normalize_tp_label(payload.get("job_category", "")) or NOT_TP_LABEL
        role_relevance_reason = safe_str(payload.get("role_relevance_reason", ""))

        job_location = self._coerce_location(payload.get("job_location", ""), source_text)
        remote_preferences_list = normalize_remote_preferences(payload.get("remote_preferences", []))
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
        notes = safe_str(payload.get("notes", ""))

        return {
            "role_relevance": role_relevance,
            "role_relevance_reason": role_relevance_reason,
            "job_category": job_category,
            "job_location": job_location,
            "remote_preferences_list": remote_preferences_list,
            "remote_days": remote_days,
            "salary_min": salary_min,
            "salary_max": salary_max,
            "salary_currency": salary_currency,
            "visa_sponsorship": visa_sponsorship,
            "contract_type": contract_type,
            "job_titles": job_titles,
            "seniorities": seniorities,
            "notes": notes,
        }

    def _apply_final_consistency(self, result: dict[str, Any], source_text: str) -> dict[str, Any]:
        if reason_strongly_says_not_relevant(result.get("role_relevance_reason", "")):
            result["role_relevance"] = NOT_RELEVANT_LABEL

        if result.get("role_relevance") == RELEVANT_LABEL:
            if not is_location_allowed(
                result.get("job_location", ""),
                result.get("remote_preferences_list", []),
                source_text,
            ):
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

        source_text, source_note = self._build_source_text(job_description, job_url)

        raw = self._call_model(
            build_unified_job_extraction_prompt(
                position_name=position_name,
                source_text=source_text,
                predefined_job_titles=self.predefined_job_titles,
                predefined_locations=self.predefined_locations,
                predefined_salaries=self.predefined_salaries,
                allowed_tp_skills=self.predefined_tp_skills,
                allowed_nontp_skills=self.predefined_nontp_skills,
            )
        )
        payload = extract_json_object(raw)

        parsed = self._parse_ai_payload(payload, source_text)
        parsed["seniorities"] = self._finalize_seniorities(position_name, parsed.get("seniorities", []))

        allowed_skills = self.predefined_tp_skills if parsed["job_category"] == TP_LABEL else self.predefined_nontp_skills

        deterministic_skills = extract_deterministic_skills(
            position_name=position_name,
            description=source_text,
            allowed_skills=allowed_skills,
            max_items=MAX_SKILLS,
        )

        ai_skills = validate_skills(payload.get("skills", []), allowed_skills, max_items=MAX_SKILLS)
        ai_skills = self._clean_skill_list(ai_skills, source_text, MAX_SKILLS)

        final_skills = deterministic_skills[:]
        for skill in ai_skills:
            if skill not in final_skills:
                final_skills.append(skill)
        parsed["skills"] = final_skills[:MAX_SKILLS]

        parsed["job_titles"] = self._fallback_job_titles(
            position_name=position_name,
            source_text=source_text,
            existing_titles=parsed.get("job_titles", []),
        )

        if source_note:
            parsed["notes"] = "; ".join([x for x in [parsed.get("notes", ""), source_note] if x])

        final_result = self._apply_final_consistency(parsed, source_text)
        return final_result
