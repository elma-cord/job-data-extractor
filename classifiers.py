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
    extract_location_candidates,
    extract_remote_days,
    extract_remote_preferences,
    extract_salary,
    extract_visa_status,
    infer_job_titles_from_position_name,
    is_location_allowed,
    load_single_column_csv,
    normalize_text,
    select_best_location,
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
        self.job_title_lookup = {x.lower(): x for x in self.predefined_job_titles}
        self.location_lookup = {x.lower(): x for x in self.predefined_locations}
        self.location_records = self._build_location_records(self.predefined_locations)
        self.broad_location_keys = {
            "united kingdom",
            "uk",
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
    def _load_salary_values(path: Path) -> list[int]:
        values = []
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                first = (row[0] or "").strip()
                if first and re.fullmatch(r"\d+", first):
                    values.append(int(first))
        return sorted(set(values))

    def _build_location_records(self, locations: list[str]) -> list[dict[str, Any]]:
        records = []
        for loc in locations:
            parts = [p.strip() for p in loc.split(",") if p.strip()]
            first_part = parts[0] if parts else loc
            key = self._canonical_location_key(loc)
            first_part_key = self._canonical_location_key(first_part)
            part_keys = {self._canonical_location_key(p) for p in parts if p.strip()}
            records.append(
                {
                    "value": loc,
                    "key": key,
                    "parts": parts,
                    "part_keys": part_keys,
                    "first_part_key": first_part_key,
                    "tokens": set(key.split()),
                    "is_uk": "uk" in key or "united kingdom" in key,
                    "len": len(loc),
                    "specificity": max(len(parts), 1),
                    "is_broad": first_part_key in {
                        "united kingdom", "uk", "england", "scotland", "wales",
                        "northern ireland", "ireland", "europe", "emea"
                    },
                }
            )
        return records

    def _call_model(self, prompt: str) -> str:
        last_error = None
        for _ in range(MAX_RETRIES + 1):
            try:
                response = self.client.responses.create(
                    model=MAIN_MODEL,
                    input=prompt,
                    max_output_tokens=400,
                )
                text = response.output_text.strip()
                if text:
                    return text
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"model_call_failed: {last_error}")

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
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

    def _build_location_remote_prompt(
        self,
        position_name: str,
        text: str,
        source_name: str,
    ) -> str:
        locations_str = ", ".join(self.predefined_locations)

        return f"""
You will receive:
1. position name
2. source text
3. source type

Task:
Extract exactly these three fields:
- job_location
- remote_preferences
- remote_days

Rules for job_location:
1. Read the text carefully and find the ACTUAL job location.
2. Prefer explicit location fields such as:
   "Location", "Locations", "All Locations", "Job Location", "Office Location", "Based in".
3. Ignore unrelated text such as:
   benefits, DEI text, company office lists, slogans, skills, tools, generic company text, headings, random short lines.
4. If multiple locations appear, choose the most specific real job location for the role.
5. Normalize the result to EXACTLY one value from the acceptable locations list below.
6. If no acceptable location can be identified, return "Unknown".

Rules for remote_preferences:
1. Allowed values are only: onsite, hybrid, remote
2. Return an array.
3. Prefer explicit labels like:
   "Workplace type", "Remote status", "Working pattern", "Work type"
4. If text says hybrid, return ["hybrid"]
5. If text says onsite, return ["onsite"]
6. If text says fully remote / remote-first / home-based, return ["remote"]
7. Do not infer remote from phrases like "remote support" unless it clearly describes the job working arrangement.
8. If not specified, return []

Rules for remote_days:
1. Return a string.
2. If not specified, return "not specified"
3. If the text says 1 day in office, return "4"
4. If it says 2 days in office, return "3"
5. If it says 1-2 days in office, return "3"
6. If it says 2-3 days in office, return "2"
7. If it says fully remote, return "not specified"
8. Only return a number when clearly supported.

Output:
Return ONLY valid JSON in exactly this format:
{{
  "job_location": "one exact value from acceptable locations list or Unknown",
  "remote_preferences": ["onsite" or "hybrid" or "remote"],
  "remote_days": "number or not specified"
}}

Acceptable locations list:
{locations_str}

Position name:
{position_name}

Source type:
{source_name}

Source text:
{text}
""".strip()

    @staticmethod
    def _clean_location_candidate_text(value: str) -> str:
        value = (value or "").strip()
        if not value:
            return ""

        value = re.sub(r"<[^>]+>", " ", value)
        value = value.replace("｜", "|")
        value = value.replace("／", "/")
        value = value.replace("&amp;", "&")

        # Common ATS/location formatting cleanup
        value = re.sub(r"(?i)^\s*locations?\s*:\s*", "", value)
        value = re.sub(r"(?i)^\s*all locations?\s*:\s*", "", value)
        value = re.sub(r"(?i)^\s*job location\s*:\s*", "", value)
        value = re.sub(r"(?i)^\s*office location\s*:\s*", "", value)

        # Remove work arrangement words from candidate location text
        value = re.sub(r"(?i)\bhybrid\b", " ", value)
        value = re.sub(r"(?i)\bonsite\b", " ", value)
        value = re.sub(r"(?i)\bremote\b", " ", value)
        value = re.sub(r"(?i)\bhome[- ]based\b", " ", value)
        value = re.sub(r"(?i)\bworking pattern\b", " ", value)
        value = re.sub(r"(?i)\bworkplace type\b", " ", value)
        value = re.sub(r"(?i)\bremote status\b", " ", value)

        # Handle UK-Birmingham like forms
        value = re.sub(r"(?i)\buk\s*-\s*([A-Za-z][A-Za-z\s\-]+)\b", r"\1, UK", value)
        value = re.sub(r"(?i)\bunited kingdom\s*-\s*([A-Za-z][A-Za-z\s\-]+)\b", r"\1, United Kingdom", value)

        # Remove junk placeholders
        value = re.sub(r"(?i)\bunavailable\b", " ", value)
        value = re.sub(r"(?i)\{\{[^}]+\}\}", " ", value)

        # Reorder "United Kingdom, London" -> "London, United Kingdom"
        m = re.match(r"(?i)^\s*(united kingdom|uk|england|scotland|wales|northern ireland)\s*,\s*([A-Za-z][A-Za-z\s\-]+)\s*$", value)
        if m:
            value = f"{m.group(2)}, {m.group(1)}"

        value = value.replace("|", ",")
        value = value.replace(";", ",")
        value = re.sub(r"\s*-\s*", ", ", value)
        value = re.sub(r"\s*/\s*", ", ", value)
        value = re.sub(r"\s+", " ", value)
        value = re.sub(r"\s*,\s*", ", ", value)
        value = re.sub(r"(,\s*){2,}", ", ", value)
        value = value.strip(" ,:-")
        return value.strip()

    @staticmethod
    def _canonical_location_key(value: str) -> str:
        value = (value or "").lower()
        value = re.sub(r"<[^>]+>", " ", value)
        value = value.replace("&", " and ")
        value = re.sub(r"(?i)\bunavailable\b", " ", value)
        value = re.sub(r"(?i)\bgreat britain\b", " united kingdom ", value)
        value = re.sub(r"(?i)\bu\.?k\.?\b", " uk ", value)
        value = re.sub(r"(?i)\bengland\b", " uk ", value)
        value = re.sub(r"(?i)\bscotland\b", " uk ", value)
        value = re.sub(r"(?i)\bwales\b", " uk ", value)
        value = re.sub(r"(?i)\bnorthern ireland\b", " uk ", value)
        value = re.sub(r"(?i)\bhybrid\b", " ", value)
        value = re.sub(r"(?i)\bonsite\b", " ", value)
        value = re.sub(r"(?i)\bremote\b", " ", value)
        value = re.sub(r"[^a-z0-9]+", " ", value)
        value = re.sub(r"\s+", " ", value).strip()
        return value

    @staticmethod
    def _looks_like_bad_location_candidate(value: str) -> bool:
        if not value:
            return True

        lowered = value.lower()
        bad_phrases = [
            "diversity",
            "equity",
            "inclusion",
            "account executive",
            "as an ",
            "you will",
            "insurance brokers",
            "mgas",
            "insurers",
            "benefits",
            "about us",
            "job description",
            "responsibilities",
            "experience",
            "department",
            "reporting to",
            "posted today",
            "job requisition",
            "full time",
            "regular",
            "fixed term",
        ]
        if any(p in lowered for p in bad_phrases):
            return True

        words = re.findall(r"[a-zA-Z]+", value)
        if len(words) > 8:
            return True

        return False

    def _candidate_variants(self, value: str) -> list[str]:
        value = self._clean_location_candidate_text(value)
        if not value:
            return []

        variants: list[str] = []

        def add(v: str) -> None:
            v = self._clean_location_candidate_text(v)
            if v and v not in variants and not self._looks_like_bad_location_candidate(v):
                variants.append(v)

        add(value)

        parts = [p.strip() for p in value.split(",") if p.strip()]
        if len(parts) >= 2:
            # More specific to broader
            add(", ".join(parts))
            add(", ".join(parts[:2]))
            add(", ".join(parts[-2:]))
            add(parts[0])

            # Prefer non-country part when first part is broad
            if self._canonical_location_key(parts[0]) in self.broad_location_keys and len(parts) >= 2:
                add(", ".join(parts[1:]))

            # Prefer region/city part before country if last part is broad
            if self._canonical_location_key(parts[-1]) in self.broad_location_keys:
                add(", ".join(parts[:-1]))
                if len(parts) >= 2:
                    add(parts[-2])

        # Handle "West Midlands United Kingdom" type leftovers
        tokens = value.split()
        if len(tokens) >= 2:
            add(" ".join(tokens))

        return variants

    def _score_location_record(self, cand_key: str, cand_tokens: set[str], record: dict[str, Any]) -> int:
        score = 0

        if cand_key == record["key"]:
            score += 1000

        if cand_key == record["first_part_key"]:
            score += 900

        if cand_key in record["part_keys"]:
            score += 850

        if record["first_part_key"] and (
            cand_key in record["first_part_key"] or record["first_part_key"] in cand_key
        ):
            score += 250

        overlap = len(cand_tokens & record["tokens"])
        if overlap:
            score += overlap * 40

        if cand_tokens and cand_tokens.issubset(record["tokens"]):
            score += 180

        # Specificity bonus: prefer city/region over country-only values
        if not record["is_broad"]:
            score += 80 + (record["specificity"] * 10)
        else:
            score -= 120

        if "uk" in cand_tokens and record["is_uk"]:
            score += 15

        # Penalize very broad matches when candidate is more specific
        if len(cand_tokens) >= 2 and record["is_broad"]:
            score -= 180

        return score

    def _normalize_location_candidate(self, value: str) -> str:
        variants = self._candidate_variants(value)
        if not variants:
            return ""

        best_value = ""
        best_score = -10**9

        for variant in variants:
            direct = self.location_lookup.get(variant.lower())
            if direct:
                return direct

            cand_key = self._canonical_location_key(variant)
            if not cand_key:
                continue

            cand_tokens = set(cand_key.split())
            if not cand_tokens:
                continue

            for record in self.location_records:
                score = self._score_location_record(cand_key, cand_tokens, record)

                if score > best_score or (
                    score == best_score
                    and score > -10**8
                    and record["len"] < len(best_value or "z" * 999)
                ):
                    best_score = score
                    best_value = record["value"]

        if best_score >= 220:
            return best_value

        return ""

    def _normalize_location_candidates(self, candidates: list[str]) -> list[str]:
        out = []
        for candidate in candidates:
            normalized = self._normalize_location_candidate(candidate)
            if normalized and normalized not in out:
                out.append(normalized)
        return out

    def _normalize_model_location(self, value: str) -> str:
        return self._normalize_location_candidate(value)

    def _normalize_model_remote_preferences(self, value: Any) -> list[str]:
        allowed = {"onsite", "hybrid", "remote"}

        if isinstance(value, list):
            raw_items = value
        elif isinstance(value, str):
            raw_items = [x.strip() for x in value.split(",")]
        else:
            raw_items = []

        cleaned = []
        for item in raw_items:
            item_l = str(item).strip().lower()
            if item_l in allowed and item_l not in cleaned:
                cleaned.append(item_l)

        ordered = [x for x in ["onsite", "hybrid", "remote"] if x in cleaned]

        if "hybrid" in ordered and "remote" in ordered:
            ordered = [x for x in ordered if x != "remote"]

        return ordered

    @staticmethod
    def _normalize_model_remote_days(value: Any) -> str:
        value = str(value or "").strip().lower()
        if not value:
            return "not specified"
        if value == "not specified":
            return "not specified"
        if re.fullmatch(r"[0-5]", value):
            return value
        return "not specified"

    def _extract_location_remote_with_ai(
        self,
        position_name: str,
        text: str,
        source_name: str,
    ) -> dict[str, Any]:
        text = clean_description(text)
        if not text:
            return {
                "job_location": "",
                "remote_preferences_list": [],
                "remote_days": "not specified",
            }

        prompt = self._build_location_remote_prompt(
            position_name=position_name,
            text=text,
            source_name=source_name,
        )
        raw = self._call_model(prompt)
        data = self._extract_json_object(raw)

        return {
            "job_location": self._normalize_model_location(data.get("job_location", "")),
            "remote_preferences_list": self._normalize_model_remote_preferences(data.get("remote_preferences", [])),
            "remote_days": self._normalize_model_remote_days(data.get("remote_days", "not specified")),
        }

    def _extract_location_remote_from_description_or_url(
        self,
        position_name: str,
        description: str,
        job_url: str,
    ) -> dict[str, Any]:
        desc = clean_description(description)

        desc_ai = self._extract_location_remote_with_ai(
            position_name=position_name,
            text=desc,
            source_name="job description",
        )

        desc_rule_location_candidates_raw = extract_location_candidates(desc, self.predefined_locations)
        desc_rule_location_candidates = self._normalize_location_candidates(desc_rule_location_candidates_raw)
        desc_rule_job_location = select_best_location(desc_rule_location_candidates)
        desc_rule_remote_preferences = extract_remote_preferences(desc)
        desc_rule_remote_days = extract_remote_days(desc)

        desc_job_location = desc_ai["job_location"] or desc_rule_job_location
        desc_location_candidates = [desc_job_location] if desc_job_location else desc_rule_location_candidates[:]

        desc_remote_preferences_list = (
            desc_ai["remote_preferences_list"] if desc_ai["remote_preferences_list"] else desc_rule_remote_preferences
        )

        desc_remote_days = (
            desc_ai["remote_days"]
            if desc_ai["remote_days"] != "not specified"
            else desc_rule_remote_days
        )

        job_location = desc_job_location
        location_candidates = desc_location_candidates[:]
        remote_preferences_list = desc_remote_preferences_list[:]
        remote_days = desc_remote_days

        notes = []
        if job_location:
            notes.append("location from current job description")
        if remote_preferences_list:
            notes.append("remote preferences from current job description")
        if remote_days != "not specified":
            notes.append("remote days from current job description")

        should_fetch = (
            not job_location
            or not remote_preferences_list
            or remote_days == "not specified"
        )

        if should_fetch and job_url:
            fetched = fetch_job_page_text(job_url)

            if fetched.ok and fetched.text:
                page_text = clean_description(fetched.text)

                page_ai = self._extract_location_remote_with_ai(
                    position_name=position_name,
                    text=page_text,
                    source_name="job page text",
                )

                page_rule_location_candidates_raw = extract_location_candidates(page_text, self.predefined_locations)
                page_rule_location_candidates = self._normalize_location_candidates(page_rule_location_candidates_raw)
                page_rule_job_location = select_best_location(page_rule_location_candidates)
                page_rule_remote_preferences = extract_remote_preferences(page_text)
                page_rule_remote_days = extract_remote_days(page_text)

                page_job_location = page_ai["job_location"] or page_rule_job_location
                page_location_candidates = [page_job_location] if page_job_location else page_rule_location_candidates[:]

                page_remote_preferences = (
                    page_ai["remote_preferences_list"]
                    if page_ai["remote_preferences_list"]
                    else page_rule_remote_preferences
                )

                page_remote_days = (
                    page_ai["remote_days"]
                    if page_ai["remote_days"] != "not specified"
                    else page_rule_remote_days
                )

                if not job_location and page_job_location:
                    job_location = page_job_location
                    location_candidates = page_location_candidates[:]
                    notes.append("location via link")

                if not remote_preferences_list and page_remote_preferences:
                    remote_preferences_list = page_remote_preferences[:]
                    notes.append("remote preferences via link")

                if remote_days == "not specified" and page_remote_days != "not specified":
                    remote_days = page_remote_days
                    notes.append("remote days via link")
            else:
                notes.append(f"url_fetch_failed: {fetched.error or fetched.status_code or 'unknown'}")

        return {
            "job_location": job_location,
            "job_location_candidates": location_candidates,
            "remote_preferences_list": remote_preferences_list,
            "remote_preferences": ", ".join(remote_preferences_list) if remote_preferences_list else "",
            "remote_days": remote_days,
            "source_notes": "; ".join(notes),
        }

    def _classify_relevance_and_category(
        self,
        position_name: str,
        job_description: str,
        location_candidates: list[str],
        remote_preferences_list: list[str],
    ) -> dict[str, str]:
        quick_rel, quick_reason = detect_basic_relevance_from_title(position_name)
        quick_tp = detect_tp_from_title(position_name)

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

        if not parsed["role_relevance"]:
            parsed["role_relevance"] = quick_rel or ""
        if not parsed["job_category"]:
            parsed["job_category"] = quick_tp or ""

        if (
            parsed["role_relevance"] == "Not Relevant"
            and quick_rel == "Relevant"
            and is_location_allowed(location_candidates, remote_preferences_list)
        ):
            parsed["role_relevance"] = "Relevant"
            parsed["role_relevance_reason"] = (
                parsed["role_relevance_reason"]
                or "Title matches target scope and at least one acceptable location was found."
            )

        if (
            parsed["role_relevance"] == "Relevant"
            and location_candidates
            and not is_location_allowed(location_candidates, remote_preferences_list)
        ):
            parsed["role_relevance"] = "Not Relevant"
            parsed["role_relevance_reason"] = (
                parsed["role_relevance_reason"]
                or "Extracted location is outside allowed regions/work patterns."
            )

        return parsed

    def _classify_job_titles(self, position_name: str, job_description: str) -> list[str]:
        title_clean = normalize_text(position_name)
        exact = self.job_title_lookup.get(title_clean.lower())
        if exact:
            return [exact]

        fast_rule_titles = infer_job_titles_from_position_name(position_name, self.predefined_job_titles)
        if fast_rule_titles:
            return fast_rule_titles[:MAX_JOB_TITLES]

        prompt = build_job_titles_prompt(
            position_name=position_name,
            job_description=job_description,
            predefined_job_titles=self.predefined_job_titles,
        )
        raw = self._call_model(prompt)
        validated = validate_job_titles(raw, self.predefined_job_titles, max_items=MAX_JOB_TITLES)
        return validated if validated else []

    def _classify_seniority(self, position_name: str, job_description: str) -> list[str]:
        rule_result = detect_seniority_from_title_and_description(position_name, job_description)

        if rule_result in (
            ["leadership"], ["lead"], ["senior"], ["junior"], ["mid"], ["entry"],
            ["senior", "lead"], ["mid", "senior"]
        ):
            return rule_result

        prompt = build_seniority_prompt(position_name=position_name, job_description=job_description)
        raw = self._call_model(prompt)
        validated = validate_seniorities(
            raw,
            ["entry", "junior", "mid", "senior", "lead", "leadership"],
            max_items=3,
        )
        return validated if validated else rule_result

    def _classify_skills(self, position_name: str, job_description: str, job_category: str) -> list[str]:
        allowed_skills = self.predefined_tp_skills if job_category == "T&P job" else self.predefined_nontp_skills
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

        location_remote = self._extract_location_remote_from_description_or_url(
            position_name=position_name,
            description=job_description,
            job_url=job_url,
        )
        notes.append(location_remote.get("source_notes", ""))

        salary = extract_salary(job_description, self.predefined_salaries)
        visa_sponsorship = extract_visa_status(job_description)

        contract_type = detect_contract_type(job_description)
        if not contract_type:
            raw_contract = self._call_model(build_contract_type_prompt(job_description))
            contract_type = validate_contract_type(raw_contract)

        relevance_data = self._classify_relevance_and_category(
            position_name=position_name,
            job_description=job_description,
            location_candidates=location_remote.get("job_location_candidates", []),
            remote_preferences_list=location_remote.get("remote_preferences_list", []),
        )

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

        if result["role_relevance"] == "Not Relevant":
            result["notes"] = "; ".join([x for x in notes if x])
            return result

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
