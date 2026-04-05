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
            "usa",
            "united states",
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

            country_group = self._detect_country_group_from_text(loc)
            is_broad = first_part_key in {
                "united kingdom", "uk", "england", "scotland", "wales",
                "northern ireland", "ireland", "europe", "emea",
                "usa", "united states"
            }

            records.append(
                {
                    "value": loc,
                    "key": key,
                    "parts": parts,
                    "part_keys": part_keys,
                    "first_part_key": first_part_key,
                    "tokens": set(key.split()),
                    "country_group": country_group,
                    "is_uk": country_group == "uk",
                    "is_usa": country_group == "usa",
                    "len": len(loc),
                    "specificity": max(len(parts), 1),
                    "is_broad": is_broad,
                    "is_city_level": len(parts) == 2 and not is_broad,
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
   "Location", "Locations", "All Locations", "Job Location", "Office Location", "Office Locations", "Based in".
3. Ignore unrelated text such as:
   benefits, DEI text, company office lists, slogans, skills, tools, generic company text, headings, random short lines.
4. If multiple locations appear, choose the most specific real job location for the role.
5. If a field says things like "United Kingdom Home", "UK Home", "Home Based", or "Remote, UK", treat that as broad UK location, not a city.
6. Normalize the result to EXACTLY one value from the acceptable locations list below.
7. If no acceptable location can be identified, return "Unknown".

Rules for remote_preferences:
1. Allowed values are only: onsite, hybrid, remote
2. Return an array.
3. Prefer explicit labels like:
   "Workplace type", "Remote status", "Working pattern", "Work type", "remote type"
4. If text says hybrid, return ["hybrid"]
5. If text says onsite, return ["onsite"]
6. If text says fully remote / remote-first / home-based, return ["remote"]
7. Do not infer remote from phrases like "remote support" unless it clearly describes the job working arrangement.
8. If not specified, return []

Rules for remote_days:
1. Return a string.
2. If not specified, return "not specified"
3. Only use remote-days logic when there is explicit work-pattern evidence such as:
   "days in office", "office days", "x days per week in the office", "hybrid working", "work from home x days"
4. If the text says 1 day in office, return "4"
5. If it says 2 days in office, return "3"
6. If it says 1-2 days in office, return "3"
7. If it says 2-3 days in office, return "2"
8. If it says fully remote, return "not specified"
9. Never use salary numbers, compensation numbers, quota numbers, team size numbers, or generic numbers.
10. Only return a number when clearly supported by work-pattern wording.

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

        value = re.sub(r"(?i)^\s*locations?\s*:\s*", "", value)
        value = re.sub(r"(?i)^\s*all locations?\s*:\s*", "", value)
        value = re.sub(r"(?i)^\s*job location\s*:\s*", "", value)
        value = re.sub(r"(?i)^\s*office location\s*:\s*", "", value)
        value = re.sub(r"(?i)^\s*office locations\s*:\s*", "", value)
        value = re.sub(r"(?i)^\s*based in\s*:\s*", "", value)
        value = re.sub(r"(?i)^\s*based in\s+", "", value)

        value = re.sub(r"(?i)\bhybrid\b", " ", value)
        value = re.sub(r"(?i)\bonsite\b", " ", value)
        value = re.sub(r"(?i)\bremote\b", " ", value)
        value = re.sub(r"(?i)\bhome[- ]based\b", " ", value)
        value = re.sub(r"(?i)\bhome based\b", " ", value)
        value = re.sub(r"(?i)\bworking pattern\b", " ", value)
        value = re.sub(r"(?i)\bworkplace type\b", " ", value)
        value = re.sub(r"(?i)\bremote status\b", " ", value)
        value = re.sub(r"(?i)\bremote type\b", " ", value)
        value = re.sub(r"(?i)\btime type\b", " ", value)

        value = re.sub(r"(?i)\buk\s*-\s*([A-Za-z][A-Za-z\s\-]+)\b", r"\1, UK", value)
        value = re.sub(r"(?i)\bunited kingdom\s*-\s*([A-Za-z][A-Za-z\s\-]+)\b", r"\1, United Kingdom", value)

        value = re.sub(r"(?i)\bunavailable\b", " ", value)
        value = re.sub(r"(?i)\{\{[^}]+\}\}", " ", value)

        m = re.match(
            r"(?i)^\s*(united kingdom|uk|england|scotland|wales|northern ireland)\s*,\s*([A-Za-z][A-Za-z\s\-]+)\s*$",
            value,
        )
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
        value = re.sub(r"(?i)\bhome[- ]based\b", " ", value)
        value = re.sub(r"(?i)\bhome based\b", " ", value)
        value = re.sub(r"(?i)\bhome\b", " ", value)
        value = re.sub(r"(?i)\bbased\b", " ", value)
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

    def _detect_country_group_from_text(self, value: str) -> str:
        value_l = (value or "").lower()
        key = self._canonical_location_key(value)
        tokens = set(key.split())

        if "uk" in tokens or "united kingdom" in value_l:
            return "uk"
        if "england" in value_l or "scotland" in value_l or "wales" in value_l or "northern ireland" in value_l:
            return "uk"

        if "usa" in tokens or "united states" in value_l:
            return "usa"
        if re.search(r"(?i),\s*(AL|AK|AZ|AR|CA|CO|CT|DC|DE|FL|GA|HI|IA|ID|IL|IN|KS|KY|LA|MA|MD|ME|MI|MN|MO|MS|MT|NC|ND|NE|NH|NJ|NM|NV|NY|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VA|VT|WA|WI|WV)\s*(,|$)", value):
            return "usa"

        return ""

    def _country_value(self, group: str) -> str:
        if group == "uk":
            for candidate in ["UK", "United Kingdom"]:
                if candidate.lower() in self.location_lookup:
                    return self.location_lookup[candidate.lower()]
            return "UK"

        if group == "usa":
            for candidate in ["USA", "United States"]:
                if candidate.lower() in self.location_lookup:
                    return self.location_lookup[candidate.lower()]
            return "USA"

        return ""

    @staticmethod
    def _contains_home_style_marker(value: str) -> bool:
        value_l = (value or "").lower()
        return bool(
            re.search(
                r"\b(home|home based|home-based|remote|remote-first|work from home|wfh)\b",
                value_l,
                flags=re.IGNORECASE,
            )
        )

    def _extract_broad_location_only(self, value: str) -> str:
        raw = (value or "").strip()
        if not raw:
            return ""

        raw_l = raw.lower()
        country_group = self._detect_country_group_from_text(raw)
        if not country_group:
            return ""

        cleaned = self._clean_location_candidate_text(raw)
        cleaned_key = self._canonical_location_key(cleaned)
        tokens = set(cleaned_key.split())

        # Broad country / region style strings should stay broad, never be forced into a city.
        broad_markers = {
            "uk", "united", "kingdom", "usa", "states",
            "england", "scotland", "wales", "ireland",
            "home", "based", "remote"
        }
        non_broad_tokens = {t for t in tokens if t not in broad_markers}

        if self._contains_home_style_marker(raw):
            return self._country_value(country_group)

        if not non_broad_tokens:
            return self._country_value(country_group)

        # Examples like "United Kingdom Home" or "UK Remote"
        if country_group == "uk" and re.search(r"\b(united kingdom|uk)\b", raw_l):
            if len(non_broad_tokens) <= 1:
                return self._country_value("uk")

        if country_group == "usa" and re.search(r"\b(united states|usa)\b", raw_l):
            if len(non_broad_tokens) <= 1:
                return self._country_value("usa")

        return ""

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
            add(", ".join(parts))
            add(", ".join(parts[:2]))
            add(", ".join(parts[-2:]))
            add(parts[0])

            if self._canonical_location_key(parts[0]) in self.broad_location_keys and len(parts) >= 2:
                add(", ".join(parts[1:]))

            if self._canonical_location_key(parts[-1]) in self.broad_location_keys:
                add(", ".join(parts[:-1]))
                if len(parts) >= 2:
                    add(parts[-2])

        tokens = value.split()
        if len(tokens) >= 2:
            add(" ".join(tokens))

        return variants

    def _effective_country_group(self, candidate_country_group: str) -> str:
        return candidate_country_group or "uk"

    def _score_location_record(
        self,
        cand_key: str,
        cand_tokens: set[str],
        record: dict[str, Any],
        candidate_country_group: str,
    ) -> int:
        effective_country_group = self._effective_country_group(candidate_country_group)

        if record["country_group"] and record["country_group"] != effective_country_group:
            return -10**6

        overlap = len(cand_tokens & record["tokens"])
        strong_match = (
            cand_key == record["key"]
            or cand_key == record["first_part_key"]
            or cand_key in record["part_keys"]
            or (record["first_part_key"] and (cand_key in record["first_part_key"] or record["first_part_key"] in cand_key))
        )

        # Prevent weak fuzzy matches from turning broad UK/USA strings into random cities.
        if not strong_match and overlap == 0:
            return -10**6

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

        if overlap:
            score += overlap * 40

        if cand_tokens and cand_tokens.issubset(record["tokens"]):
            score += 180

        if not record["is_broad"]:
            score += 80 + (record["specificity"] * 10)
        else:
            score -= 120

        if effective_country_group == "uk" and record["country_group"] == "uk":
            score += 120

        if effective_country_group == "usa" and record["country_group"] == "usa":
            score += 120

        candidate_is_city_like = len(cand_tokens) >= 1
        if candidate_is_city_like:
            if record["is_city_level"]:
                score += 160
            elif record["specificity"] > 2:
                score -= 90

        if len(cand_tokens) >= 2 and record["is_broad"]:
            score -= 180

        return score

    def _normalize_location_candidate(self, value: str) -> str:
        broad_only = self._extract_broad_location_only(value)
        if broad_only:
            return broad_only

        variants = self._candidate_variants(value)
        if not variants:
            return ""

        best_value = ""
        best_score = -10**9

        for variant in variants:
            broad_variant = self._extract_broad_location_only(variant)
            if broad_variant:
                return broad_variant

            direct = self.location_lookup.get(variant.lower())
            if direct:
                return direct

            cand_key = self._canonical_location_key(variant)
            if not cand_key:
                continue

            cand_tokens = set(cand_key.split())
            if not cand_tokens:
                continue

            candidate_country_group = self._detect_country_group_from_text(variant)

            for record in self.location_records:
                score = self._score_location_record(cand_key, cand_tokens, record, candidate_country_group)

                if score > best_score or (
                    score == best_score
                    and score > -10**8
                    and record["len"] < len(best_value or "z" * 999)
                ):
                    best_score = score
                    best_value = record["value"]

        if best_score >= 220:
            return best_value

        # Safe fallback for weak broad-country strings.
        fallback_country_group = self._detect_country_group_from_text(value)
        if fallback_country_group:
            return self._country_value(fallback_country_group)

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

    @staticmethod
    def _extract_explicit_location_snippets(text: str) -> list[str]:
        text = clean_description(text)
        if not text:
            return []

        snippets: list[str] = []

        patterns = [
            r"(?im)^\s*office locations?\s*:\s*(.+)$",
            r"(?im)^\s*office location\s*:\s*(.+)$",
            r"(?im)^\s*all locations?\s*:\s*(.+)$",
            r"(?im)^\s*job locations?\s*:\s*(.+)$",
            r"(?im)^\s*job location\s*:\s*(.+)$",
            r"(?im)^\s*locations?\s*:\s*(.+)$",
            r"(?im)^\s*based in\s*:\s*(.+)$",
            r"(?im)^\s*based in\s+(.+)$",
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                value = (match.group(1) or "").strip()
                if not value:
                    continue
                value = re.split(
                    r"(?i)\b(workplace type|work type|working pattern|remote status|remote type|salary|job type|employment type|team|department|time type|posted on)\b",
                    value,
                    maxsplit=1,
                )[0].strip(" |,-:")
                if value:
                    snippets.append(value)

        return snippets

    def _extract_explicit_location_candidates(self, text: str) -> list[str]:
        snippets = self._extract_explicit_location_snippets(text)
        candidates: list[str] = []

        for snippet in snippets:
            broad_only = self._extract_broad_location_only(snippet)
            if broad_only:
                candidates.append(broad_only)
                continue

            raw_parts = re.split(r"\s*\|\s*|\s*;\s*|\s+or\s+|\n", snippet)
            for raw_part in raw_parts:
                part = raw_part.strip()
                if not part:
                    continue

                broad_part = self._extract_broad_location_only(part)
                if broad_part:
                    candidates.append(broad_part)
                    continue

                comma_parts = [x.strip() for x in re.split(r"\s*,\s*", part) if x.strip()]
                if len(comma_parts) >= 3:
                    candidates.append(", ".join(comma_parts[:2]))
                candidates.append(part)

        normalized = self._normalize_location_candidates(candidates)

        out: list[str] = []
        for x in normalized:
            if x and x not in out:
                out.append(x)
        return out

    def _choose_best_explicit_location(self, candidates: list[str]) -> str:
        if not candidates:
            return ""

        def score(loc: str) -> tuple[int, int]:
            loc_key = self._canonical_location_key(loc)
            is_broad = loc_key in self.broad_location_keys
            is_uk = self._detect_country_group_from_text(loc) == "uk"
            is_usa = self._detect_country_group_from_text(loc) == "usa"

            rank = 0
            if not is_broad:
                rank += 100
            if is_uk:
                rank += 20
            if is_usa:
                rank += 10

            return (rank, len(loc))

        return sorted(candidates, key=score, reverse=True)[0]

    @staticmethod
    def _has_remote_days_evidence(text: str) -> bool:
        text = clean_description(text).lower()
        if not text:
            return False

        evidence_patterns = [
            r"\b\d+\s*-\s*\d+\s*days?\s+(?:a|per)?\s*week\s+in\s+the\s+office\b",
            r"\b\d+\s*-\s*\d+\s*days?\s+in\s+the\s+office\b",
            r"\b\d+\s*days?\s+(?:a|per)?\s*week\s+in\s+the\s+office\b",
            r"\b\d+\s*days?\s+in\s+the\s+office\b",
            r"\bin\s+office\s+\d+\s*-\s*\d+\s*days?\b",
            r"\bin\s+office\s+\d+\s*days?\b",
            r"\boffice\s+attendance\s+of\s+\d+\s*-\s*\d+\s*days?\b",
            r"\boffice\s+attendance\s+of\s+\d+\s*days?\b",
            r"\bhybrid\b.{0,60}\b\d+\s*-\s*\d+\s*days?\b",
            r"\bhybrid\b.{0,60}\b\d+\s*days?\b",
            r"\bwork\s+from\s+home\b.{0,60}\b\d+\s*days?\b",
            r"\b\d+\s*-\s*\d+\s*days?\s+from\s+home\b",
            r"\b\d+\s*days?\s+from\s+home\b",
        ]

        return any(re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) for pattern in evidence_patterns)

    def _safe_remote_days_from_text(self, text: str) -> str:
        text = clean_description(text)
        if not text:
            return "not specified"

        text_l = text.lower()

        if re.search(r"\bfully remote\b|\bremote-first\b|\b100%\s*remote\b", text_l):
            return "not specified"

        if not self._has_remote_days_evidence(text):
            return "not specified"

        value = extract_remote_days(text)
        value = str(value or "").strip().lower()

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

        ai_location = self._normalize_model_location(data.get("job_location", ""))
        ai_remote_preferences = self._normalize_model_remote_preferences(data.get("remote_preferences", []))
        ai_remote_days = self._normalize_model_remote_days(data.get("remote_days", "not specified"))

        if ai_remote_days != "not specified" and not self._has_remote_days_evidence(text):
            ai_remote_days = "not specified"

        return {
            "job_location": ai_location,
            "remote_preferences_list": ai_remote_preferences,
            "remote_days": ai_remote_days,
        }

    def _extract_location_remote_from_description_or_url(
        self,
        position_name: str,
        description: str,
        job_url: str,
    ) -> dict[str, Any]:
        desc = clean_description(description)

        desc_explicit_location_candidates = self._extract_explicit_location_candidates(desc)
        desc_explicit_location = self._choose_best_explicit_location(desc_explicit_location_candidates)

        desc_ai = self._extract_location_remote_with_ai(
            position_name=position_name,
            text=desc,
            source_name="job description",
        )

        desc_rule_location_candidates_raw = extract_location_candidates(desc, self.predefined_locations)
        desc_rule_location_candidates = self._normalize_location_candidates(desc_rule_location_candidates_raw)
        desc_rule_job_location = select_best_location(desc_rule_location_candidates)
        desc_rule_remote_preferences = extract_remote_preferences(desc)
        desc_rule_remote_days = self._safe_remote_days_from_text(desc)

        desc_job_location = desc_explicit_location or desc_ai["job_location"] or desc_rule_job_location
        desc_location_candidates = (
            [desc_explicit_location]
            if desc_explicit_location
            else ([desc_job_location] if desc_job_location else desc_rule_location_candidates[:])
        )

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

                page_explicit_location_candidates = self._extract_explicit_location_candidates(page_text)
                page_explicit_location = self._choose_best_explicit_location(page_explicit_location_candidates)

                page_ai = self._extract_location_remote_with_ai(
                    position_name=position_name,
                    text=page_text,
                    source_name="job page text",
                )

                page_rule_location_candidates_raw = extract_location_candidates(page_text, self.predefined_locations)
                page_rule_location_candidates = self._normalize_location_candidates(page_rule_location_candidates_raw)
                page_rule_job_location = select_best_location(page_rule_location_candidates)
                page_rule_remote_preferences = extract_remote_preferences(page_text)
                page_rule_remote_days = self._safe_remote_days_from_text(page_text)

                page_job_location = page_explicit_location or page_ai["job_location"] or page_rule_job_location
                page_location_candidates = (
                    [page_explicit_location]
                    if page_explicit_location
                    else ([page_job_location] if page_job_location else page_rule_location_candidates[:])
                )

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

                if (not job_location or job_location in {"UK", "United Kingdom", "USA", "United States"}) and page_job_location:
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
