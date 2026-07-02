import csv
import re
import time
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
from prompts import (
    build_relevant_description_prompt,
    build_unified_job_extraction_prompt,
)
from rules import (
    closest_salary_value,
    detect_allowed_corporate_role,
    detect_quick_tp_from_title,
    detect_relevant_business_sales_role,
    detect_relevant_finance_accounting_role,
    extract_deterministic_skills,
    extract_remote_days,
    extract_remote_preferences,
    get_primary_text_window,
    has_disallowed_location_signal,
    infer_job_titles_from_position_name,
    infer_skills_from_position_context,
    infer_skills_from_titles,
    is_explicitly_foreign_location_text,
    is_leadership_job_title,
    is_location_allowed,
    load_single_column_csv,
    looks_like_non_job_content,
    normalize_location_match,
    obvious_excluded_role,
    reason_strongly_says_not_relevant,
    salary_context_exists,
    skill_is_supported,
    text_requires_non_english_language,
    text_is_predominantly_non_english,
    title_has_leadership_signal,
)
from validators import (
    clean_description,
    extract_json_object,
    normalize_relevance_label,
    repair_text,
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
        use_json_format = True
        use_temperature = True
        retry_delays = [2, 5, 10]

        for attempt in range(MAX_RETRIES + 1):
            try:
                kwargs = {
                    "model": MAIN_MODEL,
                    "input": prompt,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                }
                # temperature=0 makes the structured extraction deterministic
                # (same job -> same answer) and reduces hallucinated values.
                if use_temperature:
                    kwargs["temperature"] = 0
                # Ask the model for a JSON object so we don't have to scrape it
                # out of prose.
                if use_json_format:
                    kwargs["text"] = {"format": {"type": "json_object"}}

                response = self.client.responses.create(**kwargs)
                text = (response.output_text or "").strip()

                # Only accept a response we can actually parse as a JSON object.
                # A non-empty-but-unparseable answer counts as a failed attempt
                # and is retried instead of being silently passed downstream.
                if text and extract_json_object(text):
                    return text
            except Exception as exc:
                last_error = exc
                # If this SDK/model build rejects the optional args, drop them
                # and keep retrying rather than failing the whole row.
                message = str(exc).lower()
                if use_json_format and ("format" in message or "text" in message):
                    use_json_format = False
                elif use_temperature and "temperature" in message:
                    use_temperature = False

            # Back off before the next attempt so transient errors (e.g. 429
            # rate limits) get a chance to clear instead of failing instantly.
            # No sleep after the final attempt.
            if attempt < MAX_RETRIES:
                time.sleep(retry_delays[min(attempt, len(retry_delays) - 1)])

        raise RuntimeError(f"model_call_failed: {last_error}")

    def _call_model_raw(self, prompt: str, max_output_tokens: int = 1600) -> str:
        # Plain-text model call (no JSON mode) for the relevant-description trim.
        # This is a best-effort enrichment: if it fails we return "" so a failed
        # trim never blanks the whole row.
        last_error = None
        use_temperature = True
        retry_delays = [2, 5, 10]

        for attempt in range(MAX_RETRIES + 1):
            try:
                kwargs = {
                    "model": MAIN_MODEL,
                    "input": prompt,
                    "max_output_tokens": max_output_tokens,
                }
                if use_temperature:
                    kwargs["temperature"] = 0

                response = self.client.responses.create(**kwargs)
                text = (response.output_text or "").strip()
                if text:
                    return text
            except Exception as exc:
                last_error = exc
                if use_temperature and "temperature" in str(exc).lower():
                    use_temperature = False

            if attempt < MAX_RETRIES:
                time.sleep(retry_delays[min(attempt, len(retry_delays) - 1)])

        return ""

    def _extract_relevant_description(self, source_text: str) -> str:
        # Trim a job description down to the role-relevant content and return it
        # as clean HTML (used to populate the `relevant_description` column for
        # Relevant roles only). repair_text guarantees no apostrophe-as-digit
        # artifacts survive into the stored description.
        text = clean_description(source_text or "")
        if not text:
            return ""

        html = self._call_model_raw(build_relevant_description_prompt(text))
        html = repair_text(html or "").strip()
        # Strip accidental markdown code fences if the model added them.
        html = re.sub(r"^```(?:html)?\s*", "", html, flags=re.IGNORECASE)
        html = re.sub(r"\s*```$", "", html).strip()
        return html

    @staticmethod
    def _sanitize_notes_for_relevant(notes: str) -> str:
        # When the final decision is Relevant, drop any AI-authored note segment
        # that contradicts it (e.g. "location outside acceptable list", "title not
        # in predefined list"). Keeps our own rule:/source tags intact.
        if not notes:
            return notes

        contradictory = [
            "outside allowed",
            "outside acceptable",
            "acceptable location",
            "allowed location",
            "not relevant",
            "not matching",
            "not in the predefined",
            "not in predefined",
            "not a recognized",
            "not a recognised",
            "does not match",
            "outside target scope",
            "outside allowed scope",
        ]

        kept = []
        for segment in [s.strip() for s in notes.split(";")]:
            if not segment:
                continue
            low = segment.lower()
            if (
                low.startswith("rule:")
                or low.startswith("used ")
                or "lookup" in low
                or "url_fetch" in low
            ):
                kept.append(segment)
                continue
            if any(term in low for term in contradictory):
                continue
            kept.append(segment)

        return "; ".join(kept)

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
    def _append_note(existing: str, tag: str) -> str:
        existing = (existing or "").strip()
        tag = (tag or "").strip()
        if not tag:
            return existing
        return f"{existing}; {tag}" if existing else tag

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
        text = clean_description(get_primary_text_window(text))
        if not text:
            return []

        candidates: list[tuple[int, int, str]] = []

        patterns = [
            (280, r"(?im)^\s*location city\s*[:\-]\s*(.+)$"),
            (275, r"(?im)^\s*position location\s*[:\-]\s*(.+)$"),
            (270, r"(?im)^\s*work location\s*[:\-]\s*(.+)$"),
            (265, r"(?im)^\s*location\s*$\n^\s*(.+)$"),
            (260, r"(?im)^\s*location\s*[:\-]\s*(.+)$"),
            (255, r"(?im)^\s*####\s*location\s*$\n^\s*(.+)$"),
            (250, r"(?im)^\s*job location\s*[:\-]\s*(.+)$"),
            (245, r"(?im)^\s*office location\s*[:\-]\s*(.+)$"),
            (240, r"(?im)^\s*city\s*[:\-]\s*(.+)$"),
            (230, r"(?im)^\s*based in\s+(.+)$"),
            (225, r"(?im)^\s*location\s*$\n\s*([A-Za-z][^\n]+)$"),
            (220, r"(?im)\brole is based at (?:our )?(.+?)(?: office|\.)"),
            (210, r"(?im)\bthis position is part of .+? and will be an? (?:on-site|onsite|hybrid|remote) role\b"),
            (160, r"(?im)^\s*where you[\u2019']ll work\s*[:\-]?\s*(.+)$"),
            (80, r"(?im)\bhub based\s*\((.+?)\)"),
        ]

        text_len = max(len(text), 1)

        for base_weight, pattern in patterns:
            for match in re.finditer(pattern, text):
                value = ""

                if match.groups():
                    value = (match.group(1) or "").strip()

                if not value and "this position is part of" in match.group(0).lower():
                    continue

                if not value:
                    continue

                value = re.split(
                    r"(?i)\b(hybrid|remote|onsite|on-site|salary|schedule|travel required|shift|clearance|required|benefits|reporting to|employment type|industry|career level|category|job id|posted date)\b",
                    value,
                    maxsplit=1,
                )[0].strip(" ,:-")

                recency_bonus = int((match.start() / text_len) * 80)

                for variant in self._split_location_variants(value):
                    candidates.append((base_weight + recency_bonus, match.start(), variant))

        return candidates

    def _description_has_ambiguous_location(self, description: str) -> bool:
        d = clean_description(get_primary_text_window(description)).lower()

        explicit_markers = [
            " hub based ",
            "all our roles are hub based",
            "multiple locations",
            "bristol, glasgow or london",
            "kelso or bathgate",
        ]
        if any(marker in d for marker in explicit_markers):
            return True

        location_like_or_patterns = [
            r"\b(?:based in|location|locations?)[:\s]+[a-z][a-z\s\-']+\s+or\s+[a-z][a-z\s\-']+\b",
            r"\bhub based\s*\(.+?\bor\b.+?\)",
        ]
        return any(re.search(pattern, d, flags=re.IGNORECASE) for pattern in location_like_or_patterns)

    def _explicit_description_location(self, text: str) -> str:
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
            if self._is_broad_location(normalized):
                continue

            score = weight + 120
            if "," in normalized:
                score += 20

            if score > best_score or (score == best_score and pos > best_pos):
                best_score = score
                best_value = normalized
                best_pos = pos

        return best_value

    def _has_clear_single_location_in_text(self, text: str) -> bool:
        weighted = self._extract_weighted_location_candidates(text)
        strong_candidates = []

        for weight, _pos, raw in weighted:
            normalized = normalize_location_match(raw, self.predefined_locations)
            if not normalized:
                continue
            if self._is_broad_location(normalized):
                continue
            if weight >= 220:
                strong_candidates.append(normalized)

        strong_candidates = list(dict.fromkeys(strong_candidates))
        return len(strong_candidates) == 1

    def _deterministic_location_from_text(self, text: str) -> str:
        text = clean_description(get_primary_text_window(text))
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

    def _has_strong_disallowed_explicit_location(self, text: str) -> bool:
        weighted = self._extract_weighted_location_candidates(text)
        if not weighted:
            return False

        for weight, _pos, raw in weighted:
            if weight < 220:
                continue
            normalized = normalize_location_match(raw, self.predefined_locations)
            if normalized:
                return False
            if is_explicitly_foreign_location_text(raw):
                return True

        return False

    def _should_fetch_url(self, description: str) -> bool:
        description = clean_description(description)
        if len(description) < MIN_DESCRIPTION_LENGTH_FOR_NO_FETCH:
            return True

        if self._description_has_ambiguous_location(description):
            return True

        desc_l = description.lower()
        has_location_signal = any(
            x in desc_l
            for x in [
                "location:",
                "location city:",
                "job location",
                "based in",
                "where you\u2019ll work",
                "where you'll work",
                "remote",
                "hybrid",
                "onsite",
                "on-site",
                "office",
                "work from home",
            ]
        )
        has_role_signal = any(
            x in desc_l
            for x in [
                "responsibilities",
                "requirements",
                "experience",
                "about the role",
                "we are looking for",
                "job type",
                "salary",
                "role purpose",
                "what you will do",
                "who you are",
                "job description",
            ]
        )
        return not (has_location_signal and has_role_signal)

    def _build_source_text(self, description: str, job_url: str) -> tuple[str, str, bool, str]:
        description = clean_description(description)
        notes = []
        source_text = description
        used_fetched_page = False
        page_text = ""

        if job_url and self._should_fetch_url(description):
            fetched = fetch_job_page_text(job_url)
            if fetched.ok and fetched.text:
                page_text = clean_description(fetched.text)
                if page_text:
                    used_fetched_page = True
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

        return source_text, "; ".join(notes), used_fetched_page, page_text

    # Broad UK terms that are NOT in the predefined location list (which only has
    # "United Kingdom" + cities). Map them to "United Kingdom" so a clearly-UK
    # role is never left as Unknown just because no city was named.
    _UK_BROAD_TERMS = {
        "uk", "united kingdom", "great britain", "britain", "gb",
        "england", "scotland", "wales", "northern ireland",
    }
    _UK_REGION_RE = r"(?:England|Scotland|Wales|Northern Ireland|United Kingdom|U\.?K\.?|Great Britain)"

    def _as_uk_broad(self, value: str) -> str:
        v = (value or "").strip().lower().replace(".", "")
        v = re.sub(r"\s+", " ", v).strip()
        return "United Kingdom" if v in self._UK_BROAD_TERMS else ""

    def _collapse_london(self, value: str) -> str:
        # Any London-area location -> "London, UK" (a real predefined entry).
        # Treat London as one market. Careful NOT to collapse street/place names
        # that merely contain the word "London" (e.g. "London Rd, Reading, UK",
        # "Londonderry, UK").
        if not value:
            return value
        low = value.strip().lower()

        if low in {
            "london", "london, uk", "london, england", "london, england, uk",
            "greater london", "greater london, uk",
            "london metropolitan area, uk",
            "city of london", "city of london, uk",
        }:
            return "London, UK"

        # London as a locality component ("<suburb>, London, UK").
        if re.search(r",\s*london\b", low):
            return "London, UK"

        # London boroughs and the directional London areas.
        if low.startswith("london borough of"):
            return "London, UK"
        if re.match(r"(?:east|west|north|south|central)\s+london\b", low):
            return "London, UK"

        return value

    def _recover_location_from_text(self, *texts: str) -> str:
        # Fallback used when the model gave "Unknown" (or an unmatchable value)
        # but a real UK location IS present in the text. Reads labelled location
        # lines and inline "City, <UK region>" patterns only - no loose fragment
        # scraping (which used to pull "North" out of "North or South America").
        labelled_patterns = [
            r"(?im)^\s*(?:job\s+)?location(?:\s+city)?\s*[:\-]\s*(.+)$",
            r"(?im)^\s*work location\s*[:\-]\s*(.+)$",
            r"(?im)^\s*office location\s*[:\-]\s*(.+)$",
            r"(?im)^\s*based in\s+(.+)$",
        ]

        for text in texts:
            text = clean_description(get_primary_text_window(text or ""))
            if not text:
                continue

            for pattern in labelled_patterns:
                for match in re.finditer(pattern, text):
                    value = (match.group(1) or "").strip()
                    value = re.split(
                        r"(?i)\b(hybrid|remote|onsite|on-site|salary|per annum|schedule|category|posted|employment type)\b",
                        value,
                        maxsplit=1,
                    )[0].strip(" ,:-")
                    if not value or is_explicitly_foreign_location_text(value):
                        continue
                    normalized = normalize_location_match(value, self.predefined_locations)
                    if normalized and not self._is_broad_location(normalized):
                        return normalized
                    broad = self._as_uk_broad(value)
                    if broad:
                        return broad

            for match in re.finditer(
                rf"([A-Z][A-Za-z.\-' ]+?),\s*{self._UK_REGION_RE}\b", text
            ):
                city = (match.group(1) or "").strip()
                normalized = normalize_location_match(f"{city}, UK", self.predefined_locations)
                if normalized and not self._is_broad_location(normalized):
                    return normalized

        return ""

    def _choose_best_location(
        self,
        ai_location: Any,
        description_text: str,
        page_text: str,
        used_fetched_page: bool,
    ) -> str:
        # 1) Primary: normalize the location the MODEL stated for this role.
        ai_location_s = safe_str(ai_location)
        ai_said_foreign = bool(ai_location_s) and is_explicitly_foreign_location_text(ai_location_s)
        if ai_location_s and ai_location_s.lower() != "unknown":
            if not ai_said_foreign:
                normalized = normalize_location_match(ai_location_s, self.predefined_locations)
                if normalized:
                    return self._collapse_london(normalized)
                broad = self._as_uk_broad(ai_location_s)
                if broad:
                    return broad

        # 2) Recover a real UK location from the text when the model whiffed.
        recovered = self._recover_location_from_text(description_text, page_text)
        if recovered:
            return self._collapse_london(recovered)

        # 3) Last resort: clearly UK but no city named -> "United Kingdom".
        #    Guarded so a role the model explicitly placed abroad, or any text
        #    with a disallowed-location signal, can never be defaulted to the UK.
        combined = f"{description_text}\n{page_text}"
        if not ai_said_foreign and not has_disallowed_location_signal(combined) and re.search(
            r"\b(united kingdom|england|scotland|wales|northern ireland|uk)\b",
            combined,
            flags=re.IGNORECASE,
        ):
            return "United Kingdom"

        return LOCATION_UNKNOWN

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
    def _is_complex_non_manager_role(description: str) -> bool:
        d = clean_description(description).lower()
        complexity_markers = [
            "senior stakeholders",
            "cross-functional leadership",
            "own strategy",
            "strategic",
            "thought leadership",
            "technical escalation point",
            "architecture",
            "mentor",
            "coaching",
            "drive data-driven change",
            "executive decision making",
            "own roadmap",
            "stakeholder management across teams",
        ]
        return any(marker in d for marker in complexity_markers)

    @staticmethod
    def _reason_is_negative(reason: str) -> bool:
        return reason_strongly_says_not_relevant(reason)

    @staticmethod
    def _clean_skill_list(skills: list[str], source_text: str, max_items: int) -> list[str]:
        cleaned = []
        for skill in skills:
            if skill and skill not in cleaned and skill_is_supported(skill, source_text):
                cleaned.append(skill)
        return cleaned[:max_items]

    def _finalize_seniorities(self, position_name: str, seniorities: list[str], description: str, job_titles=None) -> list[str]:
        # Seniority is AI-led, but we make sure a role's level SPAN is reflected
        # (up to 3 values). Leadership is NOT exclusive: a senior leadership title
        # ("Senior Director", "Senior VP") spans senior + lead + leadership, and a
        # title like "Senior Lead" spans senior + lead.
        order = ["entry", "junior", "mid", "senior", "lead", "leadership"]

        levels = [s for s in seniorities if s in order]

        name_l = (position_name or "").lower()
        has_senior = bool(re.search(r"\b(senior|snr|sr|svp)\b", name_l))
        has_lead_word = bool(re.search(r"\blead\b", name_l)) and "lead generation" not in name_l
        has_principal = bool(re.search(r"\bprincipal\b", name_l)) or bool(re.search(r"\bstaff\s+engineer\b", name_l))

        if has_principal and "lead" not in levels:
            levels.append("lead")
        if has_lead_word and "lead" not in levels:
            levels.append("lead")
        if has_senior and "senior" not in levels:
            levels.append("senior")

        if is_leadership_job_title(job_titles or []):
            if "leadership" not in levels:
                levels.append("leadership")
            # A senior leadership title also spans lead level.
            if has_senior and "lead" not in levels:
                levels.append("lead")
        else:
            # Non-leadership title -> never carry a stray "leadership" the model
            # may have added (e.g. Account Coordinator, Principal Architect).
            levels = [s for s in levels if s != "leadership"]

        if not levels:
            # Last-resort default only when nothing else applies.
            return ["junior", "mid", "senior"]

        # Order canonically, de-duplicate, and keep the highest 3 so the role's
        # top level is always reflected when more than three would apply.
        levels = sorted(set(levels), key=order.index)
        if len(levels) > 3:
            levels = levels[-3:]
        return levels

    # A position name only supports a leadership/C-level title if it actually
    # carries a leadership signal. Word boundaries keep "coordinator" from
    # matching "coo", etc.
    _LEADERSHIP_NAME_RE = re.compile(
        r"\b(?:chief|head\s+of|director|vice\s+president|vp|founder|co-?founder|"
        r"managing\s+director|chief\s+of\s+staff|engineering\s+manager|c[a-z]o|cxo)\b",
        re.IGNORECASE,
    )

    def _filter_job_titles(self, position_name: str, job_titles: list[str]) -> list[str]:
        title_l = clean_description(position_name).lower()
        titles = job_titles[:]

        # Guard against the model tagging a plainly non-leadership role (e.g.
        # "Customer Assurance Coordinator") with a leadership/C-level title (e.g.
        # "COO"). If the position name shows no leadership signal, drop any
        # leadership title so a real title is chosen by the fallback instead.
        if not self._LEADERSHIP_NAME_RE.search(title_l):
            titles = [t for t in titles if not is_leadership_job_title([t])]

        # A pure engineering/developer role (e.g. "Product Engineer", "Software
        # Engineer") is a software engineer and must not be mislabelled as a
        # Product Manager / Owner or Customer Service. We deliberately do NOT ban
        # "Project Manager" here - an "IT Project Manager" is a genuine Project
        # Manager, and the old blanket ban was blanking those titles entirely.
        is_engineer_role = bool(
            re.search(r"\b(engineer|developer|programmer)\b", title_l)
        ) and not re.search(r"\b(product|project|program|programme)\s+manager\b", title_l)

        if is_engineer_role:
            banned_for_engineer = {
                "Customer Service Representative",
                "Customer Support",
                "Product Manager",
                "Product Owner",
            }
            titles = [t for t in titles if t not in banned_for_engineer]

        if any(x in title_l for x in ["designer", "design lead", "brand design", "brand designer"]):
            designer_titles = ["Graphic Designer", "UI Designer", "UI/UX Designer", "UX Designer"]
            prioritized = []
            for candidate in designer_titles:
                if candidate in titles and candidate not in prioritized:
                    prioritized.append(candidate)
            if "Brand Marketing" in titles and "Brand Marketing" not in prioritized:
                prioritized.append("Brand Marketing")
            for t in titles:
                if t not in prioritized:
                    prioritized.append(t)
            titles = prioritized

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

        if not out:
            if "1st line" in title_l or "first line" in title_l:
                for candidate in ["Support Engineer", "System Administrator", "System Engineer"]:
                    if candidate in self.predefined_job_titles and candidate not in out:
                        out.append(candidate)
                out = out[:MAX_JOB_TITLES]

        if not out and "application engineer" in title_l:
            for candidate in ["System Engineer", "Support Engineer", "Solutions Engineer"]:
                if candidate in self.predefined_job_titles and candidate not in out:
                    out.append(candidate)
            out = out[:MAX_JOB_TITLES]

        if "analytics manager" in title_l:
            for candidate in ["Business Analyst", "Data/Insight Analyst"]:
                if candidate in self.predefined_job_titles and candidate not in out:
                    out.append(candidate)

        if not out and any(x in title_l for x in ["accounts payable", "financial analyst", "finance analyst", "accountant", "accounting"]):
            for candidate in ["Finance/Accounting", "Operations"]:
                if candidate in self.predefined_job_titles and candidate not in out:
                    out.append(candidate)
            out = out[:MAX_JOB_TITLES]

        if any(x in title_l for x in ["brand design", "brand designer", "design lead", "brand design lead"]):
            preferred = ["Graphic Designer", "Brand Marketing"]
            new_out = []
            for candidate in preferred:
                if candidate in self.predefined_job_titles and candidate not in new_out:
                    new_out.append(candidate)
            for existing in out:
                if existing not in new_out:
                    new_out.append(existing)
            out = new_out[:MAX_JOB_TITLES]

        out = self._filter_job_titles(position_name, out)
        return out[:MAX_JOB_TITLES]

    def _parse_ai_payload(
        self,
        payload: dict[str, Any],
        description_text: str,
        page_text: str,
        source_text: str,
        used_fetched_page: bool,
    ) -> dict[str, Any]:
        role_relevance = normalize_relevance_label(payload.get("role_relevance", ""))
        job_category = normalize_tp_label(payload.get("job_category", "")) or NOT_TP_LABEL
        role_relevance_reason = safe_str(payload.get("role_relevance_reason", ""))

        job_location = self._choose_best_location(
            ai_location=payload.get("job_location", ""),
            description_text=description_text,
            page_text=page_text,
            used_fetched_page=used_fetched_page,
        )

        ai_remote_preferences = normalize_remote_preferences(payload.get("remote_preferences", []))
        det_remote_preferences = extract_remote_preferences(source_text)
        # Union the deterministic and model-detected work patterns (canonical
        # order) so genuine combinations like ["onsite", "hybrid"] are KEPT rather
        # than collapsed to a single value - a role then surfaces for candidates
        # filtering on either pattern.
        remote_set = set(det_remote_preferences) | set(ai_remote_preferences)
        remote_preferences_list = [x for x in ["onsite", "hybrid", "remote"] if x in remote_set]

        det_remote_days = extract_remote_days(source_text)
        ai_remote_days = normalize_remote_days(payload.get("remote_days", ""))
        remote_days = det_remote_days if det_remote_days != "not specified" else ai_remote_days

        # onsite + remote together is contradictory unless the ad explicitly
        # supports both; in that case drop the fully-remote tag and keep the
        # onsite (and hybrid, if present) pattern.
        if (
            "onsite" in remote_preferences_list
            and "remote" in remote_preferences_list
            and "hybrid" not in remote_preferences_list
            and "f/t site" not in source_text.lower()
        ):
            remote_preferences_list = [x for x in remote_preferences_list if x != "remote"]

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
        clean_job_title = safe_str(payload.get("clean_job_title", ""))

        return {
            "role_relevance": role_relevance,
            "role_relevance_reason": role_relevance_reason,
            "job_category": job_category,
            "job_location": job_location,
            "clean_job_title": clean_job_title,
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

    def _has_hard_rejection(self, source_text: str, reason: str = "") -> bool:
        return (
            text_is_predominantly_non_english(source_text)
            or text_requires_non_english_language(source_text)
            or has_disallowed_location_signal(source_text)
        )

    def _apply_final_consistency(self, result: dict[str, Any], position_name: str, source_text: str) -> dict[str, Any]:
        reason = result.get("role_relevance_reason", "")

        if text_is_predominantly_non_english(source_text):
            result["role_relevance"] = NOT_RELEVANT_LABEL
            result["role_relevance_reason"] = "Job description is primarily in another language."
            result["notes"] = self._append_note(result.get("notes", ""), "rule:non_english_text")

        if text_requires_non_english_language(source_text):
            result["role_relevance"] = NOT_RELEVANT_LABEL
            result["role_relevance_reason"] = "Role requires a language other than English."
            result["notes"] = self._append_note(result.get("notes", ""), "rule:non_english_required")

        if has_disallowed_location_signal(source_text):
            result["role_relevance"] = NOT_RELEVANT_LABEL
            result["role_relevance_reason"] = "Location is outside allowed regions or work-pattern rules."
            result["notes"] = self._append_note(result.get("notes", ""), "rule:disallowed_location")

        hard_rejection = self._has_hard_rejection(source_text, result.get("role_relevance_reason", ""))
        allowed_corporate_role = detect_allowed_corporate_role(position_name, source_text)

        if not hard_rejection and allowed_corporate_role:
            result["role_relevance"] = RELEVANT_LABEL
            result["notes"] = self._append_note(result.get("notes", ""), "rule:corporate_rescue")
            if (
                not result.get("role_relevance_reason")
                or "not matching" in result.get("role_relevance_reason", "").lower()
                or "not relevant" in result.get("role_relevance_reason", "").lower()
                or "outside allowed scope" in result.get("role_relevance_reason", "").lower()
                or "outside target scope" in result.get("role_relevance_reason", "").lower()
                or "not in the predefined" in result.get("role_relevance_reason", "").lower()
                or "not in predefined" in result.get("role_relevance_reason", "").lower()
                or "not a recognized" in result.get("role_relevance_reason", "").lower()
                or "construction" in result.get("role_relevance_reason", "").lower()
                or "manufacturing" in result.get("role_relevance_reason", "").lower()
                or "hospitality" in result.get("role_relevance_reason", "").lower()
            ):
                result["role_relevance_reason"] = "Role is a genuine corporate business function within allowed scope."

        if detect_relevant_business_sales_role(position_name, source_text) and result.get("role_relevance") != NOT_RELEVANT_LABEL:
            result["role_relevance"] = RELEVANT_LABEL
            result["notes"] = self._append_note(result.get("notes", ""), "rule:sales_rescue")
            if (
                not result.get("role_relevance_reason")
                or "not matching" in result.get("role_relevance_reason", "").lower()
                or "not relevant" in result.get("role_relevance_reason", "").lower()
            ):
                result["role_relevance_reason"] = "Role is a genuine business development / sales role within target business scope."

        if detect_relevant_finance_accounting_role(position_name, source_text) and result.get("role_relevance") != NOT_RELEVANT_LABEL:
            result["role_relevance"] = RELEVANT_LABEL
            result["notes"] = self._append_note(result.get("notes", ""), "rule:finance_rescue")
            if (
                not result.get("role_relevance_reason")
                or "not matching" in result.get("role_relevance_reason", "").lower()
                or "not relevant" in result.get("role_relevance_reason", "").lower()
                or "outside allowed scope" in result.get("role_relevance_reason", "").lower()
            ):
                result["role_relevance_reason"] = "Role is a genuine finance/accounting role within allowed business scope."

        if self._reason_is_negative(reason) and not allowed_corporate_role:
            result["role_relevance"] = NOT_RELEVANT_LABEL
            result["notes"] = self._append_note(result.get("notes", ""), "rule:negative_reason_override")

        if result.get("role_relevance") == RELEVANT_LABEL:
            if not is_location_allowed(
                result.get("job_location", ""),
                result.get("remote_preferences_list", []),
                source_text,
            ):
                result["role_relevance"] = NOT_RELEVANT_LABEL
                result["job_category"] = result.get("job_category") or NOT_TP_LABEL
                result["role_relevance_reason"] = "Location is outside allowed regions or work-pattern rules."
                result["notes"] = self._append_note(result.get("notes", ""), "rule:location_gate")

        if result.get("role_relevance") == NOT_RELEVANT_LABEL:
            return self._blank_result(
                role_relevance=NOT_RELEVANT_LABEL,
                role_relevance_reason=result.get("role_relevance_reason", "") or "Role is outside allowed scope.",
                job_category=result.get("job_category", "") or NOT_TP_LABEL,
                notes=result.get("notes", ""),
            )

        result["remote_preferences"] = ", ".join(result.get("remote_preferences_list", [])) if result.get("remote_preferences_list") else ""
        if not result.get("role_relevance_reason"):
            result["role_relevance_reason"] = "Role fits allowed scope and location rules."
        # A rescue rule may have flipped this role to Relevant; drop any stale AI
        # note that still claims it was Not Relevant / outside the allowed lists.
        result["notes"] = self._sanitize_notes_for_relevant(result.get("notes", ""))
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
                notes="rule:non_job_content",
            )

        excluded, excluded_reason = obvious_excluded_role(position_name, job_description)
        if excluded:
            return self._blank_result(
                role_relevance=NOT_RELEVANT_LABEL,
                role_relevance_reason=excluded_reason,
                job_category=detect_quick_tp_from_title(position_name) or NOT_TP_LABEL,
                notes="rule:excluded_role",
            )

        source_text, source_note, used_fetched_page, page_text = self._build_source_text(job_description, job_url)

        excluded_after_fetch, excluded_after_fetch_reason = obvious_excluded_role(position_name, source_text)
        if excluded_after_fetch:
            return self._blank_result(
                role_relevance=NOT_RELEVANT_LABEL,
                role_relevance_reason=excluded_after_fetch_reason,
                job_category=detect_quick_tp_from_title(position_name) or NOT_TP_LABEL,
                notes=self._append_note(source_note, "rule:excluded_role_after_fetch"),
            )

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

        parsed = self._parse_ai_payload(
            payload=payload,
            description_text=job_description,
            page_text=page_text,
            source_text=source_text,
            used_fetched_page=used_fetched_page,
        )

        if not self._has_hard_rejection(source_text, parsed.get("role_relevance_reason", "")):
            if detect_allowed_corporate_role(position_name, source_text):
                parsed["role_relevance"] = RELEVANT_LABEL
                if (
                    not parsed.get("role_relevance_reason")
                    or "not matching" in parsed.get("role_relevance_reason", "").lower()
                    or "not relevant" in parsed.get("role_relevance_reason", "").lower()
                    or "outside allowed" in parsed.get("role_relevance_reason", "").lower()
                    or "not in the predefined" in parsed.get("role_relevance_reason", "").lower()
                    or "not a recognized" in parsed.get("role_relevance_reason", "").lower()
                ):
                    parsed["role_relevance_reason"] = "Role is a genuine corporate business function within allowed scope."

        # Finalise the job title FIRST so seniority can base its leadership
        # decision on the normalized (tagged) title, not the raw position name.
        parsed["job_titles"] = self._fallback_job_titles(
            position_name=position_name,
            source_text=source_text,
            existing_titles=parsed.get("job_titles", []),
        )

        parsed["seniorities"] = self._finalize_seniorities(
            position_name,
            parsed.get("seniorities", []),
            source_text,
            job_titles=parsed["job_titles"],
        )

        allowed_skills = self.predefined_tp_skills if parsed["job_category"] == TP_LABEL else self.predefined_nontp_skills

        # AI-led skills: trust the skills the MODEL judged most appropriate for
        # this role (validated against the allowed list). We do NOT require each
        # skill to appear literally in the text - a Relevant role should still get
        # the skills typical of that role even when the description is thin. The
        # old false-positive risk came from blind keyword matching of every allowed
        # skill (now removed); the model only picks skills that fit the specific job.
        final_skills = validate_skills(payload.get("skills", []), allowed_skills, max_items=MAX_SKILLS)

        # Safety net: a relevant role should never come back with fewer than TWO
        # skills. If the model returned 0 or 1, infer a few more from the
        # role/title context and top up (deduped).
        if len(final_skills) < 2:
            inferred = infer_skills_from_position_context(
                position_name=position_name,
                description=source_text,
                allowed_skills=allowed_skills,
                max_items=4,
            )
            for skill in inferred:
                if skill not in final_skills:
                    final_skills.append(skill)

        # Still short? Fall back to the NORMALIZED job title(s), which are reliable
        # even when the raw position name is oddly worded (e.g. "Rotary Operations
        # Controller" -> Operations skills, "Quality Engineer" -> testing skills).
        # If this still can't reach 2, we leave it - a genuinely thin posting is a
        # fair signal, and we do NOT fabricate skills that do not fit.
        if len(final_skills) < 2:
            for skill in infer_skills_from_titles(parsed["job_titles"], allowed_skills, max_items=4):
                if skill not in final_skills:
                    final_skills.append(skill)

        parsed["skills"] = final_skills[:MAX_SKILLS]

        if source_note:
            parsed["notes"] = "; ".join([x for x in [parsed.get("notes", ""), source_note] if x])

        final_result = self._apply_final_consistency(parsed, position_name, source_text)

        # Clean job title + trimmed relevant description are produced for
        # RELEVANT roles only (skipped for Not Relevant to save cost/time).
        if final_result.get("role_relevance") == RELEVANT_LABEL:
            final_result["clean_job_title"] = final_result.get("clean_job_title") or position_name
            final_result["relevant_description"] = self._extract_relevant_description(source_text)
        else:
            final_result["clean_job_title"] = ""
            final_result["relevant_description"] = ""

        return final_result
