import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

from classifiers import JobClassifier
from config import INPUT_CSV, OUTPUT_CSV
from formatters import OUTPUT_COLUMNS, build_output_row
from remote_policy_lookup import RemotePolicyLookup
from rules import detect_quick_tp_from_title


VALID_REMOTE_VALUES = {"onsite", "hybrid", "remote"}
COMMON_NON_COMPANY_HOSTS = {
    "linkedin.com",
    "www.linkedin.com",
    "uk.linkedin.com",
    "glassdoor.com",
    "www.glassdoor.com",
    "indeed.com",
    "www.indeed.com",
    "greenhouse.io",
    "boards.greenhouse.io",
    "jobs.ashbyhq.com",
}


def read_input_csv(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_output_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _append_note(existing: str, new_note: str) -> str:
    existing = (existing or "").strip()
    new_note = (new_note or "").strip()

    if not new_note:
        return existing
    if not existing:
        return new_note
    return f"{existing}; {new_note}"


def _normalize_remote_value(value: str) -> str:
    value = (value or "").strip().lower()
    return value if value in VALID_REMOTE_VALUES else ""


def _compute_remote_overall(openai_value: str, gemini_value: str) -> str:
    openai_value = _normalize_remote_value(openai_value)
    gemini_value = _normalize_remote_value(gemini_value)

    if openai_value:
        return openai_value
    if gemini_value:
        return gemini_value
    return ""


def _normalize_domain(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"^https?://", "", value)
    value = value.split("/")[0].strip()
    value = value.strip(".")
    if value.startswith("www."):
        value = value[4:]
    return value


def _domain_from_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""

    try:
        parsed = urlparse(url if "://" in url else f"https://{url}")
        host = (parsed.netloc or "").lower().strip()
        if not host:
            return ""
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def _get_best_company_domain(row: dict) -> str:
    for key in ["company_domain", "company_website", "website", "domain"]:
        value = _normalize_domain(str(row.get(key, "")))
        if value:
            return value

    job_url_domain = _domain_from_url(str(row.get("job_url", "")))
    if job_url_domain and job_url_domain not in COMMON_NON_COMPANY_HOSTS:
        return job_url_domain

    return ""


def _is_gemini_remote_lookup_enabled() -> bool:
    raw = os.getenv("ENABLE_GEMINI_REMOTE_LOOKUP", "true").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _company_key(row: dict) -> str:
    domain = _get_best_company_domain(row)
    if domain:
        return domain
    return str(row.get("company_name", "")).strip().lower()


def _company_list_status(row: dict) -> str:
    # The "List" column marks a company as Active / Inactive / Churned.
    return str(row.get("List", "") or "").strip().lower()


def _position_title(row: dict) -> str:
    return str(row.get("job_title", row.get("position_name", "")) or "")


def main() -> None:
    input_path = Path(INPUT_CSV)
    output_path = Path(OUTPUT_CSV)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    rows = read_input_csv(input_path)
    if not rows:
        print("No rows found in input CSV.")
        write_output_csv(output_path, [])
        return

    classifier = JobClassifier()
    gemini_enabled = _is_gemini_remote_lookup_enabled()
    remote_lookup = RemotePolicyLookup() if gemini_enabled else None
    output_rows = []

    # The remote-working policy is a company-level fact, so look it up once per
    # company and reuse it for every job at that company. Saves Gemini calls and
    # guarantees the same company gets the same answer across rows.
    remote_lookup_cache: dict[str, object] = {}

    # --- Company-level pre-pass (Inactive-company T&P gate) ---
    # Group every row by company and remember each company's "List" status.
    company_to_rows: dict[str, list] = defaultdict(list)
    company_status: dict[str, str] = {}
    for i, row in enumerate(rows):
        key = _company_key(row)
        company_to_rows[key].append(i)
        status = _company_list_status(row)
        if status and key not in company_status:
            company_status[key] = status

    # STAGE 1 (cheap, no AI): an Inactive company with NO T&P-looking title on
    # ANY of its positions is dropped wholesale before we spend anything on the
    # model. Uses the broad title-based T&P detector.
    stage1_dropped: set = set()
    for key, idxs in company_to_rows.items():
        if company_status.get(key) != "inactive":
            continue
        has_tp_title = any(
            detect_quick_tp_from_title(_position_title(rows[i])) == "T&P job"
            for i in idxs
        )
        if not has_tp_title:
            stage1_dropped.add(key)

    def _no_tp_row(src: dict, code_check: str, ai_check: str, note: str) -> dict:
        blanked = build_output_row(
            src,
            classifier._blank_result(
                role_relevance="Not Relevant",
                role_relevance_reason="No T&P jobs",
                job_category="",
                notes=note,
            ),
        )
        blanked["company_domain"] = _get_best_company_domain(src)
        blanked["job_category_code_check"] = code_check
        blanked["job_category_ai_check"] = ai_check
        return blanked

    total = len(rows)
    for idx, row in enumerate(rows, start=1):
        code_check = detect_quick_tp_from_title(_position_title(row))
        key = _company_key(row)

        # Stage 1 drop: inactive company with no T&P-looking title -> skip the model.
        if key in stage1_dropped:
            output_rows.append(_no_tp_row(row, code_check, "", "rule:inactive_no_tp_code"))
            if idx % 10 == 0 or idx == total:
                print(f"Processed {idx}/{total}")
            continue

        try:
            result = classifier.classify_job(row)
            final_row = build_output_row(row, result)

            company_domain = _get_best_company_domain(row)
            final_row["company_domain"] = company_domain
            final_row["job_category_code_check"] = code_check

            openai_remote = _normalize_remote_value(final_row.get("remote_preferences", ""))
            gemini_remote = ""
            gemini_note = ""

            should_run_remote_lookup = (
                gemini_enabled
                and final_row.get("role_relevance", "") == "Relevant"
                and not openai_remote
            )

            if should_run_remote_lookup:
                remote_row = dict(row)
                if company_domain:
                    remote_row["company_domain"] = company_domain

                cache_key = key

                if cache_key and cache_key in remote_lookup_cache:
                    remote_result = remote_lookup_cache[cache_key]
                else:
                    remote_result = remote_lookup.lookup(remote_row)
                    if cache_key:
                        remote_lookup_cache[cache_key] = remote_result

                gemini_remote = _normalize_remote_value(remote_result.remote_preferences)
                gemini_note = (remote_result.note or "").strip()

            elif not gemini_enabled and final_row.get("role_relevance", "") == "Relevant" and not openai_remote:
                gemini_note = "remote_policy_lookup: skipped - disabled by ENABLE_GEMINI_REMOTE_LOOKUP"

            final_row["remote_preferences"] = openai_remote
            final_row["remote_preferences_gemini"] = gemini_remote
            final_row["remote_preferences_gemini_note"] = gemini_note
            final_row["remote_preferences_overall"] = _compute_remote_overall(
                openai_value=openai_remote,
                gemini_value=gemini_remote,
            )

            if gemini_note:
                final_row["notes"] = _append_note(
                    final_row.get("notes", ""),
                    gemini_note,
                )

        except Exception as exc:
            final_row = build_output_row(
                row,
                {
                    "role_relevance": "",
                    "role_relevance_reason": "",
                    "job_category": "",
                    "job_location": "",
                    "remote_preferences": "",
                    "remote_preferences_gemini": "",
                    "remote_preferences_overall": "",
                    "remote_preferences_gemini_note": "",
                    "remote_days": "",
                    "salary_min": "",
                    "salary_max": "",
                    "salary_currency": "",
                    "visa_sponsorship": "",
                    "contract_type": "",
                    "job_titles": [],
                    "seniorities": [],
                    "skills": [],
                    "notes": f"unhandled_error: {exc}",
                },
            )
            final_row["company_domain"] = _get_best_company_domain(row)
            final_row["job_category_code_check"] = code_check

        output_rows.append(final_row)

        if idx % 10 == 0 or idx == total:
            print(f"Processed {idx}/{total}")

    # STAGE 2 (AI accuracy catch): for Inactive companies that survived Stage 1,
    # if NONE of their positions were classified as a T&P job by the model, drop
    # the whole company too. No extra model cost - classification already ran.
    for key, idxs in company_to_rows.items():
        if company_status.get(key) != "inactive" or key in stage1_dropped:
            continue
        has_ai_tp = any(
            output_rows[i].get("job_category_ai_check", "") == "T&P job"
            for i in idxs
        )
        if has_ai_tp:
            continue
        for i in idxs:
            output_rows[i] = _no_tp_row(
                rows[i],
                output_rows[i].get("job_category_code_check", ""),
                output_rows[i].get("job_category_ai_check", ""),
                "rule:inactive_no_tp_ai",
            )

    write_output_csv(output_path, output_rows)
    print(f"Done. Output written to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Fatal error: {exc}")
        sys.exit(1)
