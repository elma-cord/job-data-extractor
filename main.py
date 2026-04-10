import csv
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

from classifiers import JobClassifier
from config import INPUT_CSV, OUTPUT_CSV
from formatters import OUTPUT_COLUMNS, build_output_row
from remote_policy_lookup import RemotePolicyLookup


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
    remote_lookup = RemotePolicyLookup()
    output_rows = []

    total = len(rows)
    for idx, row in enumerate(rows, start=1):
        try:
            result = classifier.classify_job(row)
            final_row = build_output_row(row, result)

            company_domain = _get_best_company_domain(row)
            final_row["company_domain"] = company_domain

            openai_remote = _normalize_remote_value(final_row.get("remote_preferences", ""))
            gemini_remote = ""
            gemini_note = ""

            should_run_remote_lookup = (
                final_row.get("role_relevance", "") == "Relevant"
                and not openai_remote
            )

            if should_run_remote_lookup:
                remote_row = dict(row)
                if company_domain:
                    remote_row["company_domain"] = company_domain

                remote_result = remote_lookup.lookup(remote_row)
                gemini_remote = _normalize_remote_value(remote_result.remote_preferences)
                gemini_note = (remote_result.note or "").strip()

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

        output_rows.append(final_row)

        if idx % 10 == 0 or idx == total:
            print(f"Processed {idx}/{total}")

    write_output_csv(output_path, output_rows)
    print(f"Done. Output written to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Fatal error: {exc}")
        sys.exit(1)
