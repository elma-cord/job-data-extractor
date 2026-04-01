import csv
import sys
from pathlib import Path

from classifiers import JobClassifier
from config import INPUT_CSV, OUTPUT_CSV
from formatters import OUTPUT_COLUMNS, build_output_row


def read_input_csv(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_output_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


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
    output_rows = []

    total = len(rows)
    for idx, row in enumerate(rows, start=1):
        try:
            result = classifier.classify_job(row)
            final_row = build_output_row(row, result)
        except Exception as exc:
            final_row = build_output_row(
                row,
                {
                    "role_relevance": "",
                    "role_relevance_reason": "",
                    "job_category": "",
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
                    "notes": f"unhandled_error: {exc}",
                },
            )

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
