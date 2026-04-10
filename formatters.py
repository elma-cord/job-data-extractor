from typing import Any


OUTPUT_COLUMNS = [
    "company_name",
    "company_domain",
    "position_name",
    "job_url",
    "job_description",

    "role_relevance",
    "role_relevance_reason",
    "job_category",

    "job_location",
    "remote_preferences",
    "remote_preferences_gemini",
    "remote_preferences_overall",
    "remote_preferences_gemini_note",
    "remote_days",

    "salary_min",
    "salary_max",
    "salary_currency",

    "visa_sponsorship",
    "contract_type",

    "job_title_1",
    "job_title_2",
    "job_title_3",

    "seniority_1",
    "seniority_2",
    "seniority_3",

    "skill_1",
    "skill_2",
    "skill_3",
    "skill_4",
    "skill_5",
    "skill_6",
    "skill_7",
    "skill_8",
    "skill_9",
    "skill_10",

    "notes",
]


def _get_list_item(items: list[str], index: int) -> str:
    if not items or index >= len(items):
        return ""
    return items[index]


def build_output_row(raw_row: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    row = {
        "company_name": raw_row.get("company_name", ""),
        "company_domain": raw_row.get("company_domain", ""),
        "position_name": raw_row.get("job_title", raw_row.get("position_name", "")),
        "job_url": raw_row.get("job_url", ""),
        "job_description": raw_row.get("job_description", ""),

        "role_relevance": result.get("role_relevance", ""),
        "role_relevance_reason": result.get("role_relevance_reason", ""),
        "job_category": result.get("job_category", ""),

        "job_location": result.get("job_location", ""),
        "remote_preferences": result.get("remote_preferences", ""),
        "remote_preferences_gemini": result.get("remote_preferences_gemini", ""),
        "remote_preferences_overall": result.get("remote_preferences_overall", ""),
        "remote_preferences_gemini_note": result.get("remote_preferences_gemini_note", ""),
        "remote_days": result.get("remote_days", ""),

        "salary_min": result.get("salary_min", ""),
        "salary_max": result.get("salary_max", ""),
        "salary_currency": result.get("salary_currency", ""),

        "visa_sponsorship": result.get("visa_sponsorship", ""),
        "contract_type": result.get("contract_type", ""),

        "job_title_1": _get_list_item(result.get("job_titles", []), 0),
        "job_title_2": _get_list_item(result.get("job_titles", []), 1),
        "job_title_3": _get_list_item(result.get("job_titles", []), 2),

        "seniority_1": _get_list_item(result.get("seniorities", []), 0),
        "seniority_2": _get_list_item(result.get("seniorities", []), 1),
        "seniority_3": _get_list_item(result.get("seniorities", []), 2),

        "skill_1": _get_list_item(result.get("skills", []), 0),
        "skill_2": _get_list_item(result.get("skills", []), 1),
        "skill_3": _get_list_item(result.get("skills", []), 2),
        "skill_4": _get_list_item(result.get("skills", []), 3),
        "skill_5": _get_list_item(result.get("skills", []), 4),
        "skill_6": _get_list_item(result.get("skills", []), 5),
        "skill_7": _get_list_item(result.get("skills", []), 6),
        "skill_8": _get_list_item(result.get("skills", []), 7),
        "skill_9": _get_list_item(result.get("skills", []), 8),
        "skill_10": _get_list_item(result.get("skills", []), 9),

        "notes": result.get("notes", ""),
    }

    return {col: row.get(col, "") for col in OUTPUT_COLUMNS}
