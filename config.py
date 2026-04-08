from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

INPUT_CSV = BASE_DIR / "jobs_input.csv"
OUTPUT_CSV = BASE_DIR / "jobs_output.csv"

PREDEFINED_JOB_TITLES_CSV = BASE_DIR / "predefined_job_titles.csv"
PREDEFINED_LOCATIONS_CSV = BASE_DIR / "predefined_locations.csv"
PREDEFINED_TP_SKILLS_CSV = BASE_DIR / "predefined_tp_skills.csv"
PREDEFINED_NONTP_SKILLS_CSV = BASE_DIR / "predefined_nontp_skills.csv"
PREDEFINED_SALARIES_CSV = BASE_DIR / "predefined_salaries.csv"

MAIN_MODEL = "gpt-4.1-mini"

OPENAI_TIMEOUT_SECONDS = 60
MAX_RETRIES = 2
MAX_OUTPUT_TOKENS = 1200

LOCATION_UNKNOWN = "Unknown"
REMOTE_NOT_SPECIFIED = "not specified"
REMOTE_DAYS_NOT_SPECIFIED = "not specified"

RELEVANT_LABEL = "Relevant"
NOT_RELEVANT_LABEL = "Not Relevant"

TP_LABEL = "T&P job"
NOT_TP_LABEL = "Not T&P"

ALLOWED_SENIORITIES = ["entry", "junior", "mid", "senior", "lead", "leadership"]
ALLOWED_REMOTE_PREFERENCES = ["onsite", "hybrid", "remote"]
ALLOWED_CONTRACT_TYPES = ["Permanent", "FTC", "Part Time", "Freelance/Contract"]

MAX_JOB_TITLES = 3
MAX_SKILLS = 10

MIN_DESCRIPTION_LENGTH_FOR_NO_FETCH = 180
FETCH_TIMEOUT_SECONDS = 25
