import csv
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from playwright.sync_api import sync_playwright

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/145.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-GB,en;q=0.9,en-US;q=0.8",
}

INPUT_URLS_FILE = "job_urls.txt"
OUTPUT_CSV_FILE = "results.csv"

LOCATIONS_CSV = "predefined_locations.csv"
SALARIES_CSV = "predefined_salaries.csv"

REQUEST_TIMEOUT = 25
PLAYWRIGHT_TIMEOUT_MS = 30000

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4.1-mini"

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


@dataclass
class JobResult:
    job_url: str = ""
    job_title: str = ""
    role_relevance: str = ""
    role_relevance_reason: str = ""
    job_category: str = ""
    job_location: str = ""
    remote_preferences: str = ""
    remote_days: str = ""
    salary_min: str = ""
    salary_max: str = ""
    salary_currency: str = ""
    visa_sponsorship: str = ""
    job_type: str = ""
    job_description: str = ""
    source_method: str = ""
    status: str = ""
    notes: str = ""


def load_urls(filepath: str) -> List[str]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing input file: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_location_list(filepath: str) -> List[str]:
    if not os.path.exists(filepath):
        print(f"[WARN] {filepath} not found. Location normalization will be weaker.")
        return []

    df = pd.read_csv(filepath)
    col = df.columns[0]
    return (
        df[col]
        .dropna()
        .astype(str)
        .map(str.strip)
        .loc[lambda s: s != ""]
        .tolist()
    )


def load_salary_list(filepath: str) -> List[int]:
    if not os.path.exists(filepath):
        print(f"[WARN] {filepath} not found. Salary normalization will be weaker.")
        return []

    df = pd.read_csv(filepath)
    col = df.columns[0]
    vals = []
    for x in df[col].dropna().tolist():
        try:
            vals.append(int(float(str(x).replace(",", "").strip())))
        except Exception:
            pass
    return sorted(set(vals))


def nearest_salary(value: Optional[int], allowed: List[int]) -> str:
    if value is None:
        return ""
    if not allowed:
        return str(value)
    nearest = min(allowed, key=lambda x: abs(x - value))
    return str(nearest)


def clean_whitespace(text: str) -> str:
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def soup_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "noscript", "svg", "footer", "nav", "header"]):
        tag.extract()
    return clean_whitespace(soup.get_text("\n", strip=True))


def fetch_html(url: str) -> Tuple[Optional[str], str]:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        status = f"http_{resp.status_code}"
        if resp.ok and resp.text:
            return resp.text, status
        return None, status
    except Exception as e:
        return None, f"request_error: {e}"


def fetch_with_playwright(url: str) -> Tuple[Optional[str], str]:
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.route(
                "**/*",
                lambda route: route.abort()
                if route.request.resource_type in {"image", "media", "font"}
                else route.continue_()
            )
            page.goto(url, wait_until="domcontentloaded", timeout=PLAYWRIGHT_TIMEOUT_MS)
            page.wait_for_timeout(2500)
            html = page.content()
            browser.close()
            return html, "playwright"
    except Exception as e:
        return None, f"playwright_error: {e}"


def looks_like_js_shell(html: str, text: str) -> bool:
    low_text = text.lower()
    shell_signals = [
        "enable javascript",
        "javascript is required",
        "please enable javascript",
        "loading...",
    ]
    if any(sig in low_text for sig in shell_signals):
        return True

    if len(text.strip()) < 350:
        return True

    body_len = len(re.sub(r"\s+", " ", html))
    text_len = len(re.sub(r"\s+", " ", text))
    if body_len > 0 and text_len / body_len < 0.03:
        return True

    return False


def extract_jsonld_objects(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    objs = []
    scripts = soup.find_all("script", attrs={"type": "application/ld+json"})
    for s in scripts:
        raw = s.string or s.get_text(" ", strip=True)
        if not raw:
            continue
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                objs.extend([x for x in data if isinstance(x, dict)])
            elif isinstance(data, dict):
                objs.append(data)
        except Exception:
            continue
    return objs


def flatten_jobposting(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    found = []

    def walk(node: Any):
        if isinstance(node, dict):
            typ = node.get("@type")
            if typ == "JobPosting" or (isinstance(typ, list) and "JobPosting" in typ):
                found.append(node)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(obj)
    return found


def first_nonempty(*values: Any) -> str:
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return ""


def extract_structured_fields(soup: BeautifulSoup) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "title": "",
        "location_raw": "",
        "salary_min_raw": None,
        "salary_max_raw": None,
        "salary_currency_raw": "",
        "employment_type_raw": "",
        "description_raw": "",
        "company_name": "",
    }

    jsonlds = extract_jsonld_objects(soup)
    jobpostings = []
    for obj in jsonlds:
        jobpostings.extend(flatten_jobposting(obj))

    if jobpostings:
        jp = jobpostings[0]
        data["title"] = first_nonempty(jp.get("title"))
        data["description_raw"] = first_nonempty(jp.get("description"))

        hiring_org = jp.get("hiringOrganization")
        if isinstance(hiring_org, dict):
            data["company_name"] = first_nonempty(hiring_org.get("name"))

        loc = jp.get("jobLocation")
        if isinstance(loc, dict):
            addr = loc.get("address", {})
            if isinstance(addr, dict):
                data["location_raw"] = first_nonempty(
                    addr.get("addressLocality"),
                    addr.get("addressRegion"),
                    addr.get("addressCountry"),
                )
        elif isinstance(loc, list):
            parts = []
            for item in loc:
                if isinstance(item, dict):
                    addr = item.get("address", {})
                    if isinstance(addr, dict):
                        piece = first_nonempty(
                            addr.get("addressLocality"),
                            addr.get("addressRegion"),
                            addr.get("addressCountry"),
                        )
                        if piece:
                            parts.append(piece)
            data["location_raw"] = ", ".join(parts)

        base_salary = jp.get("baseSalary")
        if isinstance(base_salary, dict):
            data["salary_currency_raw"] = first_nonempty(base_salary.get("currency"))
            value = base_salary.get("value")
            if isinstance(value, dict):
                minv = value.get("minValue")
                maxv = value.get("maxValue")
                if minv is not None:
                    try:
                        data["salary_min_raw"] = int(float(minv))
                    except Exception:
                        pass
                if maxv is not None:
                    try:
                        data["salary_max_raw"] = int(float(maxv))
                    except Exception:
                        pass

        emp_type = jp.get("employmentType")
        if isinstance(emp_type, list):
            data["employment_type_raw"] = ", ".join(str(x) for x in emp_type)
        else:
            data["employment_type_raw"] = first_nonempty(emp_type)

    if not data["title"]:
        title_tag = soup.find("title")
        if title_tag:
            data["title"] = clean_whitespace(title_tag.get_text(" ", strip=True))

    return data


def strip_html(text: str) -> str:
    return clean_whitespace(BeautifulSoup(text or "", "lxml").get_text("\n", strip=True))


def clean_job_title(raw_title: str) -> str:
    if not raw_title:
        return ""

    title = clean_whitespace(raw_title)

    stop_patterns = [
        r"\s+\|\s+.*$",
        r"\s+·\s+.*$",
        r"\s+@\s+.*$",
        r"\s+\-\s+(remote|hybrid|onsite|on-site|on site|contract|full[- ]time|part[- ]time).*$",
        r"\s*,\s*(remote|hybrid|onsite|on-site|on site).*$",
        r"\s+\-\s+[A-Z][a-z]+,\s*[A-Z]{2,}.*$",
    ]
    for pat in stop_patterns:
        title = re.sub(pat, "", title, flags=re.I)

    company_words = [
        " ltd", " limited", " inc", " llc", " plc", " gmbh", " s.a.", " remote",
        " full-time", " full time", " part-time", " part time", " contract",
        " freelance", " location", " office"
    ]
    low = title.lower()
    cut = len(title)
    for w in company_words:
        idx = low.find(w)
        if idx != -1:
            cut = min(cut, idx)
    title = title[:cut].strip(" -|,·")

    return clean_whitespace(title)


def detect_remote_preferences(text: str) -> str:
    low = text.lower()
    found = []

    onsite_patterns = [
        r"\bonsite\b", r"\bon-site\b", r"\bon site\b", r"\bin office\b", r"\bin-office\b"
    ]
    hybrid_patterns = [
        r"\bhybrid\b", r"\bhybrid working\b", r"\bhybrid role\b", r"\bhybrid model\b"
    ]
    remote_patterns = [
        r"\bremote\b", r"\bfully remote\b", r"\bwork from home\b", r"\bwfh\b"
    ]

    if any(re.search(p, low) for p in onsite_patterns):
        found.append("onsite")
    if any(re.search(p, low) for p in hybrid_patterns):
        found.append("hybrid")
    if any(re.search(p, low) for p in remote_patterns):
        found.append("remote")

    return ", ".join(found)


def detect_remote_days(text: str, remote_prefs: str) -> str:
    low = text.lower()

    if "hybrid" not in remote_prefs:
        return ""

    if re.search(r"\bfully remote\b|\bremote every day\b", low):
        return ""

    remote_patterns = [
        r"(\d)\s*-\s*(\d)\s+days?\s+(?:remote|from home|wfh)",
        r"(\d)\s+days?\s+(?:remote|from home|wfh)",
    ]
    office_patterns = [
        r"(\d)\s*-\s*(\d)\s+days?\s+(?:in the office|from the office|office)",
        r"(\d)\s+days?\s+(?:in the office|from the office|office)",
    ]

    for pat in remote_patterns:
        m = re.search(pat, low)
        if m:
            if len(m.groups()) == 2:
                return m.group(2)
            return m.group(1)

    for pat in office_patterns:
        m = re.search(pat, low)
        if m:
            if len(m.groups()) == 2:
                low_office = int(m.group(1))
                high_office = int(m.group(2))
                remote_high = max(5 - low_office, 5 - high_office)
                return str(remote_high)
            office_days = int(m.group(1))
            return str(max(0, 5 - office_days))

    return ""


def parse_salary_candidates(text: str) -> List[Tuple[int, str]]:
    text = text.replace("\xa0", " ")
    candidates: List[Tuple[int, str]] = []

    patterns = [
        r"(£|\$|€)\s?(\d{1,3}(?:[,\s]\d{3})+|\d+)(?:\s?([kK]))?",
        r"\b(GBP|USD|EUR)\s?(\d{1,3}(?:[,\s]\d{3})+|\d+)(?:\s?([kK]))?",
    ]

    for pat in patterns:
        for m in re.finditer(pat, text):
            currency_raw = m.group(1)
            number_raw = m.group(2)
            k_flag = m.group(3) if len(m.groups()) >= 3 else None

            number = re.sub(r"[,\s]", "", number_raw)
            try:
                value = int(number)
                if k_flag:
                    value *= 1000
            except Exception:
                continue

            if currency_raw == "£":
                currency = "GBP"
            elif currency_raw == "$":
                currency = "USD"
            elif currency_raw == "€":
                currency = "EUR"
            else:
                currency = currency_raw.upper()

            if 5000 <= value <= 10000000:
                candidates.append((value, currency))

    return candidates


def extract_salary(text: str, structured: Dict[str, Any], allowed_salaries: List[int]) -> Tuple[str, str, str]:
    min_raw = structured.get("salary_min_raw")
    max_raw = structured.get("salary_max_raw")
    curr_raw = first_nonempty(structured.get("salary_currency_raw")).upper()

    if min_raw is not None or max_raw is not None:
        if min_raw is None and max_raw is not None:
            min_raw = max_raw
        if max_raw is None and min_raw is not None:
            max_raw = min_raw
        return (
            nearest_salary(min_raw, allowed_salaries),
            nearest_salary(max_raw, allowed_salaries),
            curr_raw,
        )

    candidates = parse_salary_candidates(text)
    if not candidates:
        return "", "", ""

    values = [v for v, _ in candidates]
    currencies = [c for _, c in candidates]

    min_val = min(values)
    max_val = max(values)
    currency = max(set(currencies), key=currencies.count)

    return (
        nearest_salary(min_val, allowed_salaries),
        nearest_salary(max_val, allowed_salaries),
        currency,
    )


def normalize_location(raw_location: str, text: str, allowed_locations: List[str]) -> str:
    if not allowed_locations:
        return clean_whitespace(raw_location)

    location_candidates = []

    if raw_location:
        location_candidates.append(clean_whitespace(raw_location))

    text_lines = [x.strip() for x in text.splitlines() if x.strip()]
    for line in text_lines[:300]:
        if re.search(r"\blocation\b|\bbased in\b|\boffice\b|\bremote\b", line, flags=re.I):
            location_candidates.append(clean_whitespace(line))

    joined = " || ".join(location_candidates)
    joined_low = joined.lower()

    for loc in allowed_locations:
        if joined_low == loc.lower():
            return loc

    for loc in allowed_locations:
        if loc.lower() in joined_low:
            return loc

    city_to_full = {}
    for loc in allowed_locations:
        city = loc.split(",")[0].strip().lower()
        if city and city not in city_to_full:
            city_to_full[city] = loc

    for city, full in city_to_full.items():
        if re.search(rf"\b{re.escape(city)}\b", joined_low):
            return full

    country_preferences = [
        "United Kingdom", "Ireland", "Germany", "France", "Netherlands",
        "Spain", "Portugal", "Poland", "Italy", "Sweden", "Denmark", "Norway",
        "Finland", "Switzerland", "Austria", "Belgium", "United States"
    ]
    for country in country_preferences:
        if country.lower() in joined_low and country in allowed_locations:
            return country

    return "Unknown"


def detect_visa_sponsorship(text: str) -> str:
    low = text.lower()

    yes_patterns = [
        r"visa sponsorship available",
        r"sponsorship available",
        r"we can sponsor",
        r"offers visa sponsorship",
        r"skilled worker visa",
        r"will sponsor visa",
    ]
    no_patterns = [
        r"no visa sponsorship",
        r"unable to sponsor",
        r"cannot sponsor",
        r"do not sponsor",
        r"without sponsorship",
        r"must have the right to work",
    ]

    if any(re.search(p, low) for p in no_patterns):
        return "no"
    if any(re.search(p, low) for p in yes_patterns):
        return "yes"
    return ""


def detect_job_type(text: str, structured_employment_type: str = "") -> str:
    low = f"{structured_employment_type} {text}".lower()

    permanent_patterns = [r"\bpermanent\b", r"\bfull[- ]time\b", r"\bstandard\b"]
    ftc_patterns = [r"\btemporary\b", r"\bfixed[- ]term\b", r"\bmaternity cover\b", r"\bftc\b"]
    part_time_patterns = [r"\bpart[- ]time\b", r"\bjob share\b", r"\bjob-share\b"]
    freelance_patterns = [r"\bfreelance\b", r"\bcontract\b", r"\bcontracting\b"]

    if any(re.search(p, low) for p in permanent_patterns):
        return "Permanent"
    if any(re.search(p, low) for p in ftc_patterns):
        return "FTC"
    if any(re.search(p, low) for p in part_time_patterns):
        return "Part Time"
    if any(re.search(p, low) for p in freelance_patterns):
        return "Freelance/Contract"
    return ""


def clean_job_description(text: str) -> str:
    if not text:
        return ""

    lines = [clean_whitespace(x) for x in text.splitlines()]
    lines = [x for x in lines if x]

    stop_headers = [
        "about the company",
        "about us",
        "benefits",
        "perks",
        "why join us",
        "what we offer",
        "company benefits",
        "our benefits",
        "equal opportunities",
        "apply now",
        "how to apply",
    ]

    cleaned = []
    for line in lines:
        low = line.lower().strip(":")
        if any(low == h or low.startswith(h + ":") for h in stop_headers):
            break
        cleaned.append(line)

    text = "\n".join(cleaned)

    boilerplate_patterns = [
        r"(?is)\babout the company\b.*$",
        r"(?is)\bbenefits\b.*$",
        r"(?is)\bperks\b.*$",
        r"(?is)\bwhat we offer\b.*$",
        r"(?is)\bapply now\b.*$",
        r"(?is)\bequal opportunity\b.*$",
    ]
    for pat in boilerplate_patterns:
        text = re.sub(pat, "", text)

    text = clean_whitespace(text)
    return text[:20000]


def classify_tp_and_relevance(title: str, description: str, location: str, remote_prefs: str) -> Tuple[str, str, str]:
    combined = f"{title}\n{description}".lower()

    tp_keywords = [
        "software engineer", "machine learning", "ml engineer", "ai engineer", "data engineer",
        "data scientist", "backend", "frontend", "full stack", "fullstack", "devops",
        "platform engineer", "security engineer", "product manager", "product designer",
        "ux designer", "ui designer", "qa engineer", "test engineer", "research engineer",
        "research scientist", "engineering manager", "site reliability", "sre",
        "technical program manager", "solutions engineer", "analytics engineer"
    ]
    non_tp_keywords = [
        "sales", "account executive", "business development", "marketing", "finance",
        "hr", "human resources", "talent acquisition", "recruiter", "operations",
        "customer support", "customer success", "legal", "office manager", "administrator",
        "executive assistant", "bookkeeper", "warehouse", "nurse", "teacher", "care assistant"
    ]

    is_tp = any(k in combined for k in tp_keywords)
    is_non_tp = any(k in combined for k in non_tp_keywords)

    category = ""
    if is_tp and not is_non_tp:
        category = "T&P"
    elif is_non_tp and not is_tp:
        category = "non-T&P"
    elif is_tp:
        category = "T&P"

    relevant = False
    reason_parts = []

    if category == "T&P":
        relevant = True
        reason_parts.append("Role looks like a tech/product position")

    if location and location != "Unknown":
        reason_parts.append(f"normalized location is {location}")

    if remote_prefs:
        reason_parts.append(f"remote preference is {remote_prefs}")

    if not relevant:
        reason_parts.append("Role does not clearly match tech/product relevance rules")

    return (
        "Relevant" if relevant else "Not relevant",
        " | ".join(reason_parts)[:500],
        category,
    )


def safe_json_loads(text: str) -> Dict[str, Any]:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    return json.loads(text)


def ai_extract_job_fields(
    job_url: str,
    raw_title: str,
    visible_text: str,
    structured_description: str,
    allowed_locations: List[str],
    allowed_salaries: List[int],
    fallback_job_title: str,
    fallback_location: str,
    fallback_remote_preferences: str,
    fallback_remote_days: str,
    fallback_salary_min: str,
    fallback_salary_max: str,
    fallback_salary_currency: str,
    fallback_visa_sponsorship: str,
    fallback_job_type: str,
    fallback_job_description: str,
) -> Dict[str, str]:
    if not client:
        return {
            "job_title": fallback_job_title,
            "role_relevance": "",
            "role_relevance_reason": "OPENAI_API_KEY missing",
            "job_category": "",
            "job_location": fallback_location,
            "remote_preferences": fallback_remote_preferences,
            "remote_days": fallback_remote_days,
            "salary_min": fallback_salary_min,
            "salary_max": fallback_salary_max,
            "salary_currency": fallback_salary_currency,
            "visa_sponsorship": fallback_visa_sponsorship,
            "job_type": fallback_job_type,
            "job_description": fallback_job_description,
        }

    location_list_text = "\n".join(allowed_locations[:3000])
    salary_list_text = ", ".join(str(x) for x in allowed_salaries[:3000])

    prompt = f"""
You are extracting structured job data from a scraped job page.

Return ONLY valid JSON with exactly these keys:
job_title
role_relevance
role_relevance_reason
job_category
job_location
remote_preferences
remote_days
salary_min
salary_max
salary_currency
visa_sponsorship
job_type
job_description

Hard rules:
1. role_relevance must be exactly "Relevant" or "Not relevant".
2. job_category must be exactly "T&P", "non-T&P", or "".
3. job_location must be exactly one value from the allowed locations list when possible.
4. If location cannot be matched confidently, use "Unknown" or "".
5. remote_preferences must use only these values: onsite, hybrid, remote
6. If multiple remote preferences apply, output them comma-separated in this exact order: onsite, hybrid, remote
7. remote_days must be a single number or ""
8. salary_min and salary_max must be the closest values from the allowed salary list
9. If only one salary is given, use the same value for both min and max
10. salary_currency must be something like GBP, USD, EUR, or ""
11. visa_sponsorship must be "yes", "no", or ""
12. job_type must be exactly one of: Permanent, FTC, Part Time, Freelance/Contract, or ""
13. clean job_title by removing company names, locations, contract types, separators, and extra text
14. job_description must contain the role description only, excluding company intro, benefits, perks, equal opportunities, apply now, and similar boilerplate
15. do not invent missing facts
16. use the page content as source of truth

Role relevance guidance:
- "Relevant" only if the role is clearly relevant to tech/product hiring logic
- "Not relevant" if it is clearly outside that scope
- role_relevance_reason should be short and specific

Allowed locations:
{location_list_text}

Allowed salaries:
{salary_list_text}

Job URL:
{job_url}

Raw title:
{raw_title}

Structured description:
{structured_description[:12000]}

Visible page text:
{visible_text[:25000]}
""".strip()

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt
        )

        data = safe_json_loads(response.output_text)

        expected_keys = [
            "job_title",
            "role_relevance",
            "role_relevance_reason",
            "job_category",
            "job_location",
            "remote_preferences",
            "remote_days",
            "salary_min",
            "salary_max",
            "salary_currency",
            "visa_sponsorship",
            "job_type",
            "job_description",
        ]

        clean_data: Dict[str, str] = {}
        for key in expected_keys:
            value = data.get(key, "")
            clean_data[key] = "" if value is None else str(value).strip()

        if clean_data["role_relevance"] not in {"Relevant", "Not relevant"}:
            clean_data["role_relevance"] = ""

        if clean_data["job_category"] not in {"T&P", "non-T&P", ""}:
            clean_data["job_category"] = ""

        if clean_data["visa_sponsorship"] not in {"yes", "no", ""}:
            clean_data["visa_sponsorship"] = ""

        if clean_data["job_type"] not in {"Permanent", "FTC", "Part Time", "Freelance/Contract", ""}:
            clean_data["job_type"] = ""

        return clean_data

    except Exception as e:
        return {
            "job_title": fallback_job_title,
            "role_relevance": "",
            "role_relevance_reason": f"AI error: {e}",
            "job_category": "",
            "job_location": fallback_location,
            "remote_preferences": fallback_remote_preferences,
            "remote_days": fallback_remote_days,
            "salary_min": fallback_salary_min,
            "salary_max": fallback_salary_max,
            "salary_currency": fallback_salary_currency,
            "visa_sponsorship": fallback_visa_sponsorship,
            "job_type": fallback_job_type,
            "job_description": fallback_job_description,
        }


def extract_best_content(html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")
    visible_text = soup_text(soup)
    structured = extract_structured_fields(soup)

    description_raw = structured.get("description_raw") or ""
    description_text = strip_html(description_raw)

    if len(description_text) < 500:
        description_text = visible_text

    return {
        "soup": soup,
        "visible_text": visible_text,
        "structured": structured,
        "description_text": description_text,
    }


def process_url(url: str, allowed_locations: List[str], allowed_salaries: List[int]) -> JobResult:
    result = JobResult(job_url=url)

    html, status = fetch_html(url)
    source_method = "html"
    notes = [status]

    if html:
        parsed = extract_best_content(html)
        if looks_like_js_shell(html, parsed["visible_text"]):
            pw_html, pw_status = fetch_with_playwright(url)
            notes.append(pw_status)
            if pw_html:
                html = pw_html
                source_method = "playwright"
                parsed = extract_best_content(html)
    else:
        pw_html, pw_status = fetch_with_playwright(url)
        notes.append(pw_status)
        if not pw_html:
            result.status = "failed"
            result.source_method = "none"
            result.notes = " | ".join(notes)
            return result
        html = pw_html
        source_method = "playwright"
        parsed = extract_best_content(html)

    visible_text = parsed["visible_text"]
    structured = parsed["structured"]
    description_text = parsed["description_text"]

    soup = BeautifulSoup(html, "lxml")
    title_tag_text = soup.title.get_text(strip=True) if soup.title else ""

    raw_title = first_nonempty(
        structured.get("title"),
        title_tag_text,
    )

    fallback_job_title = clean_job_title(raw_title)
    fallback_job_description = clean_job_description(description_text)

    location_raw = structured.get("location_raw", "")
    fallback_location = normalize_location(location_raw, visible_text, allowed_locations)

    fallback_remote_preferences = detect_remote_preferences(visible_text)
    fallback_remote_days = detect_remote_days(visible_text, fallback_remote_preferences)

    fallback_salary_min, fallback_salary_max, fallback_salary_currency = extract_salary(
        visible_text, structured, allowed_salaries
    )

    fallback_visa = detect_visa_sponsorship(visible_text)
    fallback_job_type = detect_job_type(visible_text, structured.get("employment_type_raw", ""))

    fallback_role_relevance, fallback_relevance_reason, fallback_category = classify_tp_and_relevance(
        fallback_job_title,
        fallback_job_description,
        fallback_location,
        fallback_remote_preferences
    )

    ai_data = ai_extract_job_fields(
        job_url=url,
        raw_title=raw_title,
        visible_text=visible_text,
        structured_description=description_text,
        allowed_locations=allowed_locations,
        allowed_salaries=allowed_salaries,
        fallback_job_title=fallback_job_title,
        fallback_location=fallback_location,
        fallback_remote_preferences=fallback_remote_preferences,
        fallback_remote_days=fallback_remote_days,
        fallback_salary_min=fallback_salary_min,
        fallback_salary_max=fallback_salary_max,
        fallback_salary_currency=fallback_salary_currency,
        fallback_visa_sponsorship=fallback_visa,
        fallback_job_type=fallback_job_type,
        fallback_job_description=fallback_job_description,
    )

    result.job_title = ai_data.get("job_title", "") or fallback_job_title
    result.role_relevance = ai_data.get("role_relevance", "") or fallback_role_relevance
    result.role_relevance_reason = ai_data.get("role_relevance_reason", "") or fallback_relevance_reason
    result.job_category = ai_data.get("job_category", "") or fallback_category
    result.job_location = ai_data.get("job_location", "") or fallback_location
    result.remote_preferences = ai_data.get("remote_preferences", "") or fallback_remote_preferences
    result.remote_days = ai_data.get("remote_days", "") or fallback_remote_days
    result.salary_min = ai_data.get("salary_min", "") or fallback_salary_min
    result.salary_max = ai_data.get("salary_max", "") or fallback_salary_max
    result.salary_currency = ai_data.get("salary_currency", "") or fallback_salary_currency
    result.visa_sponsorship = ai_data.get("visa_sponsorship", "") or fallback_visa
    result.job_type = ai_data.get("job_type", "") or fallback_job_type
    result.job_description = ai_data.get("job_description", "") or fallback_job_description
    result.source_method = source_method
    result.status = "ok"
    result.notes = " | ".join(notes)

    return result


def write_results(rows: List[JobResult], filepath: str) -> None:
    if not rows:
        return

    data = [asdict(r) for r in rows]
    fieldnames = list(data[0].keys())

    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def main() -> None:
    start = time.time()

    urls = load_urls(INPUT_URLS_FILE)
    allowed_locations = load_location_list(LOCATIONS_CSV)
    allowed_salaries = load_salary_list(SALARIES_CSV)

    print(f"[INFO] URLs loaded: {len(urls)}")
    print(f"[INFO] Allowed locations loaded: {len(allowed_locations)}")
    print(f"[INFO] Allowed salaries loaded: {len(allowed_salaries)}")
    print(f"[INFO] OpenAI enabled: {'yes' if client else 'no'}")

    results: List[JobResult] = []

    for idx, url in enumerate(urls, start=1):
        print(f"[{idx}/{len(urls)}] Processing: {url}")
        try:
            row = process_url(url, allowed_locations, allowed_salaries)
        except Exception as e:
            row = JobResult(
                job_url=url,
                status="failed",
                notes=f"unhandled_error: {e}"
            )
        results.append(row)

    write_results(results, OUTPUT_CSV_FILE)

    elapsed = round(time.time() - start, 2)
    print(f"[DONE] Wrote {len(results)} rows to {OUTPUT_CSV_FILE} in {elapsed}s")


if __name__ == "__main__":
    main()
