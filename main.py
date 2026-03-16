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
TP_SKILLS_CSV = "predefined_tp_skills.csv"
NONTP_SKILLS_CSV = "predefined_nontp_skills.csv"
JOB_TITLES_CSV = "predefined_job_titles.csv"

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
    salary_period: str = ""
    visa_sponsorship: str = ""
    job_type: str = ""
    job_description: str = ""

    job_title_tag_1: str = ""
    job_title_tag_2: str = ""
    job_title_tag_3: str = ""

    seniority_1: str = ""
    seniority_2: str = ""
    seniority_3: str = ""

    skill_1: str = ""
    skill_2: str = ""
    skill_3: str = ""
    skill_4: str = ""
    skill_5: str = ""
    skill_6: str = ""
    skill_7: str = ""
    skill_8: str = ""
    skill_9: str = ""
    skill_10: str = ""

    source_method: str = ""
    status: str = ""
    notes: str = ""


# -------------------------
# Generic helpers
# -------------------------

def clean_whitespace(text: str) -> str:
    text = text or ""
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def first_nonempty(*values: Any) -> str:
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return ""


def strip_html(text: str) -> str:
    return clean_whitespace(BeautifulSoup(text or "", "lxml").get_text("\n", strip=True))


def safe_json_loads(text: str) -> Dict[str, Any]:
    text = (text or "").strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    return json.loads(text)


def split_lines(text: str) -> List[str]:
    return [clean_whitespace(x) for x in (text or "").splitlines() if clean_whitespace(x)]


def dedupe_keep_order(values: List[str]) -> List[str]:
    seen = set()
    out = []
    for v in values:
        key = v.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(v)
    return out


# -------------------------
# File loaders
# -------------------------

def load_urls(filepath: str) -> List[str]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing input file: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_text_list(filepath: str) -> List[str]:
    if not os.path.exists(filepath):
        print(f"[WARN] Missing file: {filepath}")
        return []
    df = pd.read_csv(filepath)
    col = df.columns[0]
    return (
        df[col]
        .dropna()
        .astype(str)
        .map(lambda x: x.strip())
        .loc[lambda s: s != ""]
        .tolist()
    )


def load_salary_list(filepath: str) -> List[int]:
    if not os.path.exists(filepath):
        print(f"[WARN] Missing file: {filepath}")
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


# -------------------------
# Fetching
# -------------------------

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
            page.wait_for_timeout(3000)
            html = page.content()
            browser.close()
            return html, "playwright"
    except Exception as e:
        return None, f"playwright_error: {e}"


def looks_like_js_shell(html: str, text: str) -> bool:
    low_text = (text or "").lower()
    shell_signals = [
        "enable javascript",
        "javascript is required",
        "please enable javascript",
        "loading...",
    ]
    if any(sig in low_text for sig in shell_signals):
        return True

    if len(low_text.strip()) < 350:
        return True

    body_len = len(re.sub(r"\s+", " ", html or ""))
    text_len = len(re.sub(r"\s+", " ", text or ""))
    if body_len > 0 and text_len / body_len < 0.03:
        return True
    return False


# -------------------------
# HTML parsing
# -------------------------

def soup_text(soup: BeautifulSoup) -> str:
    soup_copy = BeautifulSoup(str(soup), "lxml")
    for tag in soup_copy(["script", "style", "noscript", "svg"]):
        tag.extract()
    return clean_whitespace(soup_copy.get_text("\n", strip=True))


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


# -------------------------
# Role text extraction
# -------------------------

ROLE_KEEP_HEADINGS = {
    "about the role",
    "job description",
    "role purpose",
    "purpose of the role",
    "context",
    "key responsibilities",
    "responsibilities",
    "what you’ll do",
    "what you'll do",
    "what to bring",
    "what you’ll need",
    "what you'll need",
    "required",
    "required experience",
    "essential",
    "essential skills and experience",
    "desirable",
    "desirable skills and experience",
    "qualifications",
    "skills & experience",
    "skills and experience",
    "activities",
    "to be successful",
    "who we are looking for",
    "minimum qualifications",
    "preferred qualifications",
    "nice to have",
    "key relationships",
    "success measures",
    "job requirements",
    "overview",
}

ROLE_STOP_HEADINGS = {
    "company description",
    "about the company",
    "about us",
    "benefits",
    "what we offer",
    "what do you get for all your hard work?",
    "what you'll get in return",
    "what you’ll get in return",
    "additional information",
    "why version 1?",
    "recruitment agencies",
    "privacy policy",
    "reasonable adjustments",
    "smart working",
    "equality, diversity and inclusion",
    "safeguarding",
    "contact",
    "read more",
    "follow us",
}

NOISE_LINE_PATTERNS = [
    r"^©\s*\d{4}",
    r"^follow us$",
    r"^read more$",
    r"^privacy policy$",
    r"^recruitment agencies$",
    r"^temporary roles:",
    r"^permanent and fixed term roles:",
    r"^reasonable adjustments$",
    r"^smart working$",
    r"^equality, diversity and inclusion$",
    r"^safeguarding$",
    r"^contact$",
    r"^who we are:$",
    r"^logo$",
    r"^apply$",
    r"^employees work in a hybrid mode$",
]


def line_is_noise(line: str) -> bool:
    low = line.lower().strip()
    if not low:
        return True
    for pat in NOISE_LINE_PATTERNS:
        if re.search(pat, low, flags=re.I):
            return True
    return False


def build_header_text(lines: List[str], structured: Dict[str, Any], title_tag_text: str) -> str:
    keep = []
    if title_tag_text:
        keep.append(title_tag_text)
    if structured.get("title"):
        keep.append(structured["title"])
    if structured.get("location_raw"):
        keep.append(structured["location_raw"])
    if structured.get("company_name"):
        keep.append(structured["company_name"])

    # Keep top lines likely containing header meta
    for line in lines[:40]:
        low = line.lower()
        if (
            "location" in low
            or "country" in low
            or "hybrid" in low
            or "remote" in low
            or "home based" in low
            or "onsite" in low
            or "office" in low
            or "salary" in low
            or "rate" in low
            or "contract" in low
            or "full time" in low
            or "full-time" in low
            or "part time" in low
            or "part-time" in low
        ):
            keep.append(line)

    return clean_whitespace("\n".join(dedupe_keep_order(keep)))


def extract_role_body_text(lines: List[str]) -> str:
    """
    Keep the parts most likely to be actual role content.
    Preserve key sections like Required, Desirable, Key Relationships, Qualifications, etc.
    Remove obvious footer/legal/company/ATS noise.
    """
    kept = []
    in_role_zone = False

    for i, line in enumerate(lines):
        low = line.lower().strip()

        if line_is_noise(line):
            continue

        if low in ROLE_KEEP_HEADINGS:
            in_role_zone = True
            kept.append(line)
            continue

        if low in ROLE_STOP_HEADINGS:
            # stop only if we've already collected a meaningful role zone
            if in_role_zone:
                continue
            else:
                # If company text comes first, skip it but keep searching
                continue

        # Start role zone if strong role cues appear
        if not in_role_zone:
            if (
                "key responsibilities" in low
                or "responsibilities" == low
                or "minimum qualifications" in low
                or "preferred qualifications" in low
                or "required experience" in low
                or "essential skills" in low
                or "what you’ll do" in low
                or "what you'll do" in low
                or "what to bring" in low
                or "qualifications" == low
                or "job requirements" == low
            ):
                in_role_zone = True
                kept.append(line)
                continue

        # Keep useful role content lines
        if in_role_zone:
            kept.append(line)

    # If we captured almost nothing, fallback to a filtered middle-ground
    if len(kept) < 12:
        fallback = []
        for line in lines:
            low = line.lower().strip()
            if line_is_noise(line):
                continue
            if low in ROLE_STOP_HEADINGS:
                continue
            fallback.append(line)
        kept = fallback

    return clean_whitespace("\n".join(kept))


def extract_best_content(html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")
    visible_text = soup_text(soup)
    structured = extract_structured_fields(soup)

    description_raw = structured.get("description_raw") or ""
    structured_description_text = strip_html(description_raw)

    title_tag = soup.find("title")
    title_tag_text = clean_whitespace(title_tag.get_text(" ", strip=True)) if title_tag else ""

    visible_lines = split_lines(visible_text)
    header_text = build_header_text(visible_lines, structured, title_tag_text)

    # Prefer structured job description when it looks substantial, else visible text-derived role body
    structured_lines = split_lines(structured_description_text)
    visible_role_body = extract_role_body_text(visible_lines)
    structured_role_body = extract_role_body_text(structured_lines) if len(structured_lines) >= 8 else ""

    role_body_text = structured_role_body if len(structured_role_body) > 400 else visible_role_body

    role_context_text = clean_whitespace("\n".join([header_text, role_body_text]))
    all_page_text = clean_whitespace("\n".join([
        title_tag_text,
        structured.get("title", ""),
        structured.get("company_name", ""),
        structured.get("location_raw", ""),
        structured_description_text,
        visible_text,
    ]))

    return {
        "soup": soup,
        "structured": structured,
        "title_tag_text": title_tag_text,
        "header_text": header_text,
        "role_body_text": role_body_text,
        "role_context_text": role_context_text,
        "all_page_text": all_page_text,
    }


# -------------------------
# Deterministic parsing
# -------------------------

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


def normalize_category_for_skills(job_category: str) -> str:
    low = (job_category or "").strip().lower()
    if low in {"t&p", "tp", "tech & product", "tech and product"}:
        return "T&P"
    if low in {"nont&p", "non-t&p", "non tp", "not t&p", "not tp", "nontp", "non tech", "non-tech"}:
        return "NonT&P"
    return ""


def build_location_lookup(allowed_locations: List[str]) -> Dict[str, str]:
    city_lookup = {}
    for loc in allowed_locations:
        city = loc.split(",")[0].strip().lower()
        if city and city not in city_lookup:
            city_lookup[city] = loc
    return city_lookup


def normalize_location_rule_based(text: str, allowed_locations: List[str]) -> str:
    if not allowed_locations or not text:
        return ""

    city_lookup = build_location_lookup(allowed_locations)
    lines = split_lines(text)

    # Priority 1: explicit location lines in order
    candidate_lines = []
    for line in lines[:60]:
        low = line.lower()
        if any(x in low for x in ["location", "country", "based at", "based in", "home based", "hybrid", "remote"]):
            candidate_lines.append(line)
        elif re.search(r"\b(london|manchester|birmingham|edinburgh|belfast|sheffield|aberdeen|warminster|newcastle)\b", low):
            candidate_lines.append(line)

    # Priority 2: match most specific location appearing earliest
    joined = "\n".join(candidate_lines) if candidate_lines else text
    joined_low = joined.lower()

    # Exact location string match by appearance
    exact_hits = []
    for loc in allowed_locations:
        pos = joined_low.find(loc.lower())
        if pos != -1:
            exact_hits.append((pos, len(loc), loc))

    if exact_hits:
        exact_hits.sort(key=lambda x: (x[0], -x[1]))
        return exact_hits[0][2]

    # City-only fallback by appearance order
    city_hits = []
    for city, full in city_lookup.items():
        m = re.search(rf"\b{re.escape(city)}\b", joined_low)
        if m:
            city_hits.append((m.start(), len(city), full))

    if city_hits:
        city_hits.sort(key=lambda x: (x[0], -x[1]))
        return city_hits[0][2]

    # Country fallback only if nothing more specific exists
    for broad in ["United Kingdom", "England", "Scotland", "Wales", "Northern Ireland", "Ireland"]:
        for loc in allowed_locations:
            if loc.lower() == broad.lower() and broad.lower() in joined_low:
                return loc

    return ""


def detect_remote_preferences_rule_based(text: str) -> str:
    low = (text or "").lower()
    found = []

    # Stronger precedence for explicit role text
    if re.search(r"\bhome based\b|\buk remote\b|\bfully remote\b|\bremote\b", low):
        found.append("remote")
    if re.search(r"\bhybrid\b", low):
        found.append("hybrid")
    if re.search(r"\bonsite\b|\bon-site\b|\bon site\b|\bin office\b|\bin-office\b", low):
        found.append("onsite")

    ordered = [x for x in ["onsite", "hybrid", "remote"] if x in found]
    return ", ".join(ordered)


def detect_remote_days_rule_based(text: str, remote_prefs: str) -> str:
    low = (text or "").lower()

    # Only tag days if explicit. Never infer from "hybrid" alone.
    if "hybrid" not in remote_prefs and "remote" not in remote_prefs:
        return ""

    explicit_patterns = [
        r"(\d)\s*-\s*(\d)\s+days?\s+(?:in the office|from the office|office)",
        r"(\d)\s+days?\s+(?:in the office|from the office|office)",
        r"(\d)\s*-\s*(\d)\s+days?\s+(?:from home|remote|wfh)",
        r"(\d)\s+days?\s+(?:from home|remote|wfh)",
    ]

    for pat in explicit_patterns:
        m = re.search(pat, low)
        if not m:
            continue

        if "office" in pat:
            if len(m.groups()) == 2:
                office_days_low = int(m.group(1))
                office_days_high = int(m.group(2))
                return str(max(5 - office_days_low, 5 - office_days_high))
            return str(max(0, 5 - int(m.group(1))))
        else:
            if len(m.groups()) == 2:
                return m.group(2)
            return m.group(1)

    return ""


def nearest_salary(value: Optional[int], allowed_salaries: List[int]) -> str:
    if value is None:
        return ""
    if not allowed_salaries:
        return str(value)
    nearest = min(allowed_salaries, key=lambda x: abs(x - value))
    return str(nearest)


def parse_explicit_salary(text: str, allowed_salaries: List[int]) -> Tuple[str, str, str, str]:
    """
    Return salary only when there is strong explicit evidence.
    Supports annual and daily rates.
    """
    lines = split_lines(text)
    candidate_lines = []

    for line in lines[:80]:
        low = line.lower()

        # Must have strong compensation evidence
        if (
            "salary" in low
            or "rate" in low
            or "compensation" in low
            or re.search(r"[£$€]", line)
            or re.search(r"\b(?:gbp|usd|eur|cad)\b", low)
        ):
            candidate_lines.append(line)

    # Check day rate first
    for line in candidate_lines:
        low = line.lower()
        if re.search(r"\b(p/d|per day|day rate|daily rate)\b", low):
            m = re.search(
                r"(£|\$|€)?\s?(\d{2,6})(?:[,\d]*)\s*[-–to]+\s*(£|\$|€)?\s?(\d{2,6})(?:[,\d]*)",
                line,
                flags=re.I,
            )
            if m:
                curr_sym = m.group(1) or m.group(3) or ""
                min_raw = int(re.sub(r"[^\d]", "", m.group(2)))
                max_raw = int(re.sub(r"[^\d]", "", m.group(4)))
                currency = {"£": "GBP", "$": "USD", "€": "EUR"}.get(curr_sym, "")
                return str(min_raw), str(max_raw), currency, "day"

            m_single = re.search(r"(£|\$|€)\s?(\d{2,6})", line)
            if m_single:
                curr_sym = m_single.group(1)
                val = int(m_single.group(2))
                currency = {"£": "GBP", "$": "USD", "€": "EUR"}.get(curr_sym, "")
                return str(val), str(val), currency, "day"

    # Annual or normal compensation
    for line in candidate_lines:
        low = line.lower()

        # Skip lines that look like dates/reference or standards
        if re.search(r"\b(202\d|iso|nist|reference|posted on)\b", low):
            continue

        if not (
            "salary" in low
            or "annually" in low
            or "gross annually" in low
            or "per annum" in low
            or "pa" in low
            or re.search(r"[£$€]", line)
        ):
            continue

        m_range = re.search(
            r"(£|\$|€)\s?(\d{1,3}(?:,\d{3})+|\d{4,6})\s*(?:-|–|to)\s*(£|\$|€)?\s?(\d{1,3}(?:,\d{3})+|\d{4,6})",
            line
        )
        if m_range:
            curr_sym = m_range.group(1) or m_range.group(3) or ""
            min_raw = int(re.sub(r"[^\d]", "", m_range.group(2)))
            max_raw = int(re.sub(r"[^\d]", "", m_range.group(4)))
            currency = {"£": "GBP", "$": "USD", "€": "EUR"}.get(curr_sym, "")
            return (
                nearest_salary(min_raw, allowed_salaries),
                nearest_salary(max_raw, allowed_salaries),
                currency,
                "year",
            )

        m_single = re.search(r"(£|\$|€)\s?(\d{1,3}(?:,\d{3})+|\d{4,6})", line)
        if m_single:
            curr_sym = m_single.group(1)
            val = int(re.sub(r"[^\d]", "", m_single.group(2)))
            currency = {"£": "GBP", "$": "USD", "€": "EUR"}.get(curr_sym, "")
            normalized = nearest_salary(val, allowed_salaries)
            return normalized, normalized, currency, "year"

    return "", "", "", ""


def detect_visa_rule_based(text: str) -> str:
    low = (text or "").lower()

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
        r"right to work in the uk is mandatory",
        r"must have the right to work",
    ]

    if any(re.search(p, low) for p in no_patterns):
        return "no"
    if any(re.search(p, low) for p in yes_patterns):
        return "yes"
    return ""


def detect_job_type_rule_based(text: str, structured_employment_type: str = "") -> str:
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


def fallback_seniorities(job_title: str, role_text: str) -> List[str]:
    title_low = (job_title or "").lower()
    text_low = (role_text or "").lower()

    if any(x in title_low for x in ["head of", "director", "vp ", "vice president", "chief ", "cfo", "cto", "cio", "coo", "cmo", "cro", "cpo", "cso", "engineering manager"]):
        return ["leadership"]

    if any(x in title_low for x in ["senior", "sr "]):
        return ["senior", "lead"]

    if any(x in title_low for x in ["lead ", "principal"]):
        return ["lead"]

    if any(x in title_low for x in ["junior", "jr "]):
        return ["junior", "mid"]

    if any(x in title_low for x in ["associate", "mid weight", "mid-weight"]):
        return ["mid"]

    # Experience rules
    years = []
    for m in re.finditer(r"\b(\d)\s*-\s*(\d)\s+years?\b", text_low):
        years.extend([int(m.group(1)), int(m.group(2))])
    for m in re.finditer(r"\b(\d)\+?\s+years?\b", text_low):
        years.append(int(m.group(1)))

    if years:
        min_y = min(years)
        max_y = max(years)
        out = []
        for y in range(min_y, max_y + 1):
            if y <= 1:
                out.extend(["entry", "junior"])
            elif y == 2:
                out.extend(["junior", "mid"])
            elif 3 <= y <= 5:
                out.extend(["senior", "lead"])
            elif y > 5:
                out.extend(["senior", "lead"])
        return [x for x in ["entry", "junior", "mid", "senior", "lead", "leadership"] if x in out][:3]

    return []


# -------------------------
# AI prompts
# -------------------------

def normalize_job_title_from_list(value: str, allowed_job_titles: List[str]) -> str:
    value_clean = clean_whitespace(value)
    for jt in allowed_job_titles:
        if value_clean.lower() == jt.lower():
            return jt
    return ""


def normalize_seniority_list(values: List[str]) -> List[str]:
    order = ["entry", "junior", "mid", "senior", "lead", "leadership"]
    found = []
    for v in values:
        low = str(v).strip().lower()
        if low in order and low not in found:
            found.append(low)
    return [x for x in order if x in found][:3]


def ai_check_relevance(
    job_title: str,
    role_context_text: str,
    fallback_role_relevance: str,
    fallback_reason: str,
) -> Dict[str, str]:
    if not client:
        return {
            "role_relevance": fallback_role_relevance,
            "role_relevance_reason": fallback_reason or "OPENAI_API_KEY missing",
        }

    prompt = f"""
You will receive two inputs: job title and description. Perform the following task carefully and output the results as JSON.

Return ONLY valid JSON with exactly these keys:
role_relevance
role_relevance_reason

Field rules:
- role_relevance must be exactly "Relevant" or "Not relevant"
- role_relevance_reason must be concise and specific

Role relevance rules:
- Treat close synonyms and subfunctions of accepted families as relevant.
- Relevant families include:
  Account Director, Account Executive, AI Engineer, Automation Engineer, Back End, BI Developer, Big Data Engineer, Brand Marketing, Business Analyst, Business Development Manager, Business Operations, CDO, CFO, Chief of Staff, CIO, CLO, Cloud Engineer, CMO, Computer Vision Engineer, Content Marketing, COO, Copywriting, CPO, CRM Developer, CRM Manager, CRO, CSM / Account Manager, CSO, CTO, Customer Operations, Customer Service Rep, Customer Support, Data Architect, Data Engineer, Data Scientist, Data / Insight Analyst, Database Engineer, Deep Learning Engineer, Demand / Lead Generation, Developer in Test, DevOps Engineer, Digital Marketing, Embedded Developer, Engineering Manager, Events & Community, Executive Assistant, Finance / Accounting, Founder, FP&A, Front End, Full Stack, Games Designer, Games Developer, Generalist Marketing, Graphic Designer, Graphics Developer, Growth Marketing, Head of Customer, Head of Data, Head of Design, Head of Engineering, Head of Finance, Head of HR, Head of Infrastructure, Head of Marketing, Head of Operations, Head of Product, Head of QA, Head of Sales, Human Resources, Implementation Manager, Integration Developer, Legal, Machine Learning Engineer, Marketing Analyst, Mobile Developer, Network Engineer, Operations, Partnerships, Penetration Tester, People Ops, Performance Marketing, PR / Communications, Product Manager, Product Marketing, Product Owner, Project Manager, QA Automation Tester, QA Manual Tester, Quality Assurance, Quantitative Developer, Renewals Manager, Research Engineer, RevOps, Risk & Compliance, Sales Engineer, Sales Operations, Scrum Master, SDR / BDR, Security, Security Engineer, SEO Marketing, Site Reliability Engineer, Social Media Marketing, Solutions Engineer, Support Engineer, System Administrator, System Engineer, Talent Acquisition, Technical Architect, Technical Director, Technical Writer, Testing Manager, UI Designer, UI/UX Designer, UX Designer, UX Researcher, Videography, VP of Engineering.

Important expansions:
- Finance / Accounting includes subfunctions such as accounts payable, accounts receivable, billing, payroll, controllership, bookkeeping, treasury, audit, and similar finance/accounting roles.
- Business Operations includes transformation, change, program, PMO, operational excellence, AI transformation, and similar program/change/ops roles.
- Account / client roles include account management, customer success, client services, renewals, and similar relationship-management roles.
- IT support/infrastructure roles are relevant and should not be excluded just because they are customer-facing or support-oriented.

Location rules:
- United Kingdom is allowed for onsite, hybrid, or remote.
- Ireland is allowed only if remote.
- Europe is allowed only if explicitly Remote Europe or Remote EMEA.
- Remote Global is allowed unless limited to excluded regions.
- Use the provided text only. If location is clearly UK city/UK region, that is allowed.

Language rules:
- If the description clearly requires a non-English language, mark Not relevant.
- Otherwise do not reject for language.

Reject roles clearly outside accepted business/tech functions, e.g. teacher, waiter, nurse, construction, civil engineering, retail, mechanical/manufacturing operations, beauty retail, maritime, microbiology when clearly outside the accepted families.

Input job title:
{job_title}

Input description:
{role_context_text[:22000]}
""".strip()

    try:
        response = client.responses.create(model=OPENAI_MODEL, input=prompt)
        data = safe_json_loads(response.output_text)
        role_relevance = str(data.get("role_relevance", "") or "").strip()
        reason = str(data.get("role_relevance_reason", "") or "").strip()

        if role_relevance not in {"Relevant", "Not relevant"}:
            role_relevance = fallback_role_relevance
        if not reason:
            reason = fallback_reason

        return {
            "role_relevance": role_relevance,
            "role_relevance_reason": reason,
        }
    except Exception as e:
        return {
            "role_relevance": fallback_role_relevance,
            "role_relevance_reason": f"AI error: {e}",
        }


def ai_tag_relevant_job(
    job_title: str,
    header_text: str,
    role_body_text: str,
    allowed_locations: List[str],
    allowed_salaries: List[int],
    allowed_job_titles: List[str],
    fallback_job_category: str,
    fallback_location: str,
    fallback_remote_preferences: str,
    fallback_remote_days: str,
    fallback_salary_min: str,
    fallback_salary_max: str,
    fallback_salary_currency: str,
    fallback_salary_period: str,
    fallback_visa: str,
    fallback_job_type: str,
    fallback_job_description: str,
) -> Dict[str, Any]:
    if not client:
        return {
            "job_category": fallback_job_category,
            "job_location": fallback_location,
            "remote_preferences": fallback_remote_preferences,
            "remote_days": fallback_remote_days,
            "salary_min": fallback_salary_min,
            "salary_max": fallback_salary_max,
            "salary_currency": fallback_salary_currency,
            "salary_period": fallback_salary_period,
            "visa_sponsorship": fallback_visa,
            "job_type": fallback_job_type,
            "job_description": fallback_job_description,
            "job_titles": [],
            "seniorities": [],
        }

    location_list_text = "\n".join(allowed_locations[:3000])
    salary_list_text = ", ".join(str(x) for x in allowed_salaries[:3000])
    job_titles_text = ", ".join(allowed_job_titles[:3000])

    prompt = f"""
You will receive:
1. job title
2. header/meta text
3. role description text

Return ONLY valid JSON with exactly these keys:
job_category
job_location
remote_preferences
remote_days
salary_min
salary_max
salary_currency
salary_period
visa_sponsorship
job_type
job_description
job_titles
seniorities

Rules for job_category:
- Output exactly one of: "T&P", "NonT&P"
- T&P includes software development, engineering, product, data, IT, UX/UI, QA, DevOps, infrastructure, system administration, support engineering, network engineering, solutions engineering, cloud/platform engineering and similar technical roles.
- NonT&P is everything else that is still relevant.

Rules for job_location:
- Use header/meta text first, then role description.
- Prefer the most specific visible location over a broad country fallback.
- If multiple valid locations are listed, choose the first clear normalized location from the allowed list or the closest broader valid one.
- Only return "Unknown" if there is truly no clear location evidence.

Rules for remote_preferences:
- Allowed values only: onsite, hybrid, remote
- Use role-specific/header evidence first
- Do not let generic company-wide smart-working text override a clearer role-specific line

Rules for remote_days:
- Return only a single number or ""
- Only return a number when explicit days are stated
- Never infer a number from the word "hybrid" alone

Rules for salary:
- Only tag salary when compensation/rate is explicitly stated
- salary_period must be one of: year, day, hour, month, or ""
- If a daily rate is stated (e.g. £500-£550 p/d), preserve it as min/max with salary_period=day
- Do not invent salary from unrelated numbers, dates, standards, reference ids, years, counts, targets, ISO/NIST values, etc.

Rules for visa_sponsorship:
- Return only yes, no, or ""

Rules for job_type:
- Return only one of:
  Permanent
  FTC
  Part Time
  Freelance/Contract
  or ""
- Priority: Permanent > FTC > Part Time > Freelance/Contract

Rules for job_description:
- Keep the main role content
- Preserve useful sections such as:
  responsibilities, required, essential, desirable, qualifications, what to bring, key relationships, success measures, minimum qualifications, preferred qualifications
- Remove company/benefits/footer/privacy/apply/legal/ATS boilerplate

Rules for job_titles:
- You may return more than one job title if clearly supported
- Analyze job title and role description together
- Return up to 3 exact strings from the predefined list
- Prefer the real job function over buzzwords
- Do NOT return leadership titles like Head of Engineering / leadership roles unless the role is explicitly that level
- For account-management style roles, prefer CSM/Account Manager when the function is relationship/account ownership rather than net-new sales
- For customer-facing technical roles, Solutions Engineer can coexist with another technical title when clearly supported

Rules for seniorities:
- Allowed only: entry, junior, mid, senior, lead, leadership
- Must be lowercase
- Return as JSON array in this order only: entry, junior, mid, senior, lead, leadership
- If title includes explicit org-leadership indicators like "head of", "director", "vp", "chief", return leadership when appropriate
- Do NOT use leadership only because the role has technical authority or architectural ownership
- If title says senior, include senior
- If role clearly suggests multiple levels, return up to 3

Allowed normalized locations:
{location_list_text}

Allowed salaries:
{salary_list_text}

Predefined job titles:
{job_titles_text}

Input job title:
{job_title}

Header/meta text:
{header_text[:6000]}

Role description:
{role_body_text[:22000]}
""".strip()

    try:
        response = client.responses.create(model=OPENAI_MODEL, input=prompt)
        data = safe_json_loads(response.output_text)

        job_category = normalize_category_for_skills(str(data.get("job_category", "") or "").strip()) or fallback_job_category
        job_location = str(data.get("job_location", "") or "").strip() or fallback_location
        remote_preferences = str(data.get("remote_preferences", "") or "").strip() or fallback_remote_preferences
        remote_days = str(data.get("remote_days", "") or "").strip() or fallback_remote_days
        salary_min = str(data.get("salary_min", "") or "").strip() or fallback_salary_min
        salary_max = str(data.get("salary_max", "") or "").strip() or fallback_salary_max
        salary_currency = str(data.get("salary_currency", "") or "").strip() or fallback_salary_currency
        salary_period = str(data.get("salary_period", "") or "").strip() or fallback_salary_period
        visa_sponsorship = str(data.get("visa_sponsorship", "") or "").strip()
        job_type = str(data.get("job_type", "") or "").strip()
        job_description = str(data.get("job_description", "") or "").strip() or fallback_job_description

        if visa_sponsorship not in {"yes", "no", ""}:
            visa_sponsorship = fallback_visa
        if job_type not in {"Permanent", "FTC", "Part Time", "Freelance/Contract", ""}:
            job_type = fallback_job_type
        if salary_period not in {"year", "day", "hour", "month", ""}:
            salary_period = fallback_salary_period

        if remote_preferences:
            parts = [p.strip() for p in remote_preferences.split(",") if p.strip()]
            valid_order = ["onsite", "hybrid", "remote"]
            parts = [p for p in valid_order if p in parts]
            remote_preferences = ", ".join(parts)

        raw_titles = data.get("job_titles", [])
        if not isinstance(raw_titles, list):
            raw_titles = []
        normalized_titles = []
        for t in raw_titles:
            exact = normalize_job_title_from_list(str(t), allowed_job_titles)
            if exact and exact not in normalized_titles:
                normalized_titles.append(exact)
        normalized_titles = normalized_titles[:3]

        raw_seniorities = data.get("seniorities", [])
        if not isinstance(raw_seniorities, list):
            raw_seniorities = []
        normalized_seniorities = normalize_seniority_list(raw_seniorities)

        return {
            "job_category": job_category,
            "job_location": job_location,
            "remote_preferences": remote_preferences,
            "remote_days": remote_days,
            "salary_min": salary_min,
            "salary_max": salary_max,
            "salary_currency": salary_currency,
            "salary_period": salary_period,
            "visa_sponsorship": visa_sponsorship,
            "job_type": job_type,
            "job_description": job_description,
            "job_titles": normalized_titles,
            "seniorities": normalized_seniorities,
        }
    except Exception as e:
        return {
            "job_category": fallback_job_category,
            "job_location": fallback_location,
            "remote_preferences": fallback_remote_preferences,
            "remote_days": fallback_remote_days,
            "salary_min": fallback_salary_min,
            "salary_max": fallback_salary_max,
            "salary_currency": fallback_salary_currency,
            "salary_period": fallback_salary_period,
            "visa_sponsorship": fallback_visa,
            "job_type": fallback_job_type,
            "job_description": fallback_job_description,
            "job_titles": [],
            "seniorities": [],
            "_ai_error": str(e),
        }


# -------------------------
# Skills tagging
# -------------------------

def build_skill_regex(skill: str) -> re.Pattern:
    normalized = skill.strip().lower()

    if normalized == "c++":
        return re.compile(r"(?<![A-Za-z0-9])c\+\+(?![A-Za-z0-9])", flags=re.I)
    if normalized == "c#":
        return re.compile(r"(?<![A-Za-z0-9])c#(?![A-Za-z0-9])", flags=re.I)
    if normalized == ".net":
        return re.compile(r"(?<![A-Za-z0-9])(?:\.net|dotnet|dot net)(?![A-Za-z0-9])", flags=re.I)
    if normalized == "node.js":
        return re.compile(r"(?<![A-Za-z0-9])(?:node\.js|nodejs|node js)(?![A-Za-z0-9])", flags=re.I)
    if normalized == "next.js":
        return re.compile(r"(?<![A-Za-z0-9])(?:next\.js|nextjs|next js)(?![A-Za-z0-9])", flags=re.I)
    if normalized == "vue.js":
        return re.compile(r"(?<![A-Za-z0-9])(?:vue\.js|vuejs|vue js)(?![A-Za-z0-9])", flags=re.I)
    if normalized == "nuxt.js":
        return re.compile(r"(?<![A-Za-z0-9])(?:nuxt\.js|nuxtjs|nuxt js)(?![A-Za-z0-9])", flags=re.I)
    if normalized == "git":
        return re.compile(r"(?<![A-Za-z0-9])git(?![A-Za-z0-9])", flags=re.I)

    escaped = re.escape(skill)
    return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", flags=re.I)


def build_skills_source_text(header_text: str, role_body_text: str) -> str:
    """
    Skills must come from role-focused text only.
    Exclude company description, footer, ATS, benefits, privacy, etc.
    """
    lines = split_lines("\n".join([header_text, role_body_text]))
    out = []

    disallow = [
        "smartrecruiters",
        "workday",
        "privacy policy",
        "read more",
        "follow us",
        "recruitment agencies",
        "why version 1?",
        "about the company",
        "company description",
        "benefits",
        "what we offer",
        "additional information",
        "equality, diversity and inclusion",
        "reasonable adjustments",
        "contact",
        "©",
    ]

    for line in lines:
        low = line.lower()
        if any(x in low for x in disallow):
            continue
        out.append(line)

    return clean_whitespace("\n".join(out))


def exact_match_skills_in_order(description: str, skill_list: List[str], limit: int = 10) -> List[str]:
    if not description or not skill_list:
        return []

    found = []
    for skill in skill_list:
        regex = build_skill_regex(skill)
        match = regex.search(description)
        if match:
            index = match.start()
            if not any(e["skill"].lower() == skill.lower() for e in found):
                found.append({"skill": skill, "index": index})

    found.sort(key=lambda x: x["index"])
    return [x["skill"] for x in found[:limit]]


def ai_enrich_skills(
    role_category: str,
    description: str,
    exact_skills: List[str],
    allowed_skills: List[str],
) -> List[str]:
    if not client or not role_category or not description or not allowed_skills:
        return exact_skills[:10]

    role_category = normalize_category_for_skills(role_category)
    if role_category not in {"T&P", "NonT&P"}:
        return exact_skills[:10]

    allowed_skills_text = ", ".join(allowed_skills[:5000])
    exact_skills_text = ", ".join(exact_skills)

    prompt = f"""
You are a strict skills tagger for a recruiting workflow.

Return valid json only. No markdown. No commentary.

Goal:
- Use ONLY the supplied role-focused description text.
- Keep the exact-match skills already found.
- Add only clearly evidenced missing skills from the description.
- Never use footer/platform/company-marketing text because it is not included here.
- Do not infer broad adjacent concepts unless explicitly supported by the role description.

Hard rules:
- role_category is PROVIDED as input. You MUST echo it exactly as given ("T&P" or "NonT&P"). Do not change it.
- skills MUST be chosen ONLY from the correct Allowed skills list (exact string match).
- exact_skills is the PRIMARY base set. Preserve them.
- NEVER output a skill not present in the allowed list.
- Before finalizing, verify every returned skill appears exactly in the allowed list.
- Keep exact_skills first, then add missing clearly evidenced skills.
- Return up to 10 skills total.

Output schema:
{{
  "role_category": "T&P" or "NonT&P",
  "skills": ["..."]
}}

role_category:
{role_category}

exact_skills:
{exact_skills_text}

Allowed skills:
{allowed_skills_text}

role_focused_description:
{description[:18000]}
""".strip()

    try:
        response = client.responses.create(model=OPENAI_MODEL, input=prompt)
        data = safe_json_loads(response.output_text)

        out_category = normalize_category_for_skills(str(data.get("role_category", "") or "").strip())
        if out_category != role_category:
            return exact_skills[:10]

        raw_skills = data.get("skills", [])
        if not isinstance(raw_skills, list):
            return exact_skills[:10]

        merged = []
        for sk in exact_skills:
            exact = next((s for s in allowed_skills if s.lower() == sk.lower()), "")
            if exact and exact not in merged:
                merged.append(exact)

        for sk in raw_skills:
            sk_clean = str(sk).strip()
            exact = next((s for s in allowed_skills if s.lower() == sk_clean.lower()), "")
            if exact and exact not in merged:
                merged.append(exact)

        # Final hard filter
        filtered = []
        allowed_lower = {s.lower(): s for s in allowed_skills}
        for sk in merged:
            exact = allowed_lower.get(sk.lower())
            if exact and exact not in filtered:
                filtered.append(exact)

        return filtered[:10]
    except Exception:
        return exact_skills[:10]


# -------------------------
# Main per-URL logic
# -------------------------

def process_url(
    url: str,
    allowed_locations: List[str],
    allowed_salaries: List[int],
    tp_skills: List[str],
    nontp_skills: List[str],
    allowed_job_titles: List[str],
) -> JobResult:
    result = JobResult(job_url=url)

    html, status = fetch_html(url)
    source_method = "html"
    notes = [status]

    if html:
        parsed = extract_best_content(html)
        if looks_like_js_shell(html, parsed["role_context_text"]):
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

    structured = parsed["structured"]
    title_tag_text = parsed["title_tag_text"]
    header_text = parsed["header_text"]
    role_body_text = parsed["role_body_text"]
    role_context_text = parsed["role_context_text"]

    raw_title = first_nonempty(structured.get("title"), title_tag_text)
    clean_title = clean_job_title(raw_title)

    fallback_location = normalize_location_rule_based(header_text + "\n" + role_body_text, allowed_locations)
    fallback_remote_preferences = detect_remote_preferences_rule_based(header_text + "\n" + role_body_text)
    fallback_remote_days = detect_remote_days_rule_based(header_text + "\n" + role_body_text, fallback_remote_preferences)
    fallback_salary_min, fallback_salary_max, fallback_salary_currency, fallback_salary_period = parse_explicit_salary(
        header_text + "\n" + role_body_text,
        allowed_salaries
    )
    fallback_visa = detect_visa_rule_based(role_context_text)
    fallback_job_type = detect_job_type_rule_based(header_text + "\n" + role_body_text, structured.get("employment_type_raw", ""))
    fallback_description = clean_job_description(role_body_text)

    fallback_role_relevance = "Relevant" if clean_title else "Not relevant"
    fallback_reason = "Fallback classification based on extracted title and role text."

    relevance = ai_check_relevance(
        job_title=clean_title,
        role_context_text=role_context_text,
        fallback_role_relevance=fallback_role_relevance,
        fallback_reason=fallback_reason,
    )

    result.job_title = clean_title
    result.role_relevance = relevance.get("role_relevance", "") or fallback_role_relevance
    result.role_relevance_reason = relevance.get("role_relevance_reason", "") or fallback_reason
    result.source_method = source_method
    result.status = "ok"

    # Stop immediately on Not relevant
    if result.role_relevance == "Not relevant":
        result.notes = " | ".join(notes + ["stopped_after_relevance"])
        return result

    # Better fallback category
    fallback_job_category = "T&P" if re.search(
        r"\b("
        r"engineer|developer|software|data|machine learning|ml|ai|product|designer|qa|devops|research|"
        r"system administrator|system engineer|support engineer|solutions engineer|network engineer|"
        r"architect|cloud|platform|infrastructure|it support|2nd line|second line"
        r")\b",
        (clean_title + "\n" + role_body_text).lower()
    ) else "NonT&P"

    tagged = ai_tag_relevant_job(
        job_title=clean_title,
        header_text=header_text,
        role_body_text=role_body_text,
        allowed_locations=allowed_locations,
        allowed_salaries=allowed_salaries,
        allowed_job_titles=allowed_job_titles,
        fallback_job_category=fallback_job_category,
        fallback_location=fallback_location,
        fallback_remote_preferences=fallback_remote_preferences,
        fallback_remote_days=fallback_remote_days,
        fallback_salary_min=fallback_salary_min,
        fallback_salary_max=fallback_salary_max,
        fallback_salary_currency=fallback_salary_currency,
        fallback_salary_period=fallback_salary_period,
        fallback_visa=fallback_visa,
        fallback_job_type=fallback_job_type,
        fallback_job_description=fallback_description,
    )

    result.job_category = normalize_category_for_skills(tagged.get("job_category", "")) or fallback_job_category
    result.job_location = tagged.get("job_location", "") or fallback_location
    result.remote_preferences = tagged.get("remote_preferences", "") or fallback_remote_preferences
    result.remote_days = tagged.get("remote_days", "") or fallback_remote_days
    result.salary_min = tagged.get("salary_min", "") or fallback_salary_min
    result.salary_max = tagged.get("salary_max", "") or fallback_salary_max
    result.salary_currency = tagged.get("salary_currency", "") or fallback_salary_currency
    result.salary_period = tagged.get("salary_period", "") or fallback_salary_period
    result.visa_sponsorship = tagged.get("visa_sponsorship", "") or fallback_visa
    result.job_type = tagged.get("job_type", "") or fallback_job_type
    result.job_description = clean_job_description(tagged.get("job_description", "") or fallback_description)

    # Hard guard: if there is no explicit salary evidence, leave salary blank
    explicit_salary_check = parse_explicit_salary(header_text + "\n" + role_body_text, allowed_salaries)
    if explicit_salary_check == ("", "", "", ""):
        result.salary_min = ""
        result.salary_max = ""
        result.salary_currency = ""
        result.salary_period = ""

    # Titles
    job_titles = tagged.get("job_titles", []) if isinstance(tagged.get("job_titles", []), list) else []
    result.job_title_tag_1 = job_titles[0] if len(job_titles) > 0 else ""
    result.job_title_tag_2 = job_titles[1] if len(job_titles) > 1 else ""
    result.job_title_tag_3 = job_titles[2] if len(job_titles) > 2 else ""

    # Seniority
    seniorities = tagged.get("seniorities", []) if isinstance(tagged.get("seniorities", []), list) else []
    if not seniorities:
        seniorities = fallback_seniorities(clean_title, role_body_text)
    seniorities = normalize_seniority_list(seniorities)

    result.seniority_1 = seniorities[0] if len(seniorities) > 0 else ""
    result.seniority_2 = seniorities[1] if len(seniorities) > 1 else ""
    result.seniority_3 = seniorities[2] if len(seniorities) > 2 else ""

    # Skills: STRICTLY role-focused text only
    skills_source_text = build_skills_source_text(header_text, role_body_text)
    skill_list = tp_skills if result.job_category == "T&P" else nontp_skills

    exact_skills = exact_match_skills_in_order(skills_source_text, skill_list, limit=10)
    final_skills = ai_enrich_skills(
        role_category=result.job_category,
        description=skills_source_text,
        exact_skills=exact_skills,
        allowed_skills=skill_list,
    )

    # Final strict filter against allowed list
    allowed_lower = {s.lower(): s for s in skill_list}
    final_skills = [
        allowed_lower[sk.lower()]
        for sk in final_skills
        if sk.lower() in allowed_lower
    ]
    final_skills = dedupe_keep_order(final_skills)[:10]

    padded_skills = (final_skills + [""] * 10)[:10]
    result.skill_1 = padded_skills[0]
    result.skill_2 = padded_skills[1]
    result.skill_3 = padded_skills[2]
    result.skill_4 = padded_skills[3]
    result.skill_5 = padded_skills[4]
    result.skill_6 = padded_skills[5]
    result.skill_7 = padded_skills[6]
    result.skill_8 = padded_skills[7]
    result.skill_9 = padded_skills[8]
    result.skill_10 = padded_skills[9]

    note_parts = notes + ["ran_relevant_tagging"]
    if len(exact_skills) > 0:
        note_parts.append("skills_role_text_exact_plus_ai_enrichment")
    else:
        note_parts.append("skills_role_text_ai_only")
    result.notes = " | ".join(note_parts)

    return result


# -------------------------
# Output
# -------------------------

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
    allowed_locations = load_text_list(LOCATIONS_CSV)
    allowed_salaries = load_salary_list(SALARIES_CSV)
    tp_skills = load_text_list(TP_SKILLS_CSV)
    nontp_skills = load_text_list(NONTP_SKILLS_CSV)
    allowed_job_titles = load_text_list(JOB_TITLES_CSV)

    print(f"[INFO] URLs loaded: {len(urls)}")
    print(f"[INFO] Allowed locations loaded: {len(allowed_locations)}")
    print(f"[INFO] Allowed salaries loaded: {len(allowed_salaries)}")
    print(f"[INFO] T&P skills loaded: {len(tp_skills)}")
    print(f"[INFO] NonT&P skills loaded: {len(nontp_skills)}")
    print(f"[INFO] Job titles loaded: {len(allowed_job_titles)}")
    print(f"[INFO] OpenAI enabled: {'yes' if client else 'no'}")

    results: List[JobResult] = []

    for idx, url in enumerate(urls, start=1):
        print(f"[{idx}/{len(urls)}] Processing: {url}")
        try:
            row = process_url(
                url=url,
                allowed_locations=allowed_locations,
                allowed_salaries=allowed_salaries,
                tp_skills=tp_skills,
                nontp_skills=nontp_skills,
                allowed_job_titles=allowed_job_titles,
            )
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
