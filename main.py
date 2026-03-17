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
SALARIES_CSV = "predefined_salaries.csv"  # loaded only for compatibility; no rounding is applied anymore
TP_SKILLS_CSV = "predefined_tp_skills.csv"
NONTP_SKILLS_CSV = "predefined_nontp_skills.csv"
JOB_TITLES_CSV = "predefined_job_titles.csv"

REQUEST_TIMEOUT = 25
PLAYWRIGHT_TIMEOUT_MS = 30000
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

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
    text = text.replace("\xa0", " ")
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


def squeeze_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_quotes(text: str) -> str:
    return (
        (text or "")
        .replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )


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

            def route_handler(route):
                if route.request.resource_type in {"image", "media", "font"}:
                    try:
                        route.abort()
                    except Exception:
                        pass
                else:
                    try:
                        route.continue_()
                    except Exception:
                        pass

            page.route("**/*", route_handler)
            page.goto(url, wait_until="domcontentloaded", timeout=PLAYWRIGHT_TIMEOUT_MS)
            page.wait_for_timeout(2500)
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
        "please wait while we load",
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
# HTML / JSON extraction
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


def extract_embedded_json_candidates(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    candidates = []
    for script in soup.find_all("script"):
        raw = script.string or script.get_text(" ", strip=False)
        if not raw or len(raw) < 80:
            continue

        text = raw.strip()
        if text.startswith("{") or text.startswith("["):
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    candidates.append(data)
                elif isinstance(data, list):
                    candidates.extend([x for x in data if isinstance(x, dict)])
            except Exception:
                pass

        patterns = [
            r"window\.__INITIAL_STATE__\s*=\s*(\{.*?\})\s*;",
            r"window\.__INITIAL_DATA__\s*=\s*(\{.*?\})\s*;",
            r"window\.__NEXT_DATA__\s*=\s*(\{.*?\})\s*;",
            r"window\.__NUXT__\s*=\s*(\{.*?\})\s*;",
            r"__NEXT_DATA__"  # marker only, direct JSON already covered
        ]
        for pat in patterns[:4]:
            m = re.search(pat, text, flags=re.S)
            if not m:
                continue
            try:
                data = json.loads(m.group(1))
                if isinstance(data, dict):
                    candidates.append(data)
            except Exception:
                pass
    return candidates


def walk_for_strings(node: Any, out: List[str], max_items: int = 4000) -> None:
    if len(out) >= max_items:
        return
    if isinstance(node, str):
        s = clean_whitespace(strip_html(node) if "<" in node and ">" in node else node)
        if s and len(s) > 1:
            out.append(s)
        return
    if isinstance(node, dict):
        for v in node.values():
            walk_for_strings(v, out, max_items=max_items)
    elif isinstance(node, list):
        for item in node:
            walk_for_strings(item, out, max_items=max_items)


def extract_structured_fields(soup: BeautifulSoup) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "title": "",
        "location_raw": "",
        "salary_min_raw": None,
        "salary_max_raw": None,
        "salary_currency_raw": "",
        "salary_period_raw": "",
        "employment_type_raw": "",
        "description_raw": "",
        "company_name": "",
        "json_blob_text": "",
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
                parts = [
                    first_nonempty(addr.get("streetAddress")),
                    first_nonempty(addr.get("addressLocality")),
                    first_nonempty(addr.get("addressRegion")),
                    first_nonempty(addr.get("addressCountry")),
                ]
                data["location_raw"] = ", ".join([p for p in parts if p])
        elif isinstance(loc, list):
            parts = []
            for item in loc:
                if isinstance(item, dict):
                    addr = item.get("address", {})
                    if isinstance(addr, dict):
                        piece = ", ".join([
                            p for p in [
                                first_nonempty(addr.get("addressLocality")),
                                first_nonempty(addr.get("addressRegion")),
                                first_nonempty(addr.get("addressCountry")),
                            ] if p
                        ])
                        if piece:
                            parts.append(piece)
            data["location_raw"] = ", ".join(parts)

        base_salary = jp.get("baseSalary")
        if isinstance(base_salary, dict):
            data["salary_currency_raw"] = first_nonempty(base_salary.get("currency"))
            unit_text = str(base_salary.get("unitText", "") or "").lower()
            if unit_text in {"year", "yearly"}:
                data["salary_period_raw"] = "year"
            elif unit_text in {"month", "monthly"}:
                data["salary_period_raw"] = "month"
            elif unit_text in {"day", "daily"}:
                data["salary_period_raw"] = "day"
            elif unit_text in {"hour", "hourly"}:
                data["salary_period_raw"] = "hour"

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

    embedded = extract_embedded_json_candidates(soup)
    blob_strings: List[str] = []
    for item in embedded[:20]:
        walk_for_strings(item, blob_strings, max_items=2500)
    data["json_blob_text"] = clean_whitespace("\n".join(dedupe_keep_order(blob_strings)))[:40000]

    if not data["title"]:
        title_tag = soup.find("title")
        if title_tag:
            data["title"] = clean_whitespace(title_tag.get_text(" ", strip=True))

    return data


# -------------------------
# Text extraction and cleaning
# -------------------------

ROLE_KEEP_HEADINGS = {
    "about the role",
    "about the job",
    "job description",
    "role purpose",
    "purpose of the role",
    "context",
    "key responsibilities",
    "responsibilities",
    "responsibilities include",
    "what you'll do",
    "what you’ll do",
    "what to bring",
    "what you'll need",
    "what you’ll need",
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
    "purpose",
    "landscape",
    "summary",
    "full job description summary",
    "activities",
    "responsibilities",
    "what's in it for you",
    "what’s in it for you",
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
    "about aqa",
    "logo",
}

NOISE_LINE_PATTERNS = [
    r"^©\s*\d{4}",
    r"^follow us$",
    r"^read more$",
    r"^privacy policy$",
    r"^recruitment agencies$",
    r"^temporary roles:$",
    r"^permanent and fixed term roles:$",
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


def normalize_heading(line: str) -> str:
    low = normalize_quotes(line).lower().strip(" :-•\t")
    low = re.sub(r"\s+", " ", low)
    return low


def looks_like_heading(line: str) -> bool:
    x = normalize_heading(line)
    if not x:
        return False
    if x in ROLE_KEEP_HEADINGS or x in ROLE_STOP_HEADINGS:
        return True
    if len(x) <= 70 and not re.search(r"[.!?]", x):
        keyword_hits = [
            "responsib",
            "qualif",
            "experience",
            "desirable",
            "essential",
            "what you'll",
            "what you’ll",
            "relationships",
            "measures",
            "summary",
            "purpose",
            "overview",
            "landscape",
        ]
        if any(k in x for k in keyword_hits):
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

    for line in lines[:60]:
        low = line.lower()
        if (
            "location" in low
            or "country" in low
            or "territories available" in low
            or "ideal locations" in low
            or "based at" in low
            or "based in" in low
            or "home based" in low
            or "hybrid" in low
            or "remote" in low
            or "onsite" in low
            or "office" in low
            or "salary" in low
            or "rate" in low
            or "contract" in low
            or "full time" in low
            or "full-time" in low
            or "part time" in low
            or "part-time" in low
            or "position role type" in low
        ):
            keep.append(line)

    return clean_whitespace("\n".join(dedupe_keep_order(keep)))


def prune_role_lines(lines: List[str]) -> List[str]:
    out: List[str] = []
    stop_mode = False

    for line in lines:
        norm = normalize_heading(line)
        low = line.lower().strip()

        if line_is_noise(line):
            continue

        if norm in ROLE_STOP_HEADINGS:
            stop_mode = True
            continue

        if stop_mode:
            if looks_like_heading(line) and norm in ROLE_KEEP_HEADINGS:
                stop_mode = False
            else:
                continue

        if any(
            phrase in low
            for phrase in [
                "all rights reserved",
                "privacy policy",
                "follow us",
                "read more",
                "recruitment agencies",
                "reasonable adjustments",
                "equality, diversity and inclusion",
                "drop us a note to find out more",
                "unsolicited cvs",
            ]
        ):
            continue

        out.append(line)

    return out


def extract_role_body_text(lines: List[str]) -> str:
    if not lines:
        return ""

    lines = prune_role_lines(lines)
    if not lines:
        return ""

    kept: List[str] = []
    in_role_zone = False

    starter_patterns = [
        "about the role",
        "about the job",
        "job description",
        "role purpose",
        "purpose of the role",
        "key responsibilities",
        "responsibilities",
        "minimum qualifications",
        "preferred qualifications",
        "what you'll do",
        "what you’ll do",
        "what to bring",
        "qualifications",
        "who we are looking for",
        "skills & experience",
        "skills and experience",
        "context",
        "summary",
        "full job description summary",
    ]

    for line in lines:
        norm = normalize_heading(line)
        low = line.lower().strip()

        if not in_role_zone:
            if norm in ROLE_KEEP_HEADINGS or any(p in low for p in starter_patterns):
                in_role_zone = True
                kept.append(line)
                continue

            if re.search(
                r"\b(key responsibilities|minimum qualifications|preferred qualifications|responsibilities|essential skills|what to bring|who we are looking for|job requirements)\b",
                low,
            ):
                in_role_zone = True
                kept.append(line)
                continue

        if in_role_zone:
            if norm in ROLE_STOP_HEADINGS:
                continue
            kept.append(line)

    if len(kept) < 12:
        kept = [line for line in lines if normalize_heading(line) not in ROLE_STOP_HEADINGS]

    return clean_whitespace("\n".join(kept))


def clean_job_description(text: str) -> str:
    if not text:
        return ""

    lines = split_lines(text)
    cleaned: List[str] = []

    hard_stop_patterns = [
        r"^recruitment agencies$",
        r"^privacy policy$",
        r"^read more$",
        r"^follow us$",
        r"^contact$",
        r"^reasonable adjustments$",
        r"^smart working$",
        r"^equality, diversity and inclusion$",
        r"^safeguarding$",
        r"^additional information$",
    ]

    for line in lines:
        low = line.lower().strip()

        if line_is_noise(line):
            continue

        if any(re.search(p, low, flags=re.I) for p in hard_stop_patterns):
            continue

        if any(
            phrase in low
            for phrase in [
                "all rights reserved",
                "workday, inc.",
                "smartrecruiters",
                "drop us a note to find out more",
                "follow us on linkedin",
                "apply now",
            ]
        ):
            continue

        cleaned.append(line)

    out = clean_whitespace("\n".join(dedupe_keep_order(cleaned)))
    return out[:30000]


def merge_role_bodies(*candidates: str) -> str:
    merged_lines: List[str] = []
    for text in candidates:
        for line in split_lines(text):
            merged_lines.append(line)
    __already_fixed__


def job_text_score(text: str) -> int:
    low = normalize_quotes(text or "").lower()
    score = 0

    positive_terms = [
        "responsibilities",
        "duties and responsibilities",
        "requirements",
        "qualifications",
        "experience",
        "skills",
        "knowledge, skills, and abilities",
        "person specification",
        "job description",
        "overview",
        "benefits",
        "preferred",
        "required",
        "certifications",
        "availability",
        "what you'll do",
        "what you’ll do",
        "what to bring",
        "key responsibilities",
        "minimum qualifications",
        "preferred qualifications",
        "desirable",
        "key relationships",
        "success measures",
    ]
    for term in positive_terms:
        if term in low:
            score += 3

    strong_terms = [
        "kubernetes",
        "openshift",
        "linux",
        "vmware",
        "azure",
        "storage",
        "monitoring",
        "on-call",
        "root cause",
        "python",
        "tensorflow",
        "pytorch",
        "account management",
        "customer success",
        "risk & compliance",
        "machine learning",
    ]
    for term in strong_terms:
        if term in low:
            score += 2

    negative_terms = [
        "about us",
        "company overview",
        "who we are",
        "our mission",
        "service listing",
        "our offerings",
        "follow us",
        "privacy policy",
        "recruitment agencies",
        "read more",
    ]
    for term in negative_terms:
        if term in low:
            score -= 3

    score += min(len(text) // 500, 10)
    return score


def extract_best_content(html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")
    visible_text = soup_text(soup)
    structured = extract_structured_fields(soup)

    description_raw = structured.get("description_raw") or ""
    structured_description_text = strip_html(description_raw)
    json_blob_text = structured.get("json_blob_text", "")

    title_tag = soup.find("title")
    title_tag_text = clean_whitespace(title_tag.get_text(" ", strip=True)) if title_tag else ""

    visible_lines = split_lines(visible_text)
    structured_lines = split_lines(structured_description_text)
    json_lines = split_lines(json_blob_text)

    header_text = build_header_text(
        visible_lines if visible_lines else structured_lines,
        structured,
        title_tag_text,
    )

    visible_role_body = extract_role_body_text(visible_lines)
    structured_role_body = extract_role_body_text(structured_lines) if structured_lines else ""
    json_role_body = extract_role_body_text(json_lines) if json_lines else ""

    candidates = [
        ("visible", clean_job_description(visible_role_body)),
        ("structured", clean_job_description(structured_role_body)),
        ("json", clean_job_description(json_role_body)),
    ]
    candidates = [(name, body) for name, body in candidates if body]

    best_body = ""
    if candidates:
        scored = [(job_text_score(body), len(body), name, body) for name, body in candidates]
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best_body = scored[0][3]

    role_body_text = best_body
    if len(role_body_text.strip()) < 600:
        role_body_text = merge_role_bodies(visible_role_body, structured_role_body, json_role_body)

    role_context_text = clean_whitespace("\n".join([header_text, role_body_text]))
    all_page_text = clean_whitespace("\n".join([
        title_tag_text,
        structured.get("title", ""),
        structured.get("company_name", ""),
        structured.get("location_raw", ""),
        structured_description_text,
        json_blob_text,
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

    title = clean_whitespace(normalize_quotes(raw_title))

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


def gather_location_lines(text: str) -> List[str]:
    out = []
    lines = split_lines(text)

    strong_patterns = [
        r"^location\s*:",
        r"^locations\s*:",
        r"^country\s*:",
        r"^city\s*:",
        r"^office\s*:",
        r"^based at\b",
        r"^based in\b",
        r"^home based\b",
        r"^position role type\s*:",
        r"\blocated in\b",
        r"\bthis is a .* position located in\b",
        r"\bwork location\b",
        r"\bideal locations\b",
        r"\bterritories available\b",
    ]
    soft_tokens = [" hybrid", " remote", " onsite", " on-site", " home based", " from home"]

    for idx, line in enumerate(lines[:180]):
        low = f" {normalize_quotes(line).lower()} "
        if any(re.search(p, low, flags=re.I) for p in strong_patterns):
            out.append(line)
            continue
        if idx < 20 and any(tok in low for tok in soft_tokens):
            out.append(line)

    out.extend(lines[:12])
    return dedupe_keep_order(out)


def location_value_has_evidence(location_value: str, text: str, allowed_locations: List[str]) -> bool:
    if not location_value or not text:
        return False

    candidate_lines = gather_location_lines(text)
    if not candidate_lines:
        return False

    joined = normalize_quotes("\n".join(candidate_lines)).lower()
    loc_low = normalize_quotes(location_value).lower().strip()
    if loc_low and re.search(rf"(?<![a-z]){re.escape(loc_low)}(?![a-z])", joined):
        return True

    city_lookup = build_location_lookup(allowed_locations)
    city = loc_low.split(",")[0].strip()
    mapped = city_lookup.get(city, "")
    city_to_check = mapped.split(",")[0].strip().lower() if mapped else city
    if city_to_check and re.search(rf"\b{re.escape(city_to_check)}\b", joined):
        return True

    return False


def normalize_location_rule_based(text: str, allowed_locations: List[str]) -> str:
    if not allowed_locations or not text:
        return ""

    city_lookup = build_location_lookup(allowed_locations)
    candidate_lines = gather_location_lines(text)
    if not candidate_lines:
        return ""

    joined = "\n".join(candidate_lines)
    joined_low = normalize_quotes(joined).lower()

    exact_hits = []
    for loc in allowed_locations:
        loc_low = normalize_quotes(loc).lower()
        m = re.search(rf"(?<![a-z]){re.escape(loc_low)}(?![a-z])", joined_low)
        if m:
            exact_hits.append((m.start(), -len(loc), loc))

    if exact_hits:
        exact_hits.sort(key=lambda x: (x[0], x[1]))
        return exact_hits[0][2]

    city_hits = []
    for city, full in city_lookup.items():
        m = re.search(rf"\b{re.escape(city)}\b", joined_low)
        if m:
            city_hits.append((m.start(), -len(city), full))

    if city_hits:
        city_hits.sort(key=lambda x: (x[0], x[1]))
        return city_hits[0][2]

    broad_fallbacks = [
        "United Kingdom",
        "England, United Kingdom",
        "Scotland, United Kingdom",
        "Wales, United Kingdom",
        "Northern Ireland, United Kingdom",
        "Ireland",
    ]
    for broad in broad_fallbacks:
        for loc in allowed_locations:
            if loc.lower() == broad.lower() and re.search(rf"(?<![a-z]){re.escape(broad.lower())}(?![a-z])", joined_low):
                return loc

    return ""


def detect_remote_preferences_rule_based(text: str) -> str:
    low = normalize_quotes(text).lower()
    found = []

    if re.search(r"\bhome based\b|\bhome-based\b|\buk remote\b|\bfully remote\b|\bremote\b", low):
        found.append("remote")
    if re.search(r"\bhybrid\b", low):
        found.append("hybrid")
    if re.search(r"\bonsite\b|\bon-site\b|\bon site\b|\bin office\b|\bin-office\b", low):
        found.append("onsite")

    ordered = [x for x in ["onsite", "hybrid", "remote"] if x in found]
    return ", ".join(ordered)


def detect_remote_days_rule_based(text: str, remote_prefs: str) -> str:
    low = normalize_quotes(text).lower()

    if "hybrid" not in remote_prefs and "remote" not in remote_prefs:
        return ""

    patterns = [
        r"\b(\d)\s*[-to]{1,3}\s*(\d)\s+days?\s+(?:in the office|from the office|office|on site|onsite|on-site)\b",
        r"\b(\d)\s+days?\s+(?:in the office|from the office|office|on site|onsite|on-site)\b",
        r"\b(\d)\s*[-to]{1,3}\s*(\d)\s+days?\s+(?:from home|remote|wfh)\b",
        r"\b(\d)\s+days?\s+(?:from home|remote|wfh)\b",
        r"\b(?:work|working)\s+(\d)\s+days?\s+(?:from home|remote|wfh)\b",
    ]

    for pat in patterns:
        m = re.search(pat, low)
        if not m:
            continue
        groups = [g for g in m.groups() if g is not None]
        if "office" in pat or "on site" in pat or "onsite" in pat:
            if len(groups) == 2:
                vals = [int(groups[0]), int(groups[1])]
                remote_options = [max(0, 5 - v) for v in vals]
                return str(min(remote_options))
            return str(max(0, 5 - int(groups[0])))
        if len(groups) == 2:
            return groups[1]
        return groups[0]

    return ""


def normalize_currency_symbol(sym: str) -> str:
    return {"£": "GBP", "$": "USD", "€": "EUR"}.get(sym, "")


def looks_like_non_salary_line(line: str) -> bool:
    low = line.lower()
    bad_tokens = [
        "reference:",
        "posted on",
        "posted",
        "apply by",
        "closing date",
        "interview",
        "q1",
        "q2",
        "q3",
        "q4",
        "roi",
        "employees",
        "revenue",
        "years of experience",
        "iso",
        "nist",
        "800-53",
        "2024",
        "2025",
        "2026",
        "2027",
        "2028",
        "m365",
    ]
    return any(tok in low for tok in bad_tokens)


def line_has_compensation_anchor(line: str) -> bool:
    low = line.lower()
    anchors = [
        "salary",
        "pay",
        "salary band",
        "salary location band",
        "compensation",
        "package",
        "rate",
        "per annum",
        "annum",
        "annual",
        "day rate",
        "daily rate",
        "hourly",
        "per day",
        "per hour",
        "p/d",
        "p.a",
    ]
    return any(a in low for a in anchors) or bool(re.search(r"[£$€]", line))


def parse_money_range_from_line(line: str) -> Tuple[str, str, str, str]:
    line = normalize_quotes(line)
    low = line.lower()

    if looks_like_non_salary_line(line) or not line_has_compensation_anchor(line):
        return "", "", "", ""

    period = ""
    if re.search(r"\b(p/d|per day|day rate|daily rate)\b", low):
        period = "day"
    elif re.search(r"\b(per hour|hourly)\b", low):
        period = "hour"
    elif re.search(r"\b(per month|monthly)\b", low):
        period = "month"
    elif re.search(r"\b(per annum|annum|annual|annually|salary)\b", low):
        period = "year"

    range_pat = re.compile(
        r"(£|\$|€)\s?(\d{1,3}(?:,\d{3})+|\d{4,6})(?:\.\d+)?\s*(?:-|–|to)\s*(?:\1\s?)?(\d{1,3}(?:,\d{3})+|\d{4,6})(?:\.\d+)?",
        flags=re.I,
    )
    m = range_pat.search(line)
    if m:
        currency = normalize_currency_symbol(m.group(1))
        min_raw = re.sub(r"[^\d]", "", m.group(2))
        max_raw = re.sub(r"[^\d]", "", m.group(3))
        if min_raw and max_raw:
            return min_raw, max_raw, currency, period

    single_pat = re.compile(r"(£|\$|€)\s?(\d{1,3}(?:,\d{3})+|\d{4,6})(?:\.\d+)?", flags=re.I)
    singles = list(single_pat.finditer(line))
    if len(singles) == 1:
        currency = normalize_currency_symbol(singles[0].group(1))
        raw = re.sub(r"[^\d]", "", singles[0].group(2))
        if raw:
            return raw, raw, currency, period

    return "", "", "", ""


def parse_explicit_salary(text: str, _allowed_salaries_unused: List[int]) -> Tuple[str, str, str, str]:
    lines = split_lines(text)
    candidate_lines = [line for line in lines[:120] if line_has_compensation_anchor(line)]

    for line in candidate_lines:
        parsed = parse_money_range_from_line(line)
        if parsed != ("", "", "", ""):
            return parsed

    return "", "", "", ""


def detect_visa_rule_based(text: str) -> str:
    low = normalize_quotes(text).lower()

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
        r"subject to .*right to work",
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

    if any(x in title_low for x in ["head of", "director", "vp ", "vice president", "chief ", "cfo", "cto", "cio", "coo", "cmo", "cro", "cpo", "cso"]):
        return ["leadership"]

    if any(x in title_low for x in ["senior", "sr "]):
        return ["senior", "lead"]

    if any(x in title_low for x in ["lead ", "principal"]):
        return ["lead"]

    if any(x in title_low for x in ["junior", "jr "]):
        return ["junior", "mid"]

    if any(x in title_low for x in ["associate", "mid weight", "mid-weight"]):
        return ["mid"]

    years = []
    for m in re.finditer(r"\b(\d)\s*-\s*(\d)\s+years?\b", text_low):
        years.extend([int(m.group(1)), int(m.group(2))])
    for m in re.finditer(r"\b(\d)\+?\s+years?\b", text_low):
        years.append(int(m.group(1)))

    if years:
        max_y = max(years)
        if max_y <= 1:
            return ["entry", "junior"]
        if max_y == 2:
            return ["junior", "mid"]
        if 3 <= max_y <= 5:
            return ["senior"]
        if max_y > 5:
            return ["senior", "lead"]

    return []


def is_tp_by_rules(job_title: str, role_text: str) -> bool:
    text = f"{job_title}\n{role_text}".lower()
    tp_patterns = [
        r"\bengineer\b",
        r"\bdeveloper\b",
        r"\bsoftware\b",
        r"\bdata\b",
        r"\bmachine learning\b",
        r"\bml\b",
        r"\bai\b",
        r"\bproduct\b",
        r"\bdesigner\b",
        r"\bux\b",
        r"\bui\b",
        r"\bqa\b",
        r"\bdevops\b",
        r"\bsite reliability\b",
        r"\bsre\b",
        r"\barchitect\b",
        r"\bcloud\b",
        r"\bplatform\b",
        r"\binfrastructure\b",
        r"\bsystem administrator\b",
        r"\bsystems administrator\b",
        r"\bsupport engineer\b",
        r"\btechnical support\b",
        r"\bnetwork engineer\b",
        r"\bsolutions engineer\b",
        r"\bit support\b",
        r"\b2nd line\b",
        r"\bsecond line\b",
        r"\bmsp\b",
        r"\bwindows server\b",
        r"\bactive directory\b",
        r"\bexchange\b",
        r"\bhyper-v\b",
        r"\bvmware\b",
        r"\bcitrix\b",
        r"\brouter\b",
        r"\bfirewall\b",
        r"\bvpn\b",
        r"\bremote desktop\b",
        r"\bvoip\b",
    ]
    return any(re.search(p, text) for p in tp_patterns)


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
- T&P ALSO includes IT support, helpdesk, desktop support, 2nd line / 3rd line, infrastructure support, MSP engineering, field engineering, and technical customer-environment support roles.
- NonT&P is everything else that is still relevant.

Rules for job_location:
- Use header/meta text first, then role description.
- Prefer the most specific visible location over a broad country fallback.
- If multiple valid locations are listed, choose the first clear normalized location from the allowed list or the closest broader valid one.
- Only return "Unknown" if there is truly no clear location evidence.

Rules for remote_preferences:
- Allowed values only: onsite, hybrid, remote
- Use role-specific/header evidence first.
- home based counts as remote.
- daily field travel does NOT mean onsite only.
- Do not let generic company-wide smart-working text override a clearer role-specific line.

Rules for remote_days:
- Return only a single number or ""
- Only return a number when explicit days are stated.
- Never infer a number from the word "hybrid" alone.
- Do not infer from company-wide smart-working statements.

Rules for salary:
- Only tag salary when compensation/rate is explicitly stated for this role.
- salary_period must be one of: year, day, hour, month, or ""
- If a daily rate is stated (e.g. £500-£550 p/d), preserve it as min/max with salary_period=day.
- Do not invent salary from unrelated numbers, dates, standards, reference ids, years, counts, targets, ISO/NIST values, legislation references, percentages, revenues, or employee counts.
- Do not normalize or round salary to a predefined list. Preserve the actual explicit number.

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
- Keep the main role content.
- Preserve useful sections such as:
  responsibilities, required, essential, desirable, qualifications, what to bring, key relationships, success measures, minimum qualifications, preferred qualifications, summary, activities, landscape.
- Remove company/benefits/footer/privacy/apply/legal/ATS boilerplate.
- Do not shorten to one paragraph if multiple role sections are clearly present.

Rules for job_titles:
- You may return more than one job title if clearly supported.
- Analyze job title and role description together.
- Return up to 3 exact strings from the predefined list.
- Prefer the real job function over buzzwords.
- Do NOT return leadership titles like Head of Engineering unless the role is explicitly that level.
- For account-management style roles, prefer CSM/Account Manager when the function is relationship/account ownership rather than net-new sales.
- For customer-facing technical roles, Solutions Engineer can coexist with another technical title when clearly supported.
- For AI Solution Architect, prefer Technical Architect / AI Engineer style tags, not Head of Engineering.

Rules for seniorities:
- Allowed only: entry, junior, mid, senior, lead, leadership
- Must be lowercase
- Return as JSON array in this order only: entry, junior, mid, senior, lead, leadership
- If title includes explicit org-leadership indicators like "head of", "director", "vp", "chief", return leadership when appropriate.
- Do NOT use leadership only because the role has technical authority or architectural ownership.
- If title says senior, include senior.
- If role clearly suggests multiple levels, return up to 3.

Allowed normalized locations:
{location_list_text}

Allowed salaries reference list (do not round to these; use only as background context if needed):
{salary_list_text}

Predefined job titles:
{job_titles_text}

Input job title:
{job_title}

Header/meta text:
{header_text[:6000]}

Role description:
{role_body_text[:24000]}
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

    special_map = {
        "c++": r"(?<![A-Za-z0-9])c\+\+(?![A-Za-z0-9])",
        "c#": r"(?<![A-Za-z0-9])c#(?![A-Za-z0-9])",
        ".net": r"(?<![A-Za-z0-9])(?:\.net|dotnet|dot net)(?![A-Za-z0-9])",
        "node.js": r"(?<![A-Za-z0-9])(?:node\.js|nodejs|node js)(?![A-Za-z0-9])",
        "next.js": r"(?<![A-Za-z0-9])(?:next\.js|nextjs|next js)(?![A-Za-z0-9])",
        "vue.js": r"(?<![A-Za-z0-9])(?:vue\.js|vuejs|vue js)(?![A-Za-z0-9])",
        "nuxt.js": r"(?<![A-Za-z0-9])(?:nuxt\.js|nuxtjs|nuxt js)(?![A-Za-z0-9])",
        "git": r"(?<![A-Za-z0-9])git(?![A-Za-z0-9])",
    }
    if normalized in special_map:
        return re.compile(special_map[normalized], flags=re.I)

    escaped = re.escape(skill)
    return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", flags=re.I)


def build_skills_source_text(role_body_text: str) -> str:
    lines = split_lines(role_body_text)
    out = []

    disallow = [
        "smartrecruiters",
        "workday, inc.",
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
        "uk & ireland's premier aws",
        "microsoft & oracle partner",
        "great place to work",
        "employee wellbeing",
        "annual excellence awards",
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
- Do not use page chrome, partner lists, footer text, platform names, legal text, company marketing, or brand badges.
- Do not infer adjacent technologies unless they are clearly mentioned in the role-focused description.

Hard rules:
- role_category is PROVIDED as input. You MUST echo it exactly as given ("T&P" or "NonT&P"). Do not change it.
- skills MUST be chosen ONLY from the correct Allowed skills list (exact string match).
- exact_skills is the PRIMARY base set. Preserve them.
- NEVER output a skill not present in the allowed list.
- Before finalizing, verify every returned skill appears exactly in the allowed list.
- Keep exact_skills first, then add missing clearly evidenced skills.
- Return up to 10 skills total.
- If you are not sure a skill is evidenced in the text, exclude it.

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
        allowed_salaries,
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

    if result.role_relevance == "Not relevant":
        result.notes = " | ".join(notes + ["stopped_after_relevance"])
        return result

    fallback_job_category = "T&P" if is_tp_by_rules(clean_title, role_body_text) else "NonT&P"

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

    ai_location = tagged.get("job_location", "") or ""
    evidence_text_for_location = "\n".join([header_text, role_body_text, structured.get("location_raw", "")])
    if ai_location and location_value_has_evidence(ai_location, evidence_text_for_location, allowed_locations):
        result.job_location = ai_location
    else:
        result.job_location = fallback_location

    result.remote_preferences = tagged.get("remote_preferences", "") or fallback_remote_preferences
    result.remote_days = tagged.get("remote_days", "") or fallback_remote_days
    result.salary_min = tagged.get("salary_min", "") or fallback_salary_min
    result.salary_max = tagged.get("salary_max", "") or fallback_salary_max
    result.salary_currency = tagged.get("salary_currency", "") or fallback_salary_currency
    result.salary_period = tagged.get("salary_period", "") or fallback_salary_period
    result.visa_sponsorship = tagged.get("visa_sponsorship", "") or fallback_visa
    result.job_type = tagged.get("job_type", "") or fallback_job_type
    result.job_description = clean_job_description(tagged.get("job_description", "") or fallback_description)

    # Final hard overrides so the model cannot invent values.
    result.job_location = normalize_location_rule_based(header_text + "\n" + result.job_description, allowed_locations) or result.job_location
    result.remote_preferences = detect_remote_preferences_rule_based(header_text + "\n" + result.job_description) or result.remote_preferences
    result.remote_days = detect_remote_days_rule_based(header_text + "\n" + result.job_description, result.remote_preferences)

    explicit_salary_check = parse_explicit_salary(header_text + "\n" + result.job_description, allowed_salaries)
    if explicit_salary_check == ("", "", "", ""):
        result.salary_min = ""
        result.salary_max = ""
        result.salary_currency = ""
        result.salary_period = ""
    else:
        result.salary_min, result.salary_max, result.salary_currency, result.salary_period = explicit_salary_check

    job_titles = tagged.get("job_titles", []) if isinstance(tagged.get("job_titles", []), list) else []
    result.job_title_tag_1 = job_titles[0] if len(job_titles) > 0 else ""
    result.job_title_tag_2 = job_titles[1] if len(job_titles) > 1 else ""
    result.job_title_tag_3 = job_titles[2] if len(job_titles) > 2 else ""

    seniorities = tagged.get("seniorities", []) if isinstance(tagged.get("seniorities", []), list) else []
    if not seniorities:
        seniorities = fallback_seniorities(clean_title, role_body_text)
    seniorities = normalize_seniority_list(seniorities)

    result.seniority_1 = seniorities[0] if len(seniorities) > 0 else ""
    result.seniority_2 = seniorities[1] if len(seniorities) > 1 else ""
    result.seniority_3 = seniorities[2] if len(seniorities) > 2 else ""

    skills_source_text = build_skills_source_text(result.job_description)
    skill_list = tp_skills if result.job_category == "T&P" else nontp_skills

    exact_skills = exact_match_skills_in_order(skills_source_text, skill_list, limit=10)
    final_skills = ai_enrich_skills(
        role_category=result.job_category,
        description=skills_source_text,
        exact_skills=exact_skills,
        allowed_skills=skill_list,
    )

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
        note_parts.append("skills_from_clean_role_text_exact_plus_ai_enrichment")
    else:
        note_parts.append("skills_from_clean_role_text_ai_only")
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
