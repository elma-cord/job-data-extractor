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


def clean_whitespace(text: str) -> str:
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


def soup_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.extract()
    return clean_whitespace(soup.get_text("\n", strip=True))


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
        "interested in this",
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
        r"(?is)\bequal opportunit(?:y|ies)\b.*$",
        r"(?is)\binterested in this\b.*$",
    ]
    for pat in boilerplate_patterns:
        text = re.sub(pat, "", text)

    return clean_whitespace(text)[:20000]


def normalize_category_for_skills(job_category: str) -> str:
    low = (job_category or "").strip().lower()
    if low in {"t&p", "tp", "tech & product", "tech and product"}:
        return "T&P"
    if low in {"nont&p", "non-t&p", "non tp", "not t&p", "not tp", "nontp", "non tech", "non-tech"}:
        return "NonT&P"
    return ""


def normalize_location_rule_based(raw_location: str, text: str, allowed_locations: List[str]) -> str:
    if not allowed_locations:
        return clean_whitespace(raw_location)

    candidates = []
    if raw_location:
        candidates.append(clean_whitespace(raw_location))

    lines = [x.strip() for x in text.splitlines() if x.strip()]
    for line in lines[:400]:
        if re.search(r"\blocation\b|\bbased in\b|\boffice\b|\bremote\b|\buk\b|\blondon\b", line, flags=re.I):
            candidates.append(clean_whitespace(line))

    joined = " || ".join(candidates)
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

    return "Unknown"


def detect_remote_preferences_rule_based(text: str) -> str:
    low = text.lower()
    found = []

    onsite_patterns = [r"\bonsite\b", r"\bon-site\b", r"\bon site\b", r"\bin office\b", r"\bin-office\b"]
    hybrid_patterns = [r"\bhybrid\b", r"\bhybrid working\b", r"\bhybrid role\b", r"\bhybrid model\b"]
    remote_patterns = [r"\bremote\b", r"\bfully remote\b", r"\bwork from home\b", r"\bwfh\b"]

    if any(re.search(p, low) for p in onsite_patterns):
        found.append("onsite")
    if any(re.search(p, low) for p in hybrid_patterns):
        found.append("hybrid")
    if any(re.search(p, low) for p in remote_patterns):
        found.append("remote")

    return ", ".join(found)


def detect_remote_days_rule_based(text: str, remote_prefs: str) -> str:
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
            return m.group(2) if len(m.groups()) == 2 else m.group(1)

    for pat in office_patterns:
        m = re.search(pat, low)
        if m:
            if len(m.groups()) == 2:
                low_office = int(m.group(1))
                high_office = int(m.group(2))
                return str(max(5 - low_office, 5 - high_office))
            return str(max(0, 5 - int(m.group(1))))

    return ""


def nearest_salary(value: Optional[int], allowed_salaries: List[int]) -> str:
    if value is None:
        return ""
    if not allowed_salaries:
        return str(value)
    nearest = min(allowed_salaries, key=lambda x: abs(x - value))
    return str(nearest)


def parse_salary_candidates(text: str) -> List[Tuple[int, str]]:
    text = text.replace("\xa0", " ")
    candidates: List[Tuple[int, str]] = []

    patterns = [
        r"(£|\$|€)\s?(\d{1,3}(?:[,\s]\d{3})+|\d+)(?:\s?([kK]))?",
        r"\b(GBP|USD|EUR|CAD)\s?(\d{1,3}(?:[,\s]\d{3})+|\d+)(?:\s?([kK]))?",
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


def extract_salary_rule_based(text: str, structured: Dict[str, Any], allowed_salaries: List[int]) -> Tuple[str, str, str]:
    min_raw = structured.get("salary_min_raw")
    max_raw = structured.get("salary_max_raw")
    curr_raw = first_nonempty(structured.get("salary_currency_raw")).upper()

    if min_raw is not None or max_raw is not None:
        if min_raw is None and max_raw is not None:
            min_raw = max_raw
        if max_raw is None and min_raw is not None:
            max_raw = min_raw
        return (
            nearest_salary(int(min_raw), allowed_salaries) if min_raw is not None else "",
            nearest_salary(int(max_raw), allowed_salaries) if max_raw is not None else "",
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


def detect_visa_rule_based(text: str) -> str:
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


def build_skill_regex(skill: str) -> re.Pattern:
    escaped = re.escape(skill)

    special_exact = {"c++", "c#", ".net", "node.js", "next.js", "vue.js", "nuxt.js"}
    if skill.strip().lower() in special_exact:
        pattern = rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])"
        return re.compile(pattern, flags=re.I)

    pattern = rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])"
    return re.compile(pattern, flags=re.I)


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
    ordered = [x["skill"] for x in found[:limit]]
    return ordered


def extract_best_content(html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")
    visible_text = soup_text(soup)
    structured = extract_structured_fields(soup)

    description_raw = structured.get("description_raw") or ""
    structured_description_text = strip_html(description_raw)

    title_tag = soup.find("title")
    title_tag_text = clean_whitespace(title_tag.get_text(" ", strip=True)) if title_tag else ""

    all_page_text_parts = [
        title_tag_text,
        structured.get("title", ""),
        structured.get("company_name", ""),
        structured.get("location_raw", ""),
        structured_description_text,
        visible_text,
    ]
    all_page_text = clean_whitespace("\n".join([x for x in all_page_text_parts if x]))

    return {
        "soup": soup,
        "structured": structured,
        "title_tag_text": title_tag_text,
        "all_page_text": all_page_text,
    }


def ai_check_relevance(
    job_title: str,
    full_page_text: str,
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
- role_relevance_reason must be concise

Use this logic exactly:

1. Role Relevance - Decide if the role is Relevant or Not relevant according to these criteria:

Relevant roles match any in this list or close synonyms/specializations:

Account Director, Account Executive, AI Engineer, Automation Engineer, Back End, BI Developer, Big Data Engineer, Brand Marketing, Business Analyst, Business Development Manager, Business Operations, CDO, CFO, Chief of Staff, CIO, CLO, Cloud Engineer, CMO, Computer Vision Engineer, Content Marketing, COO, Copywriting, CPO, CRM Developer, CRM Manager, CRO, CSM / Account Manager, CSO, CTO, Customer Operations, Customer Service Rep, Customer Support, Data Architect, Data Engineer, Data Scientist, Data / Insight Analyst, Database Engineer, Deep Learning Engineer, Demand / Lead Generation, Developer in Test, DevOps Engineer, Digital Marketing, Embedded Developer, Engineering Manager, Events & Community, Executive Assistant, Finance / Accounting, Founder, FP&A, Front End, Full Stack, Games Designer, Games Developer, Generalist Marketing, Graphic Designer, Graphics Developer, Growth Marketing, Head of Customer, Head of Data, Head of Design, Head of Engineering, Head of Finance, Head of HR, Head of Infrastructure, Head of Marketing, Head of Operations, Head of Product, Head of QA, Head of Sales, Human Resources, Implementation Manager, Integration Developer, Legal, Machine Learning Engineer, Marketing Analyst, Mobile Developer, Network Engineer, Operations, Partnerships, Penetration Tester, People Ops, Performance Marketing, PR / Communications, Product Manager, Product Marketing, Product Owner, Project Manager, QA Automation Tester, QA Manual Tester, Quality Assurance, Quantitative Developer, Renewals Manager, Research Engineer, RevOps, Risk & Compliance, Sales Engineer, Sales Operations, Scrum Master, SDR / BDR, Security, Security Engineer, SEO Marketing, Site Reliability Engineer, Social Media Marketing, Solutions Engineer, Support Engineer, System Administrator, System Engineer, Talent Acquisition, Technical Architect, Technical Director, Technical Writer, Testing Manager, UI Designer, UI/UX Designer, UX Designer, UX Researcher, Videography, VP of Engineering.

If the role is clearly outside tech or business functions (e.g., teacher, nurse, waiter), mark Not Relevant, even if some criteria match.

Exclude any roles related to construction, civil engineering, retail, electrical, mechanical, manufacturing, microbiology, maritime, injection molding, and beauty brands.

Location - The role is Relevant only if all stated working locations are within these allowed locations:

a) United Kingdom (onsite, hybrid, or remote)
b) For Ireland, only remote roles are allowed. If the location mentions any onsite or hybrid work, mark Not Relevant.
c) For Europe, only roles explicitly marked as “Remote Europe” or “Remote EMEA” are allowed. Roles based in specific European countries (e.g., Germany, France) are Not Relevant unless explicitly remote.
d) Remote Global (worldwide) – but exclude ads that specify or imply Asia-only, Africa-only, Remote APAC, Remote LATAM, USA, or any region outside the allowed list.
e) If the role mentions a location outside the allowed regions (e.g., USA, Canada), or salary is specified in USD/CAD and no evidence exists the role can be done in allowed regions, mark Not Relevant with explanation.
f) Accept “UK”, “Great Britain”, “London” (etc.) as United Kingdom.
g) If multiple locations are listed and at least one is in the allowed set while others are generic (“remote, anywhere”), treat as Relevant only if the contract allows working fully from the allowed location.

Language Criteria:
a) If the job description requires any language other than English, mark Not Relevant.
b) Only jobs requiring English (or English only) are Relevant.

Steps to follow:
Step 1: Extract Location and remote type (onsite, hybrid, remote) from job description.
Step 2: Check if Location is in allowed list: UK, Ireland (remote only), Remote Europe, Remote EMEA, Remote Global.
Step 3: For Ireland, if remote type is onsite or hybrid, mark Not Relevant.
Step 4: For Europe, if location is a specific country and not remote Europe/EMEA, mark Not Relevant.
Step 5: Otherwise, evaluate role relevance based on job title and other criteria.

Input job title:
{job_title}

Input description:
{full_page_text[:26000]}
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
    full_page_text: str,
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
You will receive two inputs: job title (position name) and description (full page text).

Return ONLY valid JSON with exactly these keys:
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
job_titles
seniorities

Rules for job_category:
- Output exactly one of: "T&P", "NonT&P"
- T&P includes software development, engineering, product, data, IT, UX/UI, QA, DevOps and similar tech/product roles
- NonT&P is everything else that is still relevant

Rules for job_location:
- Match exactly one value from the allowed normalized locations list
- If extracted location is not exactly in the list, choose the closest broader acceptable location
- If not found, return "Unknown" or ""

Rules for remote_preferences:
- Allowed values only: onsite, hybrid, remote
- Normalize all variants
- If multiple apply, output them comma-separated in this exact order: onsite, hybrid, remote
- If not specified, return ""

Rules for remote_days:
- Return only a single number or ""
- Return the highest remote-days number if a range is given
- If office days are stated, calculate remote days as 5 - office days and return the highest
- If fully remote, ambiguous, unclear, or no remote allowed, return ""

Rules for salary:
- salary_min and salary_max must be closest values from allowed salary list
- If only one salary is present, use the same value for both
- salary_currency should be like GBP, USD, EUR, CAD or ""

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
- Keep only the relevant role description
- Exclude company intro, benefits, perks, equal opportunities, apply now, CTA blocks, and similar boilerplate

Rules for job_titles:
- Analyze both job title and description together
- If the job title exactly matches one predefined job title, return only that one
- If ambiguous, return up to 3 most appropriate predefined job titles
- Only return exact strings from the predefined list
- Return as JSON array, ordered most to least appropriate
- If none, return []

Rules for seniorities:
- Allowed only: entry, junior, mid, senior, lead, leadership
- Must be lowercase
- Return as JSON array in this order only: entry, junior, mid, senior, lead, leadership
- If title includes “head of”, “director”, “engineering manager”, or similar leadership terms, use leadership only
- If fewer than 3 suitable values, return only those
- If none, return []

Allowed normalized locations:
{location_list_text}

Allowed salaries:
{salary_list_text}

Predefined job titles:
{job_titles_text}

Input job title:
{job_title}

Input description:
{full_page_text[:26000]}
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
        visa_sponsorship = str(data.get("visa_sponsorship", "") or "").strip()
        job_type = str(data.get("job_type", "") or "").strip()
        job_description = str(data.get("job_description", "") or "").strip() or fallback_job_description

        if visa_sponsorship not in {"yes", "no", ""}:
            visa_sponsorship = fallback_visa
        if job_type not in {"Permanent", "FTC", "Part Time", "Freelance/Contract", ""}:
            job_type = fallback_job_type

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
            "visa_sponsorship": fallback_visa,
            "job_type": fallback_job_type,
            "job_description": fallback_job_description,
            "job_titles": [],
            "seniorities": [],
            "_ai_error": str(e),
        }


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
- You are given skills already found by exact matching.
- Keep those exact-match skills.
- Add any other clearly evidenced missing skills from the description, but ONLY from the allowed list.
- Do not remove correct exact-match skills.
- Return 2 to 10 skills total.

Hard rules:
- role_category is PROVIDED as input. You MUST echo it exactly as given ("T&P" or "NonT&P"). Do not change it.
- skills MUST be an array with 2 to 10 items if evidence exists.
- skills MUST be chosen ONLY from the correct Allowed skills list (exact string match).
- Prefer concrete, clearly evidenced skills from the description.
- exact_skills is the PRIMARY base set. Preserve them unless they are empty.
- NEVER output a skill not present in the allowed list.
- Before finalizing, verify each returned skill appears exactly in the allowed list.
- Keep exact_skills first, then add missing clearly evidenced skills.
- Do not invent skills.

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

description:
{description[:22000]}
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

        return merged[:10]
    except Exception:
        return exact_skills[:10]


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
        if looks_like_js_shell(html, parsed["all_page_text"]):
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
    all_page_text = parsed["all_page_text"]
    title_tag_text = parsed["title_tag_text"]

    raw_title = first_nonempty(structured.get("title"), title_tag_text)
    clean_title = clean_job_title(raw_title)

    fallback_location = normalize_location_rule_based(structured.get("location_raw", ""), all_page_text, allowed_locations)
    fallback_remote_preferences = detect_remote_preferences_rule_based(all_page_text)
    fallback_remote_days = detect_remote_days_rule_based(all_page_text, fallback_remote_preferences)
    fallback_salary_min, fallback_salary_max, fallback_salary_currency = extract_salary_rule_based(
        all_page_text, structured, allowed_salaries
    )
    fallback_visa = detect_visa_rule_based(all_page_text)
    fallback_job_type = detect_job_type_rule_based(all_page_text, structured.get("employment_type_raw", ""))
    fallback_description = clean_job_description(all_page_text)

    fallback_role_relevance = "Relevant" if clean_title else "Not relevant"
    fallback_reason = "Fallback classification based on extracted title and page text."

    relevance = ai_check_relevance(
        job_title=clean_title,
        full_page_text=all_page_text,
        fallback_role_relevance=fallback_role_relevance,
        fallback_reason=fallback_reason,
    )

    result.job_title = clean_title
    result.role_relevance = relevance.get("role_relevance", "") or fallback_role_relevance
    result.role_relevance_reason = relevance.get("role_relevance_reason", "") or fallback_reason
    result.source_method = source_method
    result.status = "ok"

    if result.role_relevance == "Not relevant":
        result.job_category = ""
        result.job_location = ""
        result.remote_preferences = ""
        result.remote_days = ""
        result.salary_min = ""
        result.salary_max = ""
        result.salary_currency = ""
        result.visa_sponsorship = ""
        result.job_type = ""
        result.job_description = ""
        result.notes = " | ".join(notes + ["stopped_after_relevance"])
        return result

    fallback_job_category = "T&P" if re.search(
        r"\b(engineer|developer|software|data|machine learning|ai|product|designer|qa|devops|research)\b",
        clean_title.lower()
    ) else "NonT&P"

    tagged = ai_tag_relevant_job(
        job_title=clean_title,
        full_page_text=all_page_text,
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
    result.visa_sponsorship = tagged.get("visa_sponsorship", "") or fallback_visa
    result.job_type = tagged.get("job_type", "") or fallback_job_type
    result.job_description = clean_job_description(tagged.get("job_description", "") or fallback_description)

    job_titles = tagged.get("job_titles", []) if isinstance(tagged.get("job_titles", []), list) else []
    seniorities = tagged.get("seniorities", []) if isinstance(tagged.get("seniorities", []), list) else []

    result.job_title_tag_1 = job_titles[0] if len(job_titles) > 0 else ""
    result.job_title_tag_2 = job_titles[1] if len(job_titles) > 1 else ""
    result.job_title_tag_3 = job_titles[2] if len(job_titles) > 2 else ""

    result.seniority_1 = seniorities[0] if len(seniorities) > 0 else ""
    result.seniority_2 = seniorities[1] if len(seniorities) > 1 else ""
    result.seniority_3 = seniorities[2] if len(seniorities) > 2 else ""

    skill_list = tp_skills if result.job_category == "T&P" else nontp_skills
    exact_skills = exact_match_skills_in_order(result.job_description or all_page_text, skill_list, limit=10)
    final_skills = ai_enrich_skills(
        role_category=result.job_category,
        description=result.job_description or all_page_text,
        exact_skills=exact_skills,
        allowed_skills=skill_list,
    )

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
        note_parts.append("skills_exact_plus_ai_enrichment")
    else:
        note_parts.append("skills_ai_only")
    result.notes = " | ".join(note_parts)

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
