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
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


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
PLAYWRIGHT_TIMEOUT_MS = 45000
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


@dataclass
class JobResult:
    job_url: str = ""
    job_title: str = ""
    formatted_position_name: str = ""
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


def compact_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


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
        text = re.sub(r"^```(?:json|html)?\s*", "", text)
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


def normalize_quotes(text: str) -> str:
    return (
        (text or "")
        .replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("•", "-")
    )


def canonical_label(text: str) -> str:
    s = normalize_quotes(text or "").lower().strip()
    s = s.replace("&", "and")
    s = s.replace("/", "")
    s = s.replace("\\", "")
    s = s.replace(",", "")
    s = s.replace("-", "")
    s = re.sub(r"\s+", "", s)
    return s


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
                rtype = route.request.resource_type
                if rtype in {"image", "media", "font"}:
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

            try:
                page.wait_for_load_state("networkidle", timeout=12000)
            except PlaywrightTimeoutError:
                pass

            job_like_selectors = [
                "main",
                "article",
                "h1",
                "[data-ui='job-description']",
                "div[class*='job']",
                "div[id*='job']",
                "div[class*='description']",
                "section",
            ]
            for selector in job_like_selectors:
                try:
                    page.locator(selector).first.wait_for(timeout=1500)
                    break
                except Exception:
                    pass

            try:
                page.evaluate("""
                    async () => {
                        await new Promise((resolve) => {
                            let total = 0;
                            const step = 1200;
                            const timer = setInterval(() => {
                                window.scrollBy(0, step);
                                total += step;
                                if (total > 12000) {
                                    clearInterval(timer);
                                    resolve();
                                }
                            }, 120);
                        });
                        await new Promise(r => setTimeout(r, 1200));
                        window.scrollTo(0, 0);
                    }
                """)
            except Exception:
                pass

            page.wait_for_timeout(2500)

            frames_html = []
            try:
                for fr in page.frames:
                    try:
                        if fr == page.main_frame:
                            continue
                        fr_html = fr.content()
                        if fr_html and len(fr_html) > 500:
                            frames_html.append(fr_html)
                    except Exception:
                        pass
            except Exception:
                pass

            html = page.content()
            browser.close()

            if frames_html:
                html = html + "\n" + "\n".join(frames_html)

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
        ]
        for pat in patterns:
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
    for item in embedded[:25]:
        walk_for_strings(item, blob_strings, max_items=3000)
    data["json_blob_text"] = clean_whitespace("\n".join(dedupe_keep_order(blob_strings)))[:50000]

    if not data["title"]:
        title_tag = soup.find("title")
        if title_tag:
            data["title"] = clean_whitespace(title_tag.get_text(" ", strip=True))

    return data


# -------------------------
# DOM / section extraction
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
    "person specification",
    "requirements",
    "about you",
    "you will be responsible for",
    "duties and responsibilities",
    "knowledge, skills, and abilities",
    "experience",
    "responsibilities and duties",
    "what you'll be doing",
    "what you’ll be doing",
    "required qualifications",
    "preferred experience",
    "skills required",
    "must have",
    "ideal candidate",
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
    "equal opportunities for all",
    "more about",
    "reasonably adjustments",
    "how to apply",
    "application process",
    "recruitment process",
    "about 10x",
    "why join us",
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
    if len(x) <= 90 and not re.search(r"[.!?]", x):
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
            "requirements",
            "about you",
            "person specification",
            "required",
            "preferred",
            "skills",
        ]
        if any(k in x for k in keyword_hits):
            return True
    return False


def text_from_node(node) -> str:
    if not node:
        return ""
    return clean_whitespace(node.get_text("\n", strip=True))


def find_primary_root(soup: BeautifulSoup):
    candidates = []
    selectors = [
        "main",
        "article",
        "[role='main']",
        "section",
        "div[id*='job']",
        "div[class*='job']",
        "div[data-ui='job-description']",
        "div[class*='description']",
        "div[class*='posting']",
        "div[class*='content']",
        "div[class*='career']",
        "div[id*='career']",
        "div[class*='opening']",
        "div[id*='opening']",
        "iframe",
    ]
    seen = set()
    for sel in selectors:
        for node in soup.select(sel):
            txt = text_from_node(node)
            if not txt:
                continue
            key = id(node)
            if key in seen:
                continue
            seen.add(key)
            score = len(txt)
            classes = " ".join(node.get("class", [])) if node.get("class") else ""
            node_id = node.get("id", "") or ""
            low_meta = f"{classes} {node_id}".lower()
            if any(k in low_meta for k in ["job", "career", "opening", "posting", "description", "apply", "greenhouse", "lever", "rippling"]):
                score += 1200
            if node.name in {"main", "article"}:
                score += 1500
            candidates.append((score, node))
    if not candidates:
        return soup.body or soup
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def extract_dom_blocks(root) -> List[str]:
    if not root:
        return []

    blocks: List[str] = []
    allowed_tags = {"h1", "h2", "h3", "h4", "strong", "b", "p", "li", "div", "span"}

    nodes = root.find_all(list(allowed_tags), recursive=True)
    for node in nodes:
        txt = text_from_node(node)
        if not txt:
            continue
        txt = normalize_quotes(txt)
        if len(txt) > 2000 and node.name in {"div", "span"}:
            continue
        if len(txt) < 2:
            continue
        blocks.append(txt)

    blocks = dedupe_keep_order(blocks)
    return blocks[:800]


def build_section_based_role_text(blocks: List[str]) -> str:
    if not blocks:
        return ""

    kept: List[str] = []
    in_keep = False
    seen_useful_heading = False

    for block in blocks:
        norm = normalize_heading(block)
        low = block.lower().strip()

        if line_is_noise(block):
            continue

        if norm in ROLE_STOP_HEADINGS:
            if in_keep and seen_useful_heading:
                break
            continue

        if norm in ROLE_KEEP_HEADINGS or looks_like_heading(block):
            if norm in ROLE_KEEP_HEADINGS or any(
                x in norm for x in [
                    "responsib",
                    "qualif",
                    "experience",
                    "about the role",
                    "about the job",
                    "requirements",
                    "about you",
                    "person specification",
                    "minimum qualifications",
                    "preferred qualifications",
                    "what you'll do",
                    "what you’ll do",
                    "required qualifications",
                    "skills",
                    "must have",
                    "ideal candidate",
                ]
            ):
                in_keep = True
                seen_useful_heading = True
                kept.append(block)
                continue

        if in_keep:
            if any(
                phrase in low for phrase in [
                    "privacy policy",
                    "follow us",
                    "recruitment agencies",
                    "all rights reserved",
                    "read more",
                    "equal opportunities for all",
                ]
            ):
                break
            kept.append(block)

    if len(kept) < 10:
        fallback = []
        for block in blocks:
            norm = normalize_heading(block)
            low = block.lower().strip()
            if line_is_noise(block):
                continue
            if norm in ROLE_STOP_HEADINGS:
                continue
            if any(
                phrase in low for phrase in [
                    "privacy policy",
                    "follow us",
                    "recruitment agencies",
                    "all rights reserved",
                    "read more",
                ]
            ):
                continue
            fallback.append(block)
        kept = fallback

    return clean_whitespace("\n".join(kept))


def extract_header_candidates(soup: BeautifulSoup, root, structured: Dict[str, Any], title_tag_text: str) -> List[str]:
    out: List[str] = []

    if structured.get("title"):
        out.append(structured["title"])
    if title_tag_text:
        out.append(title_tag_text)
    if structured.get("company_name"):
        out.append(structured["company_name"])
    if structured.get("location_raw"):
        out.append(structured["location_raw"])

    for tag_name in ["h1", "h2", "h3"]:
        for tag in root.find_all(tag_name):
            txt = text_from_node(tag)
            if txt and 2 <= len(txt) <= 220:
                out.append(txt)

    for node in root.find_all(["p", "div", "span", "li"], recursive=True):
        txt = text_from_node(node)
        if not txt or len(txt) > 260:
            continue
        low = txt.lower()
        if any(
            token in low for token in [
                "location",
                "country",
                "city",
                "based in",
                "based at",
                "remote",
                "hybrid",
                "onsite",
                "on-site",
                "home based",
                "home-based",
                "salary",
                "rate",
                "contract",
                "position role type",
                "work location",
                "right to work",
            ]
        ):
            out.append(txt)

    return dedupe_keep_order(out)[:120]


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
        r"^benefits$",
        r"^what we offer$",
        r"^equal opportunities for all$",
        r"^how to apply$",
        r"^application process$",
    ]

    for line in lines:
        low = line.lower().strip()

        if line_is_noise(line):
            continue

        if any(re.search(p, low, flags=re.I) for p in hard_stop_patterns):
            break

        if any(
            phrase in low for phrase in [
                "all rights reserved",
                "workday, inc.",
                "smartrecruiters",
                "drop us a note to find out more",
                "follow us on linkedin",
                "apply now",
                "instagram",
                "linkedin",
                "recruitment agencies",
            ]
        ):
            continue

        cleaned.append(line)

    out = clean_whitespace("\n".join(dedupe_keep_order(cleaned)))
    return out[:35000]


def merge_role_bodies(*candidates: str) -> str:
    merged_lines: List[str] = []
    for text in candidates:
        for line in split_lines(text):
            merged_lines.append(line)
    merged_lines = dedupe_keep_order(merged_lines)
    return clean_job_description("\n".join(merged_lines))


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
        "about you",
        "you will be responsible for",
        "work location",
        "right to work",
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
        "payroll",
        "people operations",
        "talent acquisition",
        "systems engineer",
        "software engineer",
        "rust",
        "grpc",
        "protocol buffers",
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


def normalize_common_location_aliases(text: str) -> str:
    text = normalize_quotes(text or "")
    replacements = {
        r"\buk\b": "United Kingdom",
        r"\bgb\b": "United Kingdom",
        r"\bgreat britain\b": "United Kingdom",
        r"\bengland\b": "England, United Kingdom",
        r"\bscotland\b": "Scotland, United Kingdom",
        r"\bwales\b": "Wales, United Kingdom",
        r"\bnorthern ireland\b": "Northern Ireland, United Kingdom",
    }
    out = text
    for pat, repl in replacements.items():
        out = re.sub(pat, repl, out, flags=re.I)
    return out


def extract_best_content(html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")
    visible_text = soup_text(soup)
    structured = extract_structured_fields(soup)

    description_raw = structured.get("description_raw") or ""
    structured_description_text = strip_html(description_raw)
    json_blob_text = structured.get("json_blob_text", "")

    title_tag = soup.find("title")
    title_tag_text = clean_whitespace(title_tag.get_text(" ", strip=True)) if title_tag else ""

    root = find_primary_root(soup)
    dom_blocks = extract_dom_blocks(root)
    dom_role_text = build_section_based_role_text(dom_blocks)

    visible_lines = split_lines(visible_text)
    structured_lines = split_lines(structured_description_text)
    json_lines = split_lines(json_blob_text)

    visible_role_body = build_section_based_role_text(visible_lines)
    structured_role_body = build_section_based_role_text(structured_lines) if structured_lines else ""
    json_role_body = build_section_based_role_text(json_lines) if json_lines else ""

    header_candidates = extract_header_candidates(soup, root, structured, title_tag_text)
    header_text = clean_whitespace("\n".join(header_candidates))
    header_text = normalize_common_location_aliases(header_text)

    candidates = [
        ("dom", clean_job_description(dom_role_text)),
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
    if len(role_body_text.strip()) < 800:
        role_body_text = merge_role_bodies(dom_role_text, visible_role_body, structured_role_body, json_role_body)

    role_body_text = normalize_common_location_aliases(role_body_text)

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
    all_page_text = normalize_common_location_aliases(all_page_text)

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


def extract_best_title_candidate(soup: BeautifulSoup, structured_title: str, title_tag_text: str, header_text: str, role_body_text: str) -> str:
    candidates: List[str] = []

    if structured_title:
        candidates.append(structured_title)
    if title_tag_text:
        candidates.append(title_tag_text)

    for tag_name in ["h1", "h2", "h3"]:
        for tag in soup.find_all(tag_name):
            txt = clean_whitespace(tag.get_text(" ", strip=True))
            if txt and 3 <= len(txt) <= 220:
                candidates.append(txt)

    meta_props = [
        ("property", "og:title"),
        ("name", "twitter:title"),
        ("name", "title"),
    ]
    for attr, val in meta_props:
        tag = soup.find("meta", attrs={attr: val})
        if tag and tag.get("content"):
            txt = clean_whitespace(tag.get("content"))
            if txt:
                candidates.append(txt)

    early_lines = split_lines("\n".join([header_text, role_body_text]))[:16]
    for line in early_lines:
        low = line.lower()
        if len(line) > 200:
            continue
        if any(x in low for x in ["location:", "salary", "rate:", "contract", "apply", "posted", "remote in ", "hybrid", "onsite", "work location"]):
            continue
        if re.search(r"\b(job description|overview|responsibilities|qualifications|about the role|description|requirements)\b", low):
            continue
        candidates.append(line)

    cleaned = []
    for cand in candidates:
        ct = clean_job_title(cand)
        low = ct.lower().strip()
        if not ct:
            continue
        if low in {"job openings", "careers", "apply", "overview", "description", "requirements"}:
            continue
        if len(low) < 3:
            continue
        cleaned.append(ct)

    cleaned = dedupe_keep_order(cleaned)
    if cleaned:
        cleaned.sort(key=lambda x: ((" - " in x) or (" | " in x), len(x)))
        return cleaned[0]
    return ""


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
    lines = split_lines(normalize_common_location_aliases(text))

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
        r"\bremote in\b",
        r"\bright to work in the\b",
    ]
    soft_tokens = [" hybrid", " remote", " onsite", " on-site", " home based", " from home", " united kingdom"]

    for idx, line in enumerate(lines[:250]):
        low = f" {normalize_quotes(line).lower()} "
        if any(re.search(p, low, flags=re.I) for p in strong_patterns):
            out.append(line)
            continue
        if idx < 35 and any(tok in low for tok in soft_tokens):
            out.append(line)

    out.extend(lines[:20])
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


def location_specificity_score(location: str) -> int:
    if not location:
        return 0
    parts = [p.strip() for p in location.split(",") if p.strip()]
    score = len(parts) * 100 + len(location)
    low = location.lower()
    if "united kingdom" in low:
        score += 5
    if len(parts) >= 3:
        score += 50
    return score


def normalize_location_rule_based(text: str, allowed_locations: List[str]) -> str:
    if not allowed_locations or not text:
        return ""

    city_lookup = build_location_lookup(allowed_locations)
    candidate_lines = gather_location_lines(text)
    if not candidate_lines:
        return ""

    joined = normalize_common_location_aliases("\n".join(candidate_lines))
    joined_low = normalize_quotes(joined).lower()

    matches: List[Tuple[int, int, str]] = []

    for loc in allowed_locations:
        loc_norm = normalize_common_location_aliases(loc)
        loc_low = normalize_quotes(loc_norm).lower()
        m = re.search(rf"(?<![a-z]){re.escape(loc_low)}(?![a-z])", joined_low)
        if m:
            matches.append((m.start(), -location_specificity_score(loc_norm), loc))

    for city, full in city_lookup.items():
        m = re.search(rf"\b{re.escape(city)}\b", joined_low)
        if m:
            matches.append((m.start(), -location_specificity_score(full), full))

    if re.search(r"\b(united kingdom|great britain)\b", joined_low):
        for loc in allowed_locations:
            if loc.lower() in {"united kingdom", "england, united kingdom"}:
                matches.append((50, -location_specificity_score(loc), loc))

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
                matches.append((999999, -location_specificity_score(loc), loc))

    if not matches:
        return ""

    matches.sort(key=lambda x: (x[0], x[1]))
    return matches[0][2]


def detect_remote_preferences_rule_based(text: str) -> str:
    lines = gather_location_lines(text)
    low = normalize_quotes("\n".join(lines)).lower()
    found = []

    if re.search(r"\bhome based\b|\bhome-based\b|\buk remote\b|\bfully remote\b|\bremote\b|\bremote enabled\b|\bremote working\b", low):
        found.append("remote")
    if re.search(r"\bhybrid\b", low):
        found.append("hybrid")
    if re.search(r"\bonsite\b|\bon-site\b|\bon site\b|\bin office\b|\bin-office\b", low):
        found.append("onsite")

    if re.search(r"\bwork location\b.*\bremote\b", low) or re.search(r"\bremote working within the united kingdom\b", low):
        return "remote"

    ordered = [x for x in ["onsite", "hybrid", "remote"] if x in found]
    return ", ".join(ordered)


def detect_remote_days_rule_based(text: str, remote_prefs: str) -> str:
    low = normalize_quotes("\n".join(gather_location_lines(text) + split_lines(text)[:120])).lower()

    if "hybrid" not in remote_prefs and "remote" not in remote_prefs:
        return ""

    patterns = [
        r"\b(?:approx\.?\s*)?(\d)\s*[-to]{1,3}\s*(\d)\s+days?\s+(?:per week\s+)?(?:in the office|from the office|office|on site|onsite|on-site)\b",
        r"\b(?:approx\.?\s*)?(\d)\s+days?\s+(?:per week\s+)?(?:in the office|from the office|office|on site|onsite|on-site)\b",
        r"\b(?:approx\.?\s*)?(\d)\s*[-to]{1,3}\s*(\d)\s+days?\s+(?:per week\s+)?(?:from home|remote|wfh)\b",
        r"\b(?:approx\.?\s*)?(\d)\s+days?\s+(?:per week\s+)?(?:from home|remote|wfh)\b",
        r"\b(?:work|working)\s+(\d)\s+days?\s+(?:from home|remote|wfh)\b",
        r"\boffice attendance requirement of approx\.?\s*(\d)\s*[-to]{1,3}\s*(\d)\s+days?\s+per week\b",
    ]

    for pat in patterns:
        m = re.search(pat, low)
        if not m:
            continue
        groups = [g for g in m.groups() if g is not None]
        if "office" in pat or "attendance" in pat or "on site" in pat or "onsite" in pat:
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
    candidate_lines = [line for line in lines[:160] if line_has_compensation_anchor(line)]

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
        r"applicants must have the right to work in the united kingdom",
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

    if any(x in title_low for x in ["senior", "sr ", "sr."]):
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
        r"\brust\b",
        r"\bkubernetes\b",
    ]
    return any(re.search(p, text) for p in tp_patterns)


def is_relevant_by_rules(job_title: str, role_text: str, header_text: str = "") -> bool:
    text = f"{job_title}\n{header_text}\n{role_text}".lower()

    allowed_patterns = [
        r"\btalent acquisition\b",
        r"\brecruiter\b",
        r"\brecruitment\b",
        r"\bhuman resources\b",
        r"\bhead of hr\b",
        r"\bhr manager\b",
        r"\bpeople ops\b",
        r"\bpeople operations\b",
        r"\bpeople partner\b",
        r"\baccount manager\b",
        r"\baccount executive\b",
        r"\baccount director\b",
        r"\bcustomer success\b",
        r"\bcsm\b",
        r"\brenewals\b",
        r"\bclient services\b",
        r"\bbusiness analyst\b",
        r"\bbusiness operations\b",
        r"\boperations\b",
        r"\bchange manager\b",
        r"\btransformation\b",
        r"\bpmo\b",
        r"\bprogramme manager\b",
        r"\bprogram manager\b",
        r"\bproject manager\b",
        r"\brisk\b",
        r"\bcompliance\b",
        r"\blegal\b",
        r"\bfinance\b",
        r"\baccounting\b",
        r"\bfp&a\b",
        r"\brevops\b",
        r"\bsales operations\b",
        r"\bsdr\b",
        r"\bbdr\b",
        r"\bmarketing\b",
        r"\bseo\b",
        r"\bpr\b",
        r"\bcommunications\b",
        r"\bengineer\b",
        r"\bdeveloper\b",
        r"\barchitect\b",
        r"\bdevops\b",
        r"\bqa\b",
        r"\bproduct\b",
        r"\bdesigner\b",
        r"\bux\b",
        r"\bui\b",
        r"\bdata\b",
        r"\bmachine learning\b",
        r"\bai\b",
        r"\bsecurity\b",
        r"\bcloud\b",
        r"\bnetwork\b",
        r"\binfrastructure\b",
        r"\bsystems\b",
        r"\bsupport engineer\b",
        r"\bsystem administrator\b",
        r"\bsystem engineer\b",
        r"\bsolutions engineer\b",
        r"\bit support\b",
        r"\b2nd line\b",
        r"\bsecond line\b",
    ]
    excluded_patterns = [
        r"\bteacher\b",
        r"\bnurse\b",
        r"\bwaiter\b",
        r"\bchef\b",
        r"\bconstruction\b",
        r"\bcivil engineer\b",
        r"\belectrician\b",
        r"\bmechanical engineer\b",
        r"\bmanufacturing\b",
        r"\bmaritime\b",
        r"\bmicrobiology\b",
        r"\binjection molding\b",
        r"\bwarehouse\b",
        r"\bdriver\b",
        r"\bcleaner\b",
    ]

    if any(re.search(p, text) for p in allowed_patterns):
        return True
    if any(re.search(p, text) for p in excluded_patterns):
        return False
    return False


# -------------------------
# Normalizers / AI helpers
# -------------------------

def normalize_job_title_from_list(value: str, allowed_job_titles: List[str]) -> str:
    value_clean = clean_whitespace(value)
    if not value_clean:
        return ""

    for jt in allowed_job_titles:
        if value_clean.lower() == jt.lower():
            return jt

    canon_value = canonical_label(value_clean)
    for jt in allowed_job_titles:
        if canon_value == canonical_label(jt):
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


def ai_format_position_name(raw_position_name: str) -> str:
    if not raw_position_name:
        return ""
    if not client:
        return clean_job_title(raw_position_name)

    prompt = f"""
You are given a raw position name string that may contain extra information such as company names, locations, contract types, separators (e.g., "·", "-", "|"), and other trailing details.

Your task is to extract and output only the clean job title without any trailing location names, company names, contract info, or extra text.

Rules:
1. Identify and keep only the core job title at the start of the string, stopping at the first indication of any location, company, contract type, or separator.
2. Common separators that may indicate trailing info include: "·", "-", "|", commas, or words like "Full-Time", "Contract", "Ltd", "Limited", "Inc", "Remote", "Location", "Office".
3. Remove all trailing words or phrases related to location, contract types, and company names.
4. Preserve the exact wording and order of the job title portion only; do not rephrase or add words.
5. If the job title contains commas or hyphens inside the core title, preserve those, but stop extracting once trailing info starts.

Output only the cleaned job title. No extra text.

Input:
{raw_position_name}
""".strip()

    try:
        response = client.responses.create(model=OPENAI_MODEL, input=prompt)
        cleaned = clean_whitespace(response.output_text)
        return cleaned or clean_job_title(raw_position_name)
    except Exception:
        return clean_job_title(raw_position_name)


def ai_map_job_titles_only(position_name: str, description: str, allowed_job_titles: List[str]) -> List[str]:
    if not client:
        return []

    job_titles_text = ", ".join(allowed_job_titles[:3000])

    prompt = f"""
You will receive two inputs: position name and description.

Task:
Choose up to 3 best matching job titles from the predefined list.

Rules:
- Analyze both the position name and description together.
- Use ONLY job titles from the predefined list.
- If the position name exactly matches one predefined job title, return only that single title.
- If the position name is unclear or broader than the predefined list, choose up to the top 3 most appropriate job titles from the predefined list.
- Return a JSON object with one key only:
  "job_titles": ["...", "..."]
- job_titles must contain exact strings from the predefined list only.
- Order from most appropriate to least appropriate.
- If no suitable match exists, return an empty array.
- Do NOT leave job_titles empty if there is a clear best-fit title in the predefined list.

Predefined job titles:
{job_titles_text}

Position name:
{position_name}

Description:
{description[:18000]}
""".strip()

    try:
        response = client.responses.create(model=OPENAI_MODEL, input=prompt)
        data = safe_json_loads(response.output_text)
        raw_titles = data.get("job_titles", [])
        if not isinstance(raw_titles, list):
            return []

        out = []
        for t in raw_titles:
            exact = normalize_job_title_from_list(str(t), allowed_job_titles)
            if exact and exact not in out:
                out.append(exact)
        return out[:3]
    except Exception:
        return []


def ai_map_seniority_only(position_name: str, description: str) -> List[str]:
    if not client:
        return []

    prompt = f"""
You will receive two inputs: position name and description.

Determine seniority using only this allowed list:
entry, junior, mid, senior, lead, leadership

Rules:
1. First analyze the position name.
2. If title includes leadership indicators such as "head of", "director", "engineering manager", "vp", "chief", return leadership only.
3. If the title clearly contains seniority like junior, senior, lead, return the appropriate value(s).
4. If ambiguous, analyze both title and description together and return up to 3 most appropriate seniority levels.
5. Output must be JSON with one key only:
   "seniorities": ["...", "..."]
6. Use only these exact lowercase values:
   entry, junior, mid, senior, lead, leadership
7. Keep order exactly:
   entry, junior, mid, senior, lead, leadership
8. If experience is a range, include all applicable seniorities across the range:
   - 0-1 years -> entry, junior
   - 2 years -> junior, mid
   - 3-5 years -> senior, lead
9. If managerial/team-management responsibilities are clearly present, include lead.
10. If nothing suitable is identified, return an empty array.

Position name:
{position_name}

Description:
{description[:18000]}
""".strip()

    try:
        response = client.responses.create(model=OPENAI_MODEL, input=prompt)
        data = safe_json_loads(response.output_text)
        raw = data.get("seniorities", [])
        if not isinstance(raw, list):
            return []
        return normalize_seniority_list(raw)
    except Exception:
        return []


def ai_clean_job_description_html(job_description_text: str) -> str:
    if not job_description_text:
        return ""
    if not client:
        return clean_job_description(job_description_text)

    prompt = f"""
From the job description input, extract only relevant job content.

Include
1. Purpose and responsibilities (About the Role, Responsibilities, Key Duties).
2. Candidate requirements, qualifications, skills, experience (About You, Requirements, Ideally You Will Have).
3. Any text that directly explains what the job involves or what the candidate should bring.

Exclude
1. Company background
2. Benefits and perks
3. Disclaimers or application instructions
4. Marketing/promotional text
5. Unrelated content

Formatting Rules
1. HTML only:
<b> for section titles
<ul><li> for bullet lists
<p> for plain paragraphs
<em> only if in source
No markdown, no backticks, no wrappers

2. Section titles:
a) Keep original headers word-for-word
b) Preserve casing, but if ALL CAPS convert to Title Case
c) If header is “About the Role/About this Role/The Role”, remove the header and keep only the content

3. Bullets:
a) Use <ul><li> only if source has bullets
b) Do not convert plain paragraphs into bullets
c) Keep each bullet in one <li>, no merging/splitting
d) Remove empty <li></li>

4. Lists & nesting:
a) Sub-bullets -> nested <ul> inside <li>
b) If unclear, flatten under nearest header
c) Do not invent new headers

5. Preservation & cleanup:
a) Keep sentences word-for-word, no rephrasing
b) Remove stray dots and extra spaces
c) Intro text before first header -> <p>

6. Header fallback:
a) If bullets appear without header:
- Duties -> <b>Responsibilities</b>
- Qualifications/skills -> <b>Requirements</b> or <b>Skills & Experience</b> if closer
b) If nothing remains -> output empty string

7. Output only valid HTML. No explanations.

Input job description:
{job_description_text[:26000]}
""".strip()

    try:
        response = client.responses.create(model=OPENAI_MODEL, input=prompt)
        html = clean_whitespace(response.output_text)

        if html.startswith("```"):
            html = re.sub(r"^```(?:html)?\s*", "", html)
            html = re.sub(r"\s*```$", "", html)

        return html
    except Exception:
        return clean_job_description(job_description_text)


# -------------------------
# AI prompts
# -------------------------

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
You will receive two inputs: position name and job description.

Return ONLY valid JSON with exactly these keys:
role_relevance
role_relevance_reason

Rules:
- role_relevance must be exactly "Relevant" or "Not relevant"
- role_relevance_reason must be concise and specific

Relevant roles match any in this list or close synonyms/specializations:

Account Director, Account Executive, AI Engineer, Automation Engineer, Back End, BI Developer, Big Data Engineer, Brand Marketing, Business Analyst, Business Development Manager, Business Operations, CDO, CFO, Chief of Staff, CIO, CLO, Cloud Engineer, CMO, Computer Vision Engineer, Content Marketing, COO, Copywriting, CPO, CRM Developer, CRM Manager, CRO, CSM / Account Manager, CSO, CTO, Customer Operations, Customer Service Rep, Customer Support, Data Architect, Data Engineer, Data Scientist, Data / Insight Analyst, Database Engineer, Deep Learning Engineer, Demand / Lead Generation, Developer in Test, DevOps Engineer, Digital Marketing, Embedded Developer, Engineering Manager, Events & Community, Executive Assistant, Finance / Accounting, Founder, FP&A, Front End, Full Stack, Games Designer, Games Developer, Generalist Marketing, Graphic Designer, Graphics Developer, Growth Marketing, Head of Customer, Head of Data, Head of Design, Head of Engineering, Head of Finance, Head of HR, Head of Infrastructure, Head of Marketing, Head of Operations, Head of Product, Head of QA, Head of Sales, Human Resources, Implementation Manager, Integration Developer, Legal, Machine Learning Engineer, Marketing Analyst, Mobile Developer, Network Engineer, Operations, Partnerships, Penetration Tester, People Ops, Performance Marketing, PR / Communications, Product Manager, Product Marketing, Product Owner, Project Manager, QA Automation Tester, QA Manual Tester, Quality Assurance, Quantitative Developer, Renewals Manager, Research Engineer, RevOps, Risk & Compliance, Sales Engineer, Sales Operations, Scrum Master, SDR / BDR, Security, Security Engineer, SEO Marketing, Site Reliability Engineer, Social Media Marketing, Solutions Engineer, Support Engineer, System Administrator, System Engineer, Talent Acquisition, Technical Architect, Technical Director, Technical Writer, Testing Manager, UI Designer, UI/UX Designer, UX Designer, UX Researcher, Videography, VP of Engineering.

Important allow rules:
- Talent Acquisition roles ARE relevant.
- Human Resources roles ARE relevant.
- People Ops roles ARE relevant.
- Recruitment roles are relevant when they are internal/corporate talent acquisition, recruiting, people, HR, or employer-branding functions.
- IT support / infrastructure / support engineering roles ARE relevant and should not be excluded just because they are support-oriented or customer-facing.
- Business/program/change/transformation/PMO/operations roles can be relevant under Business Operations / Project Manager / Operations families.

Reject only when the role is clearly outside allowed business/tech functions, such as:
teacher, nurse, waiter, chef, construction worker, civil engineer, electrician, mechanical engineer in manufacturing/plant context, manufacturing operator, maritime crew, microbiology lab role, beauty retail staff, injection molding technician, warehouse operative, driver, cleaner.

Location rules:
- Relevant only if the working location is allowed:
  a) United Kingdom: onsite, hybrid, or remote allowed
  b) Ireland: only remote allowed
  c) Europe: only explicitly Remote Europe or Remote EMEA allowed
  d) Remote Global allowed unless explicitly limited to excluded regions
  e) If the role clearly points to USA/Canada/other non-allowed region with no evidence of UK/allowed-region work, mark Not relevant
- Accept UK cities/regions as UK
- If multiple locations are listed, use the stated working arrangement and whether at least one valid allowed working option clearly exists

Language rules:
- If the role clearly requires a non-English language, mark Not relevant
- English-only roles are fine

Decision priority:
1. First determine whether the role family is in the allowed list or a close synonym
2. Then determine whether the location/remote setup is allowed
3. Only then decide final relevance

Be careful:
- Do NOT mark a role Not relevant simply because it is recruitment/HR if it is clearly Talent Acquisition / Human Resources / People Ops
- Do NOT mark a role Not relevant simply because it is support/infrastructure if it is clearly IT / systems / cloud / network / support engineering
- Do NOT mark a role Not relevant as “non-informative” if the extracted text clearly contains an actual engineering/business role title or responsibilities
- Use the actual job title and role duties, not generic company overview text

Input position name:
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
1. position name
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
- If multiple valid locations are listed, choose the most specific clear normalized location from the allowed list.
- If the page says “Remote working within the UK”, “UK remote”, or similar, return the correct UK location from the allowed list, not blank.
- Only return "Unknown" if there is truly no clear location evidence.

Rules for remote_preferences:
- Allowed values only: onsite, hybrid, remote
- Use role-specific/header evidence first.
- home based counts as remote.
- daily field travel does NOT mean onsite only.
- Do not let generic company-wide smart-working text override a clearer role-specific line.
- If the role-specific line says remote in UK / remote working within the UK, return remote only.

Rules for remote_days:
- Return only a single number or ""
- Only return a number when explicit days are stated.
- Never infer a number from the word "hybrid" alone.
- Do not infer from company-wide smart-working statements.

Rules for salary:
- Only tag salary when compensation/rate is explicitly stated for this role.
- salary_period must be one of: year, day, hour, month, or ""
- If a daily rate is stated (e.g. £500-£550 p/d), preserve it as min/max with salary_period=day.
- Do not invent salary from unrelated numbers, dates, standards, reference ids, years, counts, targets, legislation references, percentages, revenues, or employee counts.
- Preserve the actual explicit number. Do not round to predefined salary list.

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
- Keep the main role content only.
- Preserve useful sections such as:
  responsibilities, required, essential, desirable, qualifications, what to bring, key relationships, success measures, minimum qualifications, preferred qualifications, summary, activities, landscape, about you, requirements.
- Remove company marketing, benefits, footer, privacy, legal, ATS boilerplate, follow-us, equal-opportunity sections.
- Do not shorten to one paragraph if multiple role sections are clearly present.

Rules for job_titles:
- Analyze both the position name and role description together.
- Use ONLY job titles from the predefined list.
- Return job_titles as a JSON array of up to 3 exact strings from the predefined list.
- If the position name exactly matches one predefined job title, return only that single title.
- If the position name is unclear, broader than the predefined list, or does not exactly match the predefined list, choose up to the top 3 most appropriate job titles from the predefined list based on both title and description.
- Order job_titles from most appropriate to least appropriate.
- If fewer than 3 suitable titles exist, return only those.
- If no suitable title exists, return an empty array.
- Prefer functional fit over literal wording.
- Never invent a title outside the predefined list.
- Do NOT leave job_titles empty if there is a clear best-fit title in the predefined list.
- Examples:
  - "People Operations Manager" -> ["People Ops"]
  - "Software Development Engineer" -> choose the closest engineering title(s) supported by the description, such as ["Back End"] or ["Full Stack"]
  - "National Account Manager - Wholesale" -> choose the best account/client-facing title from the predefined list based on responsibilities
  - "Sr. Engineer, Systems" -> ["System Engineer"] or ["System Administrator"] depending on the description
  - "DTN Software engineer" -> choose the closest software/infrastructure engineering title(s) from the predefined list supported by the description
- For account-management style roles, prefer "CSM/Account Manager" when the function is relationship/account ownership rather than net-new sales.
- For customer-facing technical roles, "Solutions Engineer" can coexist with another technical title when clearly supported.
- For AI Solution Architect, prefer "Technical Architect" / "AI Engineer" style tags, not Head of Engineering.
- For People/HR roles, map to "People Ops", "Human Resources", or "Talent Acquisition" based on the real function.
- For data/analytics roles, map to the closest valid title such as "Data/Insight Analyst", "Data Scientist", "Data Engineer", etc., based on duties rather than title wording alone.

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

Allowed salaries reference list:
{salary_list_text}

Predefined job titles:
{job_titles_text}

Input position name:
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
    plain = strip_html(role_body_text)
    lines = split_lines(plain)
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

    raw_title = extract_best_title_candidate(
        soup=parsed["soup"],
        structured_title=structured.get("title", ""),
        title_tag_text=title_tag_text,
        header_text=header_text,
        role_body_text=role_body_text,
    )
    clean_title = clean_job_title(raw_title)
    formatted_position_name = ai_format_position_name(raw_title or clean_title or title_tag_text)

    evidence_text = "\n".join([header_text, role_body_text, structured.get("location_raw", "")])
    evidence_text = normalize_common_location_aliases(evidence_text)

    fallback_location = normalize_location_rule_based(evidence_text, allowed_locations)
    fallback_remote_preferences = detect_remote_preferences_rule_based(evidence_text)
    fallback_remote_days = detect_remote_days_rule_based(evidence_text, fallback_remote_preferences)
    fallback_salary_min, fallback_salary_max, fallback_salary_currency, fallback_salary_period = parse_explicit_salary(
        evidence_text,
        allowed_salaries,
    )
    fallback_visa = detect_visa_rule_based(role_context_text)
    fallback_job_type = detect_job_type_rule_based(evidence_text, structured.get("employment_type_raw", ""))
    fallback_description = clean_job_description(role_body_text)

    substantial_role_text = len(clean_job_description(role_body_text)) >= 700
    fallback_role_relevance = "Relevant" if ((clean_title and is_relevant_by_rules(clean_title, role_body_text, header_text)) or substantial_role_text) else "Not relevant"
    fallback_reason = "Fallback classification based on extracted title and role text."

    if not clean_title and role_body_text:
        if is_relevant_by_rules(role_body_text[:300], role_body_text, header_text):
            fallback_role_relevance = "Relevant"

    relevance = ai_check_relevance(
        job_title=formatted_position_name or clean_title,
        role_context_text=role_context_text,
        fallback_role_relevance=fallback_role_relevance,
        fallback_reason=fallback_reason,
    )

    result.job_title = clean_title
    result.formatted_position_name = formatted_position_name or clean_title
    result.role_relevance = relevance.get("role_relevance", "") or fallback_role_relevance
    result.role_relevance_reason = relevance.get("role_relevance_reason", "") or fallback_reason
    result.source_method = source_method
    result.status = "ok"

    if result.role_relevance == "Not relevant" and substantial_role_text and (
        is_relevant_by_rules(clean_title, role_body_text, header_text) or is_relevant_by_rules(role_body_text[:300], role_body_text, header_text)
    ):
        result.role_relevance = "Relevant"
        result.role_relevance_reason = "Rule-based override: extracted title and role description indicate an allowed relevant role."

    if result.role_relevance == "Not relevant":
        result.notes = " | ".join(notes + ["stopped_after_relevance"])
        return result

    fallback_job_category = "T&P" if is_tp_by_rules(clean_title, role_body_text) else "NonT&P"

    tagged = ai_tag_relevant_job(
        job_title=formatted_position_name or clean_title,
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
    if ai_location and location_value_has_evidence(ai_location, evidence_text, allowed_locations):
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

    final_html_description = ai_clean_job_description_html(role_body_text)
    if not final_html_description:
        final_html_description = clean_job_description(tagged.get("job_description", "") or fallback_description)
    result.job_description = final_html_description

    hard_evidence_text = "\n".join([header_text, strip_html(result.job_description), structured.get("location_raw", "")])
    hard_evidence_text = normalize_common_location_aliases(hard_evidence_text)

    result.job_location = normalize_location_rule_based(hard_evidence_text, allowed_locations) or result.job_location
    result.remote_preferences = detect_remote_preferences_rule_based(hard_evidence_text) or result.remote_preferences
    result.remote_days = detect_remote_days_rule_based(hard_evidence_text, result.remote_preferences)

    explicit_salary_check = parse_explicit_salary(hard_evidence_text, allowed_salaries)
    if explicit_salary_check == ("", "", "", ""):
        result.salary_min = ""
        result.salary_max = ""
        result.salary_currency = ""
        result.salary_period = ""
    else:
        result.salary_min, result.salary_max, result.salary_currency, result.salary_period = explicit_salary_check

    job_titles = tagged.get("job_titles", []) if isinstance(tagged.get("job_titles", []), list) else []
    if not job_titles:
        job_titles = ai_map_job_titles_only(
            position_name=formatted_position_name or clean_title,
            description=strip_html(result.job_description) or role_body_text,
            allowed_job_titles=allowed_job_titles,
        )

    result.job_title_tag_1 = job_titles[0] if len(job_titles) > 0 else ""
    result.job_title_tag_2 = job_titles[1] if len(job_titles) > 1 else ""
    result.job_title_tag_3 = job_titles[2] if len(job_titles) > 2 else ""

    seniorities = tagged.get("seniorities", []) if isinstance(tagged.get("seniorities", []), list) else []
    if not seniorities:
        seniorities = ai_map_seniority_only(
            position_name=formatted_position_name or clean_title,
            description=strip_html(result.job_description) or role_body_text,
        )
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

    note_parts = notes + ["ran_relevant_tagging", "formatted_position_name_added", "html_clean_description_added"]
    if len(job_titles) == 0:
        note_parts.append("job_titles_empty")
    else:
        note_parts.append("job_titles_present")
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
