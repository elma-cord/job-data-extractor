import json
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/145.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-GB,en;q=0.9,en-US;q=0.8",
}

REQUEST_TIMEOUT = 20
PLAYWRIGHT_TIMEOUT_MS = 18000
PLAYWRIGHT_NETWORKIDLE_MS = 4500
PLAYWRIGHT_POST_LOAD_WAIT_MS = 900

# Domains where raw HTML is usually enough and Playwright is often wasteful
NEVER_RENDER_DOMAINS = {
    "boards.greenhouse.io",
    "job-boards.greenhouse.io",
    "jobs.ashbyhq.com",
    "jobs.lever.co",
    "apply.workable.com",
    "smartrecruiters.com",
}

# Domains where JS rendering is more often needed
LIKELY_RENDER_DOMAINS = {
    "myworkdayjobs.com",
    "workday.com",
    "oraclecloud.com",
    "successfactors.com",
    "icims.com",
    "ultipro.com",
    "dayforcehcm.com",
    "recruitee.com",
    "pinpointhq.com",
}

STRONG_JOB_SIGNALS = [
    "responsibilities",
    "requirements",
    "qualifications",
    "job description",
    "about the role",
    "about you",
    "minimum qualifications",
    "preferred qualifications",
    "what you'll do",
    "what you’ll do",
    "experience",
    "apply now",
    "apply for this job",
    "job title",
    "location:",
]


# -------------------------
# Reusable HTTP session
# -------------------------

_session = requests.Session()
_retry = Retry(
    total=2,
    connect=2,
    read=2,
    status=2,
    status_forcelist=(429, 500, 502, 503, 504),
    backoff_factor=0.4,
    allowed_methods=frozenset(["GET", "HEAD"]),
)
_adapter = HTTPAdapter(max_retries=_retry, pool_connections=100, pool_maxsize=100)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)
_session.headers.update(HEADERS)


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


def normalize_heading(line: str) -> str:
    low = normalize_quotes(line).lower().strip(" :-•\t")
    low = re.sub(r"\s+", " ", low)
    return low


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


def get_domain(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""


def domain_matches(domain: str, candidates: set[str]) -> bool:
    return any(domain == d or domain.endswith("." + d) for d in candidates)


# -------------------------
# Fetching
# -------------------------

def fetch_html(url: str) -> Tuple[Optional[str], str]:
    try:
        resp = _session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        status = f"http_{resp.status_code}"
        if resp.ok and resp.text:
            return resp.text, status
        return None, status
    except Exception as e:
        return None, f"request_error: {e}"


def fetch_with_playwright(url: str) -> Tuple[Optional[str], str]:
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--no-sandbox",
                    "--disable-blink-features=AutomationControlled",
                ],
            )
            context = browser.new_context(
                user_agent=HEADERS["User-Agent"],
                locale="en-GB",
                java_script_enabled=True,
                ignore_https_errors=True,
            )
            page = context.new_page()

            def route_handler(route):
                rtype = route.request.resource_type
                if rtype in {"image", "media", "font", "stylesheet"}:
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
                page.wait_for_load_state("networkidle", timeout=PLAYWRIGHT_NETWORKIDLE_MS)
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
                    page.locator(selector).first.wait_for(timeout=600)
                    break
                except Exception:
                    pass

            try:
                page.evaluate(
                    """
                    async () => {
                        const body = document.body;
                        if (!body) return;
                        const totalHeight = Math.max(
                            body.scrollHeight,
                            document.documentElement ? document.documentElement.scrollHeight : 0
                        );
                        if (totalHeight < 2200) return;

                        window.scrollBy(0, 1200);
                        await new Promise(r => setTimeout(r, 220));
                        window.scrollBy(0, 1200);
                        await new Promise(r => setTimeout(r, 220));
                        window.scrollTo(0, 0);
                    }
                    """
                )
            except Exception:
                pass

            page.wait_for_timeout(PLAYWRIGHT_POST_LOAD_WAIT_MS)

            frames_html = []
            try:
                for fr in page.frames:
                    try:
                        if fr == page.main_frame:
                            continue
                        fr_html = fr.content()
                        if fr_html and len(fr_html) > 1200:
                            frames_html.append(fr_html)
                    except Exception:
                        pass
            except Exception:
                pass

            html = page.content()
            context.close()
            browser.close()

            if frames_html:
                html = html + "\n" + "\n".join(frames_html)

            return html, "playwright"
    except Exception as e:
        return None, f"playwright_error: {e}"


def looks_like_js_shell(html: str, text: str) -> bool:
    low_text = normalize_quotes(text or "").lower()
    compact_text = re.sub(r"\s+", " ", low_text).strip()

    shell_signals = [
        "enable javascript",
        "javascript is required",
        "please enable javascript",
        "loading...",
        "please wait while we load",
        "application error",
        "access denied",
        "verify you are human",
        "captcha",
        "checking your browser",
        "cf-challenge",
        "cloudflare",
    ]
    if any(sig in compact_text for sig in shell_signals):
        return True

    strong_job_signal_hits = sum(1 for sig in STRONG_JOB_SIGNALS if sig in compact_text)
    if strong_job_signal_hits >= 2 and len(compact_text) >= 220:
        return False

    if len(compact_text) < 180:
        return True

    body_len = len(re.sub(r"\s+", " ", html or ""))
    text_len = len(re.sub(r"\s+", " ", text or ""))
    if body_len > 0 and text_len / body_len < 0.02 and strong_job_signal_hits == 0:
        return True

    return False


def should_try_playwright(url: str, html: str, parsed_role_context_text: str) -> bool:
    domain = get_domain(url)

    if domain_matches(domain, NEVER_RENDER_DOMAINS):
        return False

    if domain_matches(domain, LIKELY_RENDER_DOMAINS):
        return True

    if looks_like_js_shell(html, parsed_role_context_text):
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
                        piece = ", ".join(
                            [
                                p
                                for p in [
                                    first_nonempty(addr.get("addressLocality")),
                                    first_nonempty(addr.get("addressRegion")),
                                    first_nonempty(addr.get("addressCountry")),
                                ]
                                if p
                            ]
                        )
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
    "role and responsibilities",
    "main responsibilities and accountabilities",
    "other responsibilities",
    "skills and competencies",
    "experience you'll bring",
    "experience you’ll bring",
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
    "inclusion & diversity",
    "application",
    "what's in it for you",
    "what’s in it for you",
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


def looks_like_heading(line: str) -> bool:
    x = normalize_heading(line)
    if not x:
        return False
    if x in ROLE_KEEP_HEADINGS or x in ROLE_STOP_HEADINGS:
        return True

    raw = normalize_quotes(line).strip()
    if raw.endswith(":") and len(raw) <= 120:
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
            "activities",
            "coordination",
            "administration",
            "delivery",
            "focus",
            "responsibilities",
        ]
        if any(k in x for k in keyword_hits):
            return True

    if len(raw) <= 80 and raw == raw.title() and not re.search(r"[.!?]", raw):
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
    return blocks[:900]


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
                    "role and responsibilities",
                    "main responsibilities and accountabilities",
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
                "office attendance",
                "office based",
                "internal job title",
                "function:",
                "people leader:",
            ]
        ):
            out.append(txt)

    return dedupe_keep_order(out)[:180]


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
        r"^inclusion & diversity$",
        r"^application:$",
        r"^about us:$",
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
        "role and responsibilities",
        "main responsibilities and accountabilities",
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
        "office attendance requirement",
        "crm analytics",
        "campaign performance",
        "loyalty scheme",
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
    all_page_text = clean_whitespace(
        "\n".join(
            [
                title_tag_text,
                structured.get("title", ""),
                structured.get("company_name", ""),
                structured.get("location_raw", ""),
                structured_description_text,
                json_blob_text,
                visible_text,
            ]
        )
    )
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
