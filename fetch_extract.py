import json
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/145.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-GB,en;q=0.9,en-US;q=0.8",
}

# Faster timeout for fallback-only fetching
REQUEST_TIMEOUT = 8

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
    "remote",
    "hybrid",
    "onsite",
    "on-site",
]


# -------------------------
# Reusable HTTP session
# -------------------------

_session = requests.Session()
_retry = Retry(
    total=1,
    connect=1,
    read=1,
    status=1,
    status_forcelist=(429, 500, 502, 503, 504),
    backoff_factor=0.2,
    allowed_methods=frozenset(["GET", "HEAD"]),
)
_adapter = HTTPAdapter(max_retries=_retry, pool_connections=50, pool_maxsize=50)
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


# Compatibility stub only. Not used in the fast workflow.
def fetch_with_playwright(url: str) -> Tuple[Optional[str], str]:
    return None, "playwright_disabled"


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
    if strong_job_signal_hits >= 2 and len(compact_text) >= 180:
        return False

    if len(compact_text) < 140:
        return True

    body_len = len(re.sub(r"\s+", " ", html or ""))
    text_len = len(re.sub(r"\s+", " ", text or ""))
    if body_len > 0 and text_len / body_len < 0.02 and strong_job_signal_hits == 0:
        return True

    return False


# Compatibility stub only. Fast workflow never auto-renders.
def should_try_playwright(url: str, html: str, parsed_role_context_text: str) -> bool:
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


def walk_for_strings(node: Any, out: List[str], max_items: int = 1500) -> None:
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

        emp_type = jp.get("employmentType")
        if isinstance(emp_type, list):
            data["employment_type_raw"] = ", ".join(str(x) for x in emp_type)
        else:
            data["employment_type_raw"] = first_nonempty(emp_type)

    embedded = extract_embedded_json_candidates(soup)
    blob_strings: List[str] = []
    for item in embedded[:10]:
        walk_for_strings(item, blob_strings, max_items=1500)
    data["json_blob_text"] = clean_whitespace("\n".join(dedupe_keep_order(blob_strings)))[:12000]

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
    "key responsibilities",
    "responsibilities",
    "what you'll do",
    "what you’ll do",
    "required",
    "qualifications",
    "minimum qualifications",
    "preferred qualifications",
    "requirements",
    "about you",
    "experience",
    "location",
    "work location",
    "remote",
    "hybrid",
}

ROLE_STOP_HEADINGS = {
    "company description",
    "about the company",
    "about us",
    "benefits",
    "what we offer",
    "additional information",
    "recruitment agencies",
    "privacy policy",
    "reasonable adjustments",
    "smart working",
    "equality, diversity and inclusion",
    "safeguarding",
    "contact",
    "read more",
    "follow us",
    "logo",
    "application",
    "how to apply",
    "application process",
    "recruitment process",
}

NOISE_LINE_PATTERNS = [
    r"^©\s*\d{4}",
    r"^follow us$",
    r"^read more$",
    r"^privacy policy$",
    r"^recruitment agencies$",
    r"^reasonable adjustments$",
    r"^smart working$",
    r"^equality, diversity and inclusion$",
    r"^safeguarding$",
    r"^contact$",
    r"^logo$",
    r"^apply$",
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
    if raw.endswith(":") and len(raw) <= 100:
        return True

    if len(x) <= 70 and not re.search(r"[.!?]", x):
        if any(k in x for k in ["responsib", "qualif", "experience", "requirements", "skills", "location", "remote", "hybrid"]):
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
        "div[class*='description']",
        "div[class*='posting']",
        "div[class*='content']",
        "div[class*='career']",
        "div[class*='opening']",
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
            if any(k in low_meta for k in ["job", "career", "opening", "posting", "description", "apply"]):
                score += 800
            if node.name in {"main", "article"}:
                score += 1000
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
        if len(txt) > 1600 and node.name in {"div", "span"}:
            continue
        if len(txt) < 2:
            continue
        blocks.append(txt)

    return dedupe_keep_order(blocks)[:300]


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
                x in norm for x in ["responsib", "qualif", "experience", "requirements", "skills", "location", "remote", "hybrid"]
            ):
                in_keep = True
                seen_useful_heading = True
                kept.append(block)
                continue

        if in_keep:
            if any(
                phrase in low
                for phrase in ["privacy policy", "follow us", "recruitment agencies", "all rights reserved", "read more"]
            ):
                break
            kept.append(block)

    if len(kept) < 6:
        fallback = []
        for block in blocks:
            norm = normalize_heading(block)
            low = block.lower().strip()
            if line_is_noise(block):
                continue
            if norm in ROLE_STOP_HEADINGS:
                continue
            if any(phrase in low for phrase in ["privacy policy", "follow us", "recruitment agencies", "all rights reserved", "read more"]):
                continue
            fallback.append(block)
        kept = fallback

    return clean_whitespace("\n".join(kept))


def extract_header_candidates(
    soup: BeautifulSoup,
    root,
    structured: Dict[str, Any],
    title_tag_text: str,
) -> List[str]:
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
        if not txt or len(txt) > 220:
            continue
        low = txt.lower()
        if any(
            token in low
            for token in [
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
                "salary",
                "contract",
                "work location",
                "right to work",
                "office attendance",
                "office based",
            ]
        ):
            out.append(txt)

    return dedupe_keep_order(out)[:100]


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
            phrase in low
            for phrase in [
                "all rights reserved",
                "smartrecruiters",
                "follow us on linkedin",
                "instagram",
                "linkedin",
                "recruitment agencies",
            ]
        ):
            continue

        cleaned.append(line)

    out = clean_whitespace("\n".join(dedupe_keep_order(cleaned)))
    return out[:12000]


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
        "requirements",
        "qualifications",
        "experience",
        "skills",
        "job description",
        "overview",
        "preferred",
        "required",
        "what you'll do",
        "what you’ll do",
        "minimum qualifications",
        "preferred qualifications",
        "about you",
        "location",
        "remote",
        "hybrid",
        "office attendance",
    ]
    for term in positive_terms:
        if term in low:
            score += 3

    negative_terms = [
        "about us",
        "company overview",
        "who we are",
        "our mission",
        "follow us",
        "privacy policy",
        "recruitment agencies",
        "read more",
    ]
    for term in negative_terms:
        if term in low:
            score -= 3

    score += min(len(text) // 600, 6)
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
    if len(role_body_text.strip()) < 400:
        role_body_text = merge_role_bodies(
            dom_role_text,
            visible_role_body,
            structured_role_body,
            json_role_body,
        )

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
