import json
import re
import time
from dataclasses import dataclass
from typing import Optional

import requests
from bs4 import BeautifulSoup


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-GB,en;q=0.9,en-US;q=0.8",
}


BLOCK_PATTERNS = [
    r"access denied",
    r"forbidden",
    r"captcha",
    r"verify you are human",
    r"checking your browser",
    r"unusual traffic",
    r"cloudflare",
    r"perimeterx",
    r"datadome",
    r"akamai",
    r"incapsula",
    r"imperva",
]


@dataclass
class FetchResult:
    url: str
    final_url: str
    status_code: Optional[int]
    ok: bool
    blocked: bool
    text: str
    error: str


def _clean_text(text: str) -> str:
    text = text or ""
    text = text.replace("\xa0", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _join_address_parts(parts: list[str]) -> str:
    clean = []
    for part in parts:
        part = (part or "").strip()
        if part and part not in clean:
            clean.append(part)
    return ", ".join(clean)


def _collect_structured_lines(obj, out: list[str]) -> None:
    if isinstance(obj, dict):
        address_locality = obj.get("addressLocality")
        address_region = obj.get("addressRegion")
        address_country = obj.get("addressCountry")

        if any([address_locality, address_region, address_country]):
            location_line = _join_address_parts([
                str(address_locality or ""),
                str(address_region or ""),
                str(address_country or ""),
            ])
            if location_line:
                out.append(f"Location: {location_line}")

        for key, value in obj.items():
            key_l = str(key).lower()

            if key_l in {"location", "joblocation", "baselocation", "locations"} and isinstance(value, str):
                out.append(f"Location: {value}")

            elif key_l in {"workplacetype", "remotestatus", "locationtype", "remote_type", "workplace"} and isinstance(value, str):
                out.append(f"Workplace type: {value}")

            elif key_l == "applicantlocationrequirements":
                if isinstance(value, str):
                    out.append(f"Location: {value}")
                elif isinstance(value, dict):
                    _collect_structured_lines(value, out)
                elif isinstance(value, list):
                    for item in value:
                        _collect_structured_lines(item, out)

            elif key_l in {"address", "joblocation"} and isinstance(value, dict):
                _collect_structured_lines(value, out)

            elif isinstance(value, (dict, list)):
                _collect_structured_lines(value, out)

    elif isinstance(obj, list):
        for item in obj:
            _collect_structured_lines(item, out)


def _extract_structured_text_from_html(soup: BeautifulSoup) -> str:
    lines = []

    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text() or ""
        raw = raw.strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
            _collect_structured_lines(data, lines)
        except Exception:
            continue

    return _clean_text("\n".join(lines))


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "img", "picture", "source"]):
        if tag.get("type") == "application/ld+json":
            continue
        tag.decompose()

    meta_texts = []
    for attr in ("description", "og:description", "twitter:description"):
        node = soup.find("meta", attrs={"name": attr}) or soup.find("meta", attrs={"property": attr})
        if node and node.get("content"):
            meta_texts.append(node.get("content", ""))

    structured_text = _extract_structured_text_from_html(BeautifulSoup(html or "", "html.parser"))
    body_text = soup.get_text(separator="\n")

    combined_parts = []
    if meta_texts:
        combined_parts.append("\n".join(meta_texts))
    if structured_text:
        combined_parts.append(structured_text)
    if body_text:
        combined_parts.append(body_text)

    combined = "\n".join(combined_parts)
    return _clean_text(combined)


def _is_blocked(status_code: Optional[int], text: str) -> bool:
    if status_code in {401, 403, 406, 429, 503}:
        return True
    lower = (text or "").lower()
    return any(re.search(p, lower) for p in BLOCK_PATTERNS)


def fetch_job_page_text(url: str, timeout: int = 25, sleep_seconds: float = 0.0) -> FetchResult:
    if not url:
        return FetchResult(
            url=url,
            final_url=url,
            status_code=None,
            ok=False,
            blocked=False,
            text="",
            error="missing_url",
        )

    try:
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

        response = requests.get(
            url,
            headers=DEFAULT_HEADERS,
            timeout=timeout,
            allow_redirects=True,
        )

        html = response.text or ""
        text = _html_to_text(html)
        blocked = _is_blocked(response.status_code, text)

        return FetchResult(
            url=url,
            final_url=str(response.url),
            status_code=response.status_code,
            ok=response.ok and not blocked,
            blocked=blocked,
            text=text,
            error="",
        )
    except Exception as exc:
        return FetchResult(
            url=url,
            final_url=url,
            status_code=None,
            ok=False,
            blocked=False,
            text="",
            error=f"fetch_error: {exc}",
        )
