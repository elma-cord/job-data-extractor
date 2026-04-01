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


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "img", "picture", "source"]):
        tag.decompose()

    # keep meta content if useful
    meta_texts = []
    for attr in ("description", "og:description", "twitter:description"):
        node = soup.find("meta", attrs={"name": attr}) or soup.find("meta", attrs={"property": attr})
        if node and node.get("content"):
            meta_texts.append(node.get("content", ""))

    body_text = soup.get_text(separator="\n")
    combined = "\n".join(meta_texts + [body_text])
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
