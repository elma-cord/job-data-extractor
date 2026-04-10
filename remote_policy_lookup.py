import json
import os
import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

from google import genai
from google.genai import types


GEMINI_REMOTE_MODEL = "gemini-2.5-flash"

UNKNOWN_POLICY = "unknown"
ALLOWED_POLICIES = {"onsite", "hybrid", "remote", "unknown"}

COMMON_NON_COMPANY_HOSTS = {
    "linkedin.com",
    "www.linkedin.com",
    "uk.linkedin.com",
    "glassdoor.com",
    "www.glassdoor.com",
    "indeed.com",
    "www.indeed.com",
    "greenhouse.io",
    "boards.greenhouse.io",
    "jobs.ashbyhq.com",
}


@dataclass
class RemotePolicyResult:
    remote_preferences: str
    note: str
    source_count: int = 0


def _clean_text(value: str) -> str:
    value = value or ""
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _extract_json_object(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        return {}

    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}

    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _normalize_policy(value: str) -> str:
    value = _clean_text(value).lower()
    if value in ALLOWED_POLICIES:
        return value
    return UNKNOWN_POLICY


def _normalize_domain(value: str) -> str:
    value = _clean_text(value).lower()
    value = re.sub(r"^https?://", "", value)
    value = value.split("/")[0].strip()
    value = value.strip(".")
    if value.startswith("www."):
        value = value[4:]
    return value


def _domain_from_url(url: str) -> str:
    url = _clean_text(url)
    if not url:
        return ""

    try:
        parsed = urlparse(url if "://" in url else f"https://{url}")
        host = (parsed.netloc or "").lower().strip()
        if not host:
            return ""
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def _get_best_company_domain(row: dict) -> str:
    for key in ["company_domain", "company_website", "website", "domain"]:
        value = _normalize_domain(str(row.get(key, "")))
        if value:
            return value

    job_url_domain = _domain_from_url(str(row.get("job_url", "")))
    if job_url_domain and job_url_domain not in COMMON_NON_COMPANY_HOSTS:
        return job_url_domain

    return ""


def _build_prompt(company_name: str, company_domain: str) -> str:
    company_name = _clean_text(company_name)
    company_domain = _clean_text(company_domain)

    query_hint = company_domain or company_name or "the company"

    return f"""
You are checking a company's CURRENT remote working policy using live web search.

Company name: {company_name}
Company domain: {company_domain}

Task:
Find the current company-level remote working policy for this company.
Prefer official sources first:
1. company careers site
2. company jobs/careers pages
3. company about/workplace/culture pages
4. only then reputable secondary sources

Important rules:
- We are looking for company-wide or broadly stated workplace policy, not just one random job.
- If evidence is mixed, outdated, weak, or only about one specific role, return "unknown".
- Be conservative. Do not guess.
- Accept only one of these values:
  - "onsite"
  - "hybrid"
  - "remote"
  - "unknown"
- "hybrid" means there is clear evidence of both office and remote/home working.
- "remote" means clearly remote-first / remote-only / distributed / work from anywhere.
- "onsite" means mainly office/site-based with no clear company-wide hybrid/remote policy.
- If nothing trustworthy is found, return "unknown".

Return valid JSON only:
{{
  "remote_preferences": "onsite|hybrid|remote|unknown",
  "reason": "very short reason",
  "sources": [
    {{
      "title": "",
      "url": ""
    }}
  ]
}}

Search target: remote working policy for {query_hint}
""".strip()


class RemotePolicyLookup:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("missing_gemini_api_key")
        self.client = genai.Client(api_key=api_key)

    def lookup(self, row: dict) -> RemotePolicyResult:
        company_name = _clean_text(str(row.get("company_name", "")))
        company_domain = _get_best_company_domain(row)

        if not company_name and not company_domain:
            return RemotePolicyResult(
                remote_preferences=UNKNOWN_POLICY,
                note="remote_policy_lookup: skipped - missing company name/domain",
                source_count=0,
            )

        prompt = _build_prompt(company_name=company_name, company_domain=company_domain)

        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        config = types.GenerateContentConfig(
            tools=[grounding_tool],
            temperature=0,
        )

        try:
            response = self.client.models.generate_content(
                model=GEMINI_REMOTE_MODEL,
                contents=prompt,
                config=config,
            )
        except Exception as exc:
            return RemotePolicyResult(
                remote_preferences=UNKNOWN_POLICY,
                note=f"remote_policy_lookup_error: {exc}",
                source_count=0,
            )

        text = getattr(response, "text", "") or ""
        payload = _extract_json_object(text)

        policy = _normalize_policy(str(payload.get("remote_preferences", "")))
        reason = _clean_text(str(payload.get("reason", "")))

        sources = payload.get("sources", [])
        cleaned_sources = []
        if isinstance(sources, list):
            for item in sources[:2]:
                if not isinstance(item, dict):
                    continue
                title = _clean_text(str(item.get("title", "")))
                url = _clean_text(str(item.get("url", "")))
                if title or url:
                    cleaned_sources.append({"title": title, "url": url})

        if policy == UNKNOWN_POLICY and not reason:
            reason = "insufficient trustworthy evidence"

        source_count = len(cleaned_sources)

        source_note_parts = []
        for item in cleaned_sources:
            title = item.get("title", "")
            url = item.get("url", "")
            if title and url:
                source_note_parts.append(f"{title} ({url})")
            elif title:
                source_note_parts.append(title)
            elif url:
                source_note_parts.append(url)

        source_note = "; ".join(source_note_parts)

        if source_note:
            note = f"remote_policy_lookup: {policy} - {reason}; sources: {source_note}"
        else:
            note = f"remote_policy_lookup: {policy} - {reason}"

        return RemotePolicyResult(
            remote_preferences=policy,
            note=note[:1200],
            source_count=source_count,
        )
