import re
from typing import List


def clean_whitespace(text: str) -> str:
    text = text or ""
    text = text.replace("\xa0", " ")
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_html(text: str) -> str:
    from bs4 import BeautifulSoup
    return clean_whitespace(BeautifulSoup(text or "", "lxml").get_text("\n", strip=True))


def split_lines(text: str) -> List[str]:
    return [clean_whitespace(x) for x in (text or "").splitlines() if clean_whitespace(x)]


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


def html_escape(text: str) -> str:
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def plain_text_to_html_preserve_structure(job_description_text: str) -> str:
    text = clean_whitespace(job_description_text)
    if not text:
        return ""

    lines = split_lines(text)
    if not lines:
        return ""

    html_parts: List[str] = []
    bullet_buffer: List[str] = []

    def flush_bullets() -> None:
        nonlocal bullet_buffer, html_parts
        if bullet_buffer:
            html_parts.append("<ul>")
            for item in bullet_buffer:
                html_parts.append(f"<li>{html_escape(item)}</li>")
            html_parts.append("</ul>")
            bullet_buffer = []

    def is_heading_line(line: str) -> bool:
        raw = normalize_quotes(line).strip()
        norm = raw.lower().strip(" :-•\t")
        if raw.endswith(":") and len(raw) <= 120:
            return True
        if norm in {
            "role and responsibilities",
            "key responsibilities",
            "skills, knowledge and expertise",
            "person specification",
            "required qualifications",
            "preferred qualifications",
            "responsibilities",
            "about you",
            "experience you'll bring",
            "experience you’ll bring",
            "main responsibilities and accountabilities",
            "other responsibilities",
            "skills and competencies",
            "strategic and growth focused activities",
            "recruitment delivery",
            "recruitment coordination and administration",
        }:
            return True
        if len(raw) <= 90 and raw == raw.title() and not re.search(r"[.!?]$", raw):
            return True
        return False

    def is_bullet_like(line: str) -> bool:
        raw = normalize_quotes(line).strip()
        if re.match(r"^[-*•]\s+", raw):
            return True
        if len(raw) <= 240 and not raw.endswith(".") and not raw.endswith(":") and raw[:1].isupper():
            return True
        return False

    for line in lines:
        raw = normalize_quotes(line).strip()
        if not raw:
            continue

        if is_heading_line(raw):
            flush_bullets()
            heading = raw[:-1].strip() if raw.endswith(":") else raw
            html_parts.append(f"<b>{html_escape(heading)}</b>")
            continue

        cleaned_bullet = re.sub(r"^[-*•]\s+", "", raw).strip()
        if is_bullet_like(raw):
            bullet_buffer.append(cleaned_bullet)
            continue

        flush_bullets()
        html_parts.append(f"<p>{html_escape(raw)}</p>")

    flush_bullets()
    return "\n".join(html_parts).strip()


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
