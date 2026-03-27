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
        "react.js": r"(?<![A-Za-z0-9])(?:react\.js|reactjs|react js)(?![A-Za-z0-9])",
        "angular.js": r"(?<![A-Za-z0-9])(?:angular\.js|angularjs|angular js)(?![A-Za-z0-9])",
        "git": r"(?<![A-Za-z0-9])git(?![A-Za-z0-9])",
        "aws": r"(?<![A-Za-z0-9])aws(?![A-Za-z0-9])",
        "gcp": r"(?<![A-Za-z0-9])gcp(?![A-Za-z0-9])",
        "sql": r"(?<![A-Za-z0-9])sql(?![A-Za-z0-9])",
        "seo": r"(?<![A-Za-z0-9])seo(?![A-Za-z0-9])",
        "crm": r"(?<![A-Za-z0-9])crm(?![A-Za-z0-9])",
        "erp": r"(?<![A-Za-z0-9])erp(?![A-Za-z0-9])",
        "sap": r"(?<![A-Za-z0-9])sap(?![A-Za-z0-9])",
        "api": r"(?<![A-Za-z0-9])api(?![A-Za-z0-9])",
        "apis": r"(?<![A-Za-z0-9])apis(?![A-Za-z0-9])",
        "qa": r"(?<![A-Za-z0-9])qa(?![A-Za-z0-9])",
        "ui": r"(?<![A-Za-z0-9])ui(?![A-Za-z0-9])",
        "ux": r"(?<![A-Za-z0-9])ux(?![A-Za-z0-9])",
        "bi": r"(?<![A-Za-z0-9])bi(?![A-Za-z0-9])",
        "etl": r"(?<![A-Za-z0-9])etl(?![A-Za-z0-9])",
        "kpi": r"(?<![A-Za-z0-9])kpi(?:s)?(?![A-Za-z0-9])",
        "okrs": r"(?<![A-Za-z0-9])okrs?(?![A-Za-z0-9])",
    }
    if normalized in special_map:
        return re.compile(special_map[normalized], flags=re.I)

    escaped = re.escape(skill.strip())
    escaped = escaped.replace(r"\ ", r"[\s\-]+")
    return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", flags=re.I)


def build_skills_source_text(role_body_text: str) -> str:
    plain = strip_html(role_body_text)
    lines = split_lines(normalize_quotes(plain))
    out = []

    disallow = [
        "smartrecruiters",
        "workday, inc.",
        "privacy policy",
        "read more",
        "follow us",
        "recruitment agencies",
        "about the company",
        "company description",
        "benefits",
        "what we offer",
        "additional information",
        "equality, diversity and inclusion",
        "reasonable adjustments",
        "contact",
        "all rights reserved",
        "cookie policy",
        "terms of use",
        "great place to work",
        "employee wellbeing",
        "annual excellence awards",
        "why join us",
        "who we are",
        "our mission",
        "our values",
        "apply now",
        "submit application",
    ]

    for line in lines:
        low = line.lower()

        if len(line) < 2:
            continue
        if any(x in low for x in disallow):
            continue
        if re.match(r"^©\s*\d{4}", low):
            continue
        if re.match(r"^(linkedin|instagram|facebook|twitter|x\.com)\b", low):
            continue

        out.append(line)

    return clean_whitespace("\n".join(out))


def exact_match_skills_in_order(description: str, skill_list: List[str], limit: int = 10) -> List[str]:
    if not description or not skill_list:
        return []

    description = normalize_quotes(description)
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
