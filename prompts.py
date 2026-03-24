def build_relevance_prompt(job_title: str, role_context_text: str) -> str:
    return f"""
Return valid JSON only:
{{
  "role_relevance": "Relevant" or "Not relevant",
  "role_relevance_reason": "short reason"
}}

Task:
Decide if this role is relevant.

Relevant families include:
- Account / Customer: Account Director, Account Executive, CSM / Account Manager, Renewals, Customer Operations, Customer Support
- Business / Ops: Business Analyst, Business Operations, Operations, Project Manager, PMO, Programme Manager, Change / Transformation
- HR / Talent: Talent Acquisition, Human Resources, People Ops, Recruiter, Recruitment
- Finance / Legal / Risk: Finance / Accounting, FP&A, Legal, Risk & Compliance, RevOps, Sales Operations
- Marketing: Product Marketing, Digital Marketing, Growth, Performance, SEO, PR / Communications, Marketing Analyst
- Technical: Software / Data / AI / ML / Product / UX / UI / QA / Security / DevOps / Cloud / Infrastructure / Network / Support / System Engineer / System Administrator / Solutions Engineer

Important allow rules:
- Internal recruitment / TA / HR / People roles are relevant.
- IT support / infrastructure / support engineering roles are relevant.
- Business/program/change/transformation/PMO/operations roles can be relevant.
- National Account Manager / Key Account Manager / Account Manager / Customer Success style roles can be relevant under CSM / Account Manager.

Clearly not relevant examples:
teacher, nurse, waiter, chef, construction worker, civil engineer, electrician, mechanical engineer in plant/manufacturing context, warehouse operative, driver, cleaner.

Location rules:
- UK: onsite, hybrid, remote allowed
- Ireland: remote only
- Europe: only explicitly Remote Europe / Remote EMEA
- Remote Global allowed unless clearly restricted to excluded regions
- If clearly USA/Canada/other non-allowed region with no UK/allowed evidence, mark Not relevant

Language rule:
- If clearly requires a non-English language, mark Not relevant

Decision order:
1. Role family
2. Location / remote setup
3. Final answer

Position name:
{job_title}

Description:
{role_context_text[:14000]}
""".strip()


def build_core_fields_prompt(
    job_title: str,
    header_text: str,
    role_body_text: str,
    allowed_locations: list[str],
) -> str:
    location_list_text = "\n".join(allowed_locations[:2000])

    return f"""
Return valid JSON only:
{{
  "job_category": "T&P" or "NonT&P",
  "job_location": "",
  "remote_preferences": "onsite|hybrid|remote|",
  "remote_days": "",
  "visa_sponsorship": "yes|no|",
  "job_type": "Permanent|FTC|Part Time|Freelance/Contract|"
}}

Rules:
- job_category:
  - T&P = software, engineering, data, IT, product, UX/UI, QA, DevOps, cloud, infra, network, support, systems, solutions engineering
  - NonT&P = other relevant business roles

- job_location:
  - Prefer explicit header/location labels first
  - Prefer the most specific visible valid location
  - If "UK remote" / "remote working within the UK", return the matching UK location if clear
  - Return "" if unclear

- remote_preferences:
  - Allowed only: onsite, hybrid, remote, or ""
  - home based = remote
  - office attendance wording usually = hybrid
  - do not let generic company-wide flexibility override role-specific wording

- remote_days:
  - Return number as string or ""
  - Only when explicit office/remote day evidence exists
  - 1-2 office days/week => 3 remote days
  - 60% office based => convert to 5-day week remainder
  - Do NOT infer from flexible working, compressed week, benefits text, or generic hybrid wording

- visa_sponsorship:
  - yes, no, or ""

- job_type:
  - Permanent, FTC, Part Time, Freelance/Contract, or ""
  - Priority: Permanent > FTC > Part Time > Freelance/Contract

Allowed locations:
{location_list_text}

Position name:
{job_title}

Header/meta text:
{header_text[:4000]}

Role description:
{role_body_text[:12000]}
""".strip()


def build_salary_prompt(job_title: str, header_text: str, role_body_text: str) -> str:
    return f"""
Return valid JSON only:
{{
  "salary_min": "",
  "salary_max": "",
  "salary_currency": "",
  "salary_period": ""
}}

Rules:
- Extract salary only if explicitly stated for this role
- salary_period must be one of: year, day, hour, month, or ""
- If one amount only, use same value for min and max
- Do NOT use years of experience, dates, ids, percentages, employee counts, revenue, benefits, office attendance percentages, or unrelated numbers
- Keep raw explicit numeric salary values
- If no clear salary, return empty strings

Position name:
{job_title}

Header/meta text:
{header_text[:3000]}

Role description:
{role_body_text[:10000]}
""".strip()


def build_job_titles_prompt(position_name: str, description: str, allowed_job_titles: list[str]) -> str:
    job_titles_text = ", ".join(allowed_job_titles[:2500])

    return f"""
Return valid JSON only:
{{
  "job_titles": ["..."]
}}

Task:
Choose up to 3 best matching job titles from the predefined list.

Rules:
- Use only titles from the predefined list
- If the position name exactly matches one predefined title, return only that one
- Otherwise choose up to 3 best functional matches based on title + description
- Prefer functional fit over literal wording

Important mappings:
- National Account Manager / Key Account Manager / Account Manager / Customer Success style roles should include "CSM / Account Manager" when available
- Systems / infrastructure / Kubernetes / Linux / OpenShift / virtualisation-heavy roles should prefer "System Engineer"
- CRM / loyalty / campaign / customer analytics roles should prefer "Marketing Analyst" and/or "Data / Insight Analyst"
- Do not force "Account Executive" unless clearly new-business / hunter / AE style

Predefined job titles:
{job_titles_text}

Position name:
{position_name}

Description:
{description[:12000]}
""".strip()


def build_seniority_prompt(position_name: str, description: str) -> str:
    return f"""
Return valid JSON only:
{{
  "seniorities": ["entry|junior|mid|senior|lead|leadership"]
}}

Rules:
- Allowed values only: entry, junior, mid, senior, lead, leadership
- Keep order: entry, junior, mid, senior, lead, leadership
- If title includes head of, director, technical director, engineering manager, vp, chief => leadership
- Plain manager titles usually mean senior and/or lead, not mid
- For PMO / Programme / Project / Operations / Account Manager:
  - do not include mid unless clearly junior/associate/assistant
  - include lead for ownership / stakeholder management / governance / team leadership
  - include senior for broad ownership / lifecycle / forecasting / oversight / complex stakeholder work
- If title explicitly says junior / senior / lead, include that
- If experience is:
  - 0-1 years => entry, junior
  - 2 years => junior, mid
  - 3-5 years => senior
  - 5+ years => senior, lead
- Return up to 3 values
- If unclear, return empty array

Position name:
{position_name}

Description:
{description[:12000]}
""".strip()


def build_skills_prompt(role_category: str, description: str, exact_skills: list[str], allowed_skills: list[str]) -> str:
    allowed_skills_text = ", ".join(allowed_skills[:3500])
    exact_skills_text = ", ".join(exact_skills)

    return f"""
Return valid JSON only:
{{
  "role_category": "T&P or NonT&P",
  "skills": ["..."]
}}

Goal:
- Keep exact_skills first
- Add only clearly evidenced missing skills from the description
- Use only skills from the allowed list
- Do not infer adjacent technologies
- Do not use footer text, page chrome, marketing text, legal text, partner lists, or unrelated content
- Return up to 10 skills total
- If unsure, exclude

role_category:
{role_category}

exact_skills:
{exact_skills_text}

Allowed skills:
{allowed_skills_text}

Description:
{description[:12000]}
""".strip()
