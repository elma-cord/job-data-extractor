def build_relevance_prompt(job_title: str, role_context_text: str) -> str:
    return f"""
Return valid JSON only:
{{
  "role_relevance": "Relevant" or "Not relevant",
  "job_category": "T&P" or "NonT&P",
  "role_relevance_reason": "short reason"
}}

You are screening a role for a recruiting workflow.

Use only:
1. position name
2. current provided job description / role context

Important rules:
- The provided description is the primary source of truth.
- Do not guess from company name.
- Ignore legal text, privacy text, cookie text, navigation, benefits-only text, and unrelated company marketing text.
- Keep the reason short and concrete.
- Business/commercial roles can still be relevant. Do not reject a role just because it is not technical.

Relevant roles include technical and business functions such as:
software, engineering, product, data, AI/ML, QA, DevOps, cloud, infrastructure, security, UX/UI, IT support, solutions engineering, HR, people ops, talent, finance, accounting, legal, compliance, risk, revops, sales ops, operations, PMO, programme/project/change/transformation, customer success, account management, renewals, implementation, partnerships, executive assistant, chief of staff, founder, C-level, sales development, business development.

Treat these as relevant:
- PMO / programme / project / transformation / business operations roles
- IT support / infrastructure / support engineering roles
- National Account Manager / Key Account Manager / Account Manager / Customer Success roles
- Finance / accounting / legal / compliance / revops / sales ops
- HR / TA / People roles
- Business Development Representative / Sales Development Representative / BDR / SDR / Account Executive

Clearly not relevant:
teacher, nurse, waiter, chef, cleaner, warehouse operative, driver, construction, civil engineering, retail, electrician, mechanical/manufacturing plant roles, microbiology, maritime, injection molding, beauty brand store roles.

Location rules:
- UK: onsite, hybrid, or remote allowed
- Ireland: remote only
- Europe: allowed only if explicitly Remote Europe or Remote EMEA
- Global / worldwide remote: allowed unless clearly restricted to excluded regions
- If location is missing or unclear, do not reject for that reason alone
- If role clearly requires USA/Canada/APAC/LATAM or another excluded region, mark Not relevant

Language rule:
- If a non-English language is clearly required, mark Not relevant
- If another language is only preferred/bonus, do not reject for that reason

T&P:
software, engineering, product, data, AI/ML, QA, DevOps, cloud, systems, network, security, IT support, infrastructure, solutions engineering, technical architecture

NonT&P:
HR, talent, people, operations, PMO, project/programme/change, finance, legal, compliance, risk, marketing, partnerships, executive assistant, chief of staff, account/customer/renewals/implementation roles, sales/business development roles

Position name:
{job_title}

Current provided job description / role context:
{role_context_text[:12000]}
""".strip()


def build_core_fields_prompt(
    job_title: str,
    header_text: str,
    role_body_text: str,
    allowed_locations: list[str],
) -> str:
    location_list_text = "\n".join(allowed_locations[:1800])

    return f"""
Return valid JSON only:
{{
  "job_category": "T&P" or "NonT&P" or "",
  "job_location": "",
  "remote_preferences": "",
  "remote_days": "",
  "salary_min": "",
  "salary_max": "",
  "salary_currency": "",
  "salary_period": "",
  "visa_sponsorship": "yes|no|",
  "job_type": "Permanent|FTC|Part Time|Freelance/Contract|"
}}

Extract fields from the CURRENT PROVIDED JOB DESCRIPTION only.
Be strict. Do not guess.
If unsupported, return empty string.
Exceptions:
- remote_preferences may be "not specified"
- remote_days may be "not specified"

Rules:

1. job_location
- Find the work location from the provided text.
- Prefer location lines near: location, based in, office, work location, country, city, based at, remote in.
- Choose one normalized location from the allowed list where possible.
- If exact match is unavailable, choose the closest broader match from the list.
- Do not infer from company HQ, company name, or currency alone.

2. remote_preferences
- Normalize only to: onsite, hybrid, remote
- Multiple values allowed, in this order only: onsite, hybrid, remote
- If unclear, return "not specified"

3. remote_days
- Return highest remote days in a 5-day week as a string number, or "not specified"
- Only return a number when office days or remote days are explicit
- Examples:
  - 2 days in office => 3
  - 1-2 days in office => 4
  - 2-3 remote days => 3
  - 60% office based => 2
- Fully remote => "not specified"

4. salary
- Extract only explicit salary for this role
- If one number only, use it as both min and max
- salary_period must be one of: year, day, hour, month, or ""
- Do not use years of experience, dates, ids, percentages, employee counts, revenue, benefits budgets, or unrelated numbers

5. visa_sponsorship
- yes only if explicitly available
- no only if explicitly unavailable / right to work required / no sponsorship
- otherwise ""

6. job_type
Use exactly one of:
- Permanent
- FTC
- Part Time
- Freelance/Contract

Map:
- Permanent => permanent, full time, full-time
- FTC => temporary, fixed term, fixed-term, maternity cover
- Part Time => part time, part-time, job share
- Freelance/Contract => freelance, contract, contracting

7. job_category
- Return T&P or NonT&P only if clear
- Business development, SDR/BDR, account, customer success, operations, finance, legal, HR, PMO, marketing are usually NonT&P.
- Do not label a role T&P just because it works with technical customers.

Allowed normalized locations:
{location_list_text}

Position name:
{job_title}

Header/meta text:
{header_text[:2500]}

Current provided job description:
{role_body_text[:10000]}
""".strip()


def build_location_normalization_prompt(input_location: str, allowed_locations: list[str]) -> str:
    location_list_text = "\n".join(allowed_locations[:2000])

    return f"""
Output only one value:
- the chosen normalized location exactly as it appears in the acceptable locations list
- or Unknown

Choose the best matching normalized location from the list.
If multiple match, choose the most specific.
If the input is a small town not in the list, choose the closest broader match.
If none match, output Unknown only.

Input location:
{input_location}

Acceptable normalized locations:
{location_list_text}
""".strip()


def build_salary_normalization_prompt(salary_text: str, predefined_salaries: list[int]) -> str:
    salaries_text = ", ".join(str(x) for x in predefined_salaries[:3000])

    return f"""
Output only one line in this exact format:
[min salary] - [max salary] [currency code]

Tasks:
- Extract min and max salary from the text
- Snap both to the closest values from the predefined list
- If only one salary is present, use it for both
- Identify the currency
- Never output a salary not in the predefined list
- If invalid, output:
 -  

Salary text:
{salary_text}

Predefined salary values:
{salaries_text}
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

Extract salary only if explicitly stated for this role in the CURRENT PROVIDED JOB DESCRIPTION.

Rules:
- salary_period must be: year, day, hour, month, or ""
- If one amount only, use same value for min and max
- Keep raw explicit numeric salary values only
- Do not round or normalize
- Do not use years of experience, dates, ids, percentages, employee counts, revenue, equity-only figures, bonus-only figures, benefits budgets, or unrelated numbers
- If salary is ambiguous or not clearly tied to the role, return empty strings

Position name:
{job_title}

Header/meta text:
{header_text[:2000]}

Current provided job description:
{role_body_text[:8000]}
""".strip()


def build_job_titles_prompt(position_name: str, description: str, allowed_job_titles: list[str]) -> str:
    job_titles_text = ", ".join(allowed_job_titles[:2200])

    return f"""
Return valid JSON only:
{{
  "job_titles": ["..."]
}}

You are mapping a position to an ALLOWED JOB TITLE LIST.

You will receive:
1. the input position name
2. the current provided job description
3. the predefined allowed job titles list

Your task:
Choose up to 3 best matching job titles from the predefined list.

Hard rules:
- Use ONLY titles from the predefined list.
- Prefer the closest functional match from the allowed list.
- If the position name exactly matches one predefined title, return ONLY that one.
- If the title strongly and clearly maps to one allowed title, return ONLY that one.
- Do not force 3 titles.
- Do not guess from company background.
- Do not return broad random alternatives if one clear match exists.
- If no suitable match exists, return [].

Very important:
- Job title mapping should be driven primarily by the INPUT POSITION NAME.
- Use description to confirm function, not to invent distant alternatives.
- Do not output unrelated families just because some words overlap in the description.

Critical mapping rules:
- Business Development Representative / Sales Development Representative / BDR / SDR -> prefer SDR / BDR
- National Account Manager / Key Account Manager / Account Manager / Customer Success Manager -> prefer CSM / Account Manager when suitable
- Account Executive -> use only when clearly AE/new-business/hunter/closing role
- PMO / Programme / Change / Transformation -> prefer the closest operations/project/programme title
- People Ops / Talent / HR / Recruiter -> prefer the closest HR / TA title
- Finance / FP&A / Accounting -> prefer the closest finance title
- Marketing / campaign / brand / content / CRM / demand gen roles -> prefer marketing titles
- Support / helpdesk / technical support / service desk roles -> prefer Support Engineer / Customer Support / Customer Service Representative where appropriate
- System Engineer should be used ONLY when there is strong systems / infrastructure / platform / network / linux / kubernetes / windows server / vmware / citrix type evidence
- Do NOT map marketing, business development, account, customer, finance, HR, PMO, legal, operations roles to System Engineer
- Do NOT mix unrelated title families

Bad example for Business Development Representative:
- System Administrator
- Content Marketing
- Account Executive

Good example for Business Development Representative:
- SDR / BDR

Predefined job titles:
{job_titles_text}

Input position name:
{position_name}

Current provided job description:
{description[:9000]}
""".strip()


def build_seniority_prompt(position_name: str, description: str) -> str:
    return f"""
Return valid JSON only:
{{
  "seniorities": ["entry|junior|mid|senior|lead|leadership"]
}}

Choose up to 3 seniority values using only:
entry, junior, mid, senior, lead, leadership

Rules:
- Use only levels strongly supported by title and description
- Do not over-tag
- Keep lowercase
- Keep order: entry, junior, mid, senior, lead, leadership

Strong rules:
- Head / Director / VP / Chief / C-level => leadership
- Engineering Manager or similar strong people-management title => leadership
- Plain manager titles usually mean senior and/or lead, not mid
- Do not include mid for manager titles unless clearly supported

Guidance:
- Junior titles => junior
- Senior titles => senior
- Lead / Principal / Staff often => senior and/or lead
- Assistant / Associate / Graduate / Intern => entry and/or junior
- If unclear, return []

Position name:
{position_name}

Current provided job description:
{description[:9000]}
""".strip()


def build_contract_type_prompt(text: str) -> str:
    return f"""
Output only one of these exact values:
Permanent
FTC
Part Time
Freelance/Contract

Or output empty string if no valid type is found.

Priority if multiple are mentioned:
Permanent > FTC > Part Time > Freelance/Contract

Mappings:
- Permanent => permanent, full time, full-time, standard
- FTC => temporary, fixed term, fixed-term, maternity cover
- Part Time => part time, part-time, job share
- Freelance/Contract => freelance, contract, contracting

Input text:
{text[:8000]}
""".strip()


def build_job_description_prompt(description_text: str) -> str:
    return f"""
Return valid JSON only:
{{
  "status": "unused"
}}

Compatibility prompt only.
Do not transform or rewrite the description.
The workflow no longer outputs a generated job_description column.

Input text:
{description_text[:1000]}
""".strip()


def build_skills_full_prompt(role_category: str, description: str, candidate_skills: list[str], allowed_skills: list[str]) -> str:
    allowed_skills_text = ", ".join(allowed_skills[:3000])
    candidate_skills_text = ", ".join(candidate_skills[:80])

    return f"""
Return valid JSON only:
{{
  "role_category": "T&P" or "NonT&P",
  "skills": ["..."]
}}

Rules:
- role_category is provided. Echo it exactly.
- skills must contain 2 to 10 items when clearly supported
- Use only skills from the allowed list
- Prefer concrete skills clearly evidenced by the CURRENT PROVIDED JOB DESCRIPTION
- candidate_skills is the primary source when useful
- Never output a skill not in the allowed list
- Remove any skill not exactly present in the allowed list
- Do not use footer, legal, navigation, marketing, or unrelated company text
- Do not invent generic skills

role_category:
{role_category}

candidate_skills:
{candidate_skills_text}

Allowed skills:
{allowed_skills_text}

Current provided job description:
{description[:10000]}
""".strip()


def build_skills_additional_prompt(
    role_category: str,
    description: str,
    existing_skills: list[str],
    candidate_skills: list[str],
    allowed_skills: list[str],
) -> str:
    allowed_skills_text = ", ".join(allowed_skills[:3000])
    existing_skills_text = ", ".join(existing_skills[:80])
    candidate_skills_text = ", ".join(candidate_skills[:80])

    return f"""
Return valid JSON only:
{{
  "role_category": "T&P" or "NonT&P",
  "additional_skills": ["..."]
}}

Rules:
- role_category is provided. Echo it exactly.
- Return 2 to 5 additional skills when clearly supported
- Do not repeat any skill from existing_skills
- Use only skills from the allowed list
- candidate_skills is the primary source when useful
- If needed, infer from the CURRENT PROVIDED JOB DESCRIPTION
- Remove any skill not exactly present in the allowed list
- Do not use footer, legal, navigation, marketing, or unrelated company text
- Do not invent generic skills

role_category:
{role_category}

existing_skills:
{existing_skills_text}

candidate_skills:
{candidate_skills_text}

Allowed skills:
{allowed_skills_text}

Current provided job description:
{description[:10000]}
""".strip()


def build_skills_prompt(role_category: str, description: str, exact_skills: list[str], allowed_skills: list[str]) -> str:
    return build_skills_full_prompt(
        role_category=role_category,
        description=description,
        candidate_skills=exact_skills,
        allowed_skills=allowed_skills,
    )
