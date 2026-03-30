def build_relevance_prompt(job_title: str, role_context_text: str) -> str:
    return f"""
Return valid JSON only:
{{
  "role_relevance": "Relevant" or "Not relevant",
  "job_category": "T&P" or "NonT&P",
  "role_relevance_reason": "short reason"
}}

You will receive two inputs: position name and job description.

Decide role relevance using these rules:

1. Relevant roles match this kind of allowed list or close synonyms/specializations:
Account Director, Account Executive, AI Engineer, Automation Engineer, Back End, BI Developer, Big Data Engineer, Brand Marketing, Business Analyst, Business Development Manager, Business Operations, CDO, CFO, Chief of Staff, CIO, CLO, Cloud Engineer, CMO, Computer Vision Engineer, Content Marketing, COO, Copywriting, CPO, CRM Developer, CRM Manager, CRO, CSM / Account Manager, CSO, CTO, Customer Operations, Customer Service Representative, Customer Support, Data Architect, Data Engineer, Data Scientist, Data / Insight Analyst, Database Engineer, Deep Learning Engineer, Demand / Lead Generation, Developer in Test, DevOps Engineer, Digital Marketing, Embedded Developer, Engineering Manager, Events and Community, Executive Assistant, Finance / Accounting, Founder, FP&A, Front End, Full Stack, Games Designer, Games Developer, Generalist Marketing, Graphic Designer, Graphics Developer, Growth Marketing, Head of Customer, Head of Data, Head of Design, Head of Engineering, Head of Finance, Head of HR, Head of Infrastructure, Head of Marketing, Head of Operations, Head of Product, Head of QA, Head of Sales, Human Resources, Implementation Manager, Integration Developer, Legal, Machine Learning Engineer, Marketing Analyst, Mobile Developer, Network Engineer, Operations, Partnerships, Penetration Tester, People Ops, Performance Marketing, PR / Communications, Product Manager, Product Marketing, Product Owner, Project Manager, QA Automation Tester, QA Manual Tester, Quality Assurance, Quantitative Developer, Renewals Manager, Research Engineer, RevOps, Risk and Compliance, Sales Engineer, Sales Operations, Scrum Master, SDR / BDR, Security, Security Engineer, SEO Marketing, Site Reliability Engineer, Social Media Marketing, Solutions Engineer, Support Engineer, System Administrator, System Engineer, Talent Acquisition, Technical Architect, Technical Director, Technical Writer, Testing Manager, UI Designer, UI/UX Designer, UX Designer, UX Researcher, Videography, VP of Engineering.

2. Clearly outside-tech/business roles are Not relevant even if some words overlap.

3. Exclude roles related to construction, civil engineering, retail, electrical, mechanical, manufacturing, microbiology, maritime, injection molding, and beauty brands.

4. Location rules:
- United Kingdom: onsite, hybrid, or remote allowed
- Ireland: only remote allowed
- Europe: only explicitly Remote Europe or Remote EMEA allowed
- Remote Global / worldwide allowed unless clearly restricted to APAC, LATAM, Africa, USA, Canada, or another disallowed region
- If the role mentions a disallowed region with no evidence it can be worked from an allowed region, mark Not relevant
- If salary is clearly USD/CAD and there is no evidence the role can be done from allowed regions, mark Not relevant
- If location is missing or unclear, do not reject for that alone

5. Language rules:
- If the role requires any language other than English, mark Not relevant
- Preferred / bonus extra language is okay

6. T&P category:
software development, engineering, product management, data science, IT, UX/UI, QA, DevOps, security, cloud, infrastructure, support engineering, systems, solutions engineering

7. NonT&P category:
business development, SDR/BDR, account management, customer success, renewals, implementation, partnerships, operations, PMO, finance, accounting, legal, HR, marketing, executive assistant, chief of staff, business operations

Output only JSON.

Position name:
{job_title}

Job description:
{role_context_text[:14000]}
""".strip()


def build_core_fields_prompt(
    job_title: str,
    header_text: str,
    role_body_text: str,
    allowed_locations: list[str],
) -> str:
    location_list_text = "\n".join(allowed_locations[:2500])

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

You will receive a relevant job only.

Perform these tasks carefully:

1. Position Location
- Read the entire description to find the work location
- Focus near keywords like: location, based in, office in, work location, remote in, country, city, based at
- If multiple locations appear, select the most specific one
- Match one entry from the acceptable normalized locations list where possible
- If the extracted location does not exactly exist in the list, choose the closest broader location
- If no match is found, return ""

2. Position Remote Preferences
- Extract all remote working preferences anywhere in the description
- Normalize onsite variants to onsite
- Normalize hybrid variants to hybrid
- Normalize remote variants to remote
- Output in this order only: onsite, hybrid, remote
- Separate multiple values with comma + space
- If none appear, return "not specified"

3. Remote Days
Return only the highest number of remote work days in a 5-day week, as a string, or "not specified"
Rules:
- Return a number only if remote/office days are explicitly stated
- If a range is given, return the highest remote days
- If office days are stated, calculate remote days as 5 - office days
- If fully remote, remote every day, ambiguous, or no remote work allowed, return "not specified"

4. Salary
- Extract explicit minimum and maximum salary for this role
- If only one salary exists, use it as both min and max
- Identify currency code
- salary_period must be one of: year, day, hour, month, or ""
- Keep raw numeric salary values only
- Do not use years of experience, dates, ids, percentages, bonus-only values, benefits budgets, employee counts, revenue, or unrelated numbers

5. Visa Sponsorship
- yes only if explicitly available
- no only if explicitly unavailable / right to work required
- otherwise ""

6. Contract Type
Normalize into exactly one of:
- Permanent
- FTC
- Part Time
- Freelance/Contract

7. job_category
- T&P for technical/product roles
- NonT&P for business/commercial/ops/HR/finance/legal/marketing/account/customer roles

Acceptable normalized locations:
{location_list_text}

Position name:
{job_title}

Header/meta text:
{header_text[:3000]}

Job description:
{role_body_text[:12000]}
""".strip()


def build_location_normalization_prompt(input_location: str, allowed_locations: list[str]) -> str:
    location_list_text = "\n".join(allowed_locations[:2500])

    return f"""
Output only the chosen normalized location exactly as it appears in the acceptable locations list, or output Unknown.

You are given a position location string and a list of acceptable normalized locations.

Rules:
1. If multiple acceptable locations match, select the most specific one.
2. If none match, output Unknown.
3. If the input location does not exactly exist in the list, choose the most appropriate broader location.
4. If the input is a small town or village not in the list, choose the closest broader matching location from the list.

Input location:
{input_location}

Acceptable normalized locations:
{location_list_text}
""".strip()


def build_salary_normalization_prompt(salary_text: str, predefined_salaries: list[int]) -> str:
    salaries_text = ", ".join(str(x) for x in predefined_salaries[:5000])

    return f"""
Output only one line in this exact format:
[min salary] - [max salary] [currency code]

You are given a salary as free text and a predefined list of standard salary amounts.

Tasks:
1. For each extracted number, find the closest matching value from the predefined salary list.
2. If only one salary number is present, use it as both minimum and maximum.
3. Identify the currency code.
4. Never output a number not in the predefined list.

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

Extract salary only if it is explicitly stated for this role.

Rules:
- If one amount only, use same value for min and max
- Keep raw explicit numeric salary values only
- salary_period must be one of: year, day, hour, month, or ""
- Do not use years of experience, dates, ids, percentages, employee counts, revenue, bonus-only values, equity-only values, or unrelated numbers

Position name:
{job_title}

Header/meta text:
{header_text[:2500]}

Job description:
{role_body_text[:10000]}
""".strip()


def build_job_titles_prompt(position_name: str, description: str, allowed_job_titles: list[str]) -> str:
    job_titles_text = ", ".join(allowed_job_titles[:3000])

    return f"""
Return valid JSON only:
{{
  "job_titles": ["..."]
}}

You will receive two inputs: position name and job description.

Task:
- If the position name exactly matches one job title from the predefined list, return only that one.
- If the title is unclear or ambiguous, select up to the top 3 most appropriate job titles from the predefined list.
- Only choose job titles that exactly exist in the predefined list.
- Order from most to least appropriate.
- If no suitable title exists, return an empty array.

Important:
- Business Development Representative / Sales Development Representative / BDR / SDR -> prefer SDR / BDR
- Do not overuse System Engineer
- Do not map marketing, HR, finance, PMO, customer success, account, or BDR roles to System Engineer
- Use both title and description together

Predefined list of job titles:
{job_titles_text}

Position name:
{position_name}

Job description:
{description[:12000]}
""".strip()


def build_seniority_prompt(position_name: str, description: str) -> str:
    return f"""
Return valid JSON only:
{{
  "seniorities": ["entry|junior|mid|senior|lead|leadership"]
}}

Determine seniority using:
entry, junior, mid, senior, lead, leadership

Rules:
- If title contains head of, director, engineering manager, or similar, choose leadership
- If title clearly signals junior/senior/lead, include that
- If unclear, use title + description
- Order must always be: entry, junior, mid, senior, lead, leadership
- Output lowercase only
- If no suitable seniority is found, return an empty array

Position name:
{position_name}

Job description:
{description[:10000]}
""".strip()


def build_contract_type_prompt(text: str) -> str:
    return f"""
Output only one of these exact values:
Permanent
FTC
Part Time
Freelance/Contract

Rules:
1. If multiple types are mentioned, choose one using this priority:
Permanent > FTC > Part Time > Freelance/Contract
2. Map:
- Permanent -> permanent, full time, full-time, standard
- FTC -> temporary, fixed term, fixed-term, maternity cover
- Part Time -> part time, part-time, job share
- Freelance/Contract -> freelance, contract, contracting
3. If no valid type is found, output empty string

Input text:
{text[:10000]}
""".strip()


def build_job_description_prompt(description_text: str) -> str:
    return f"""
Return valid JSON only:
{{
  "status": "unused"
}}

Compatibility prompt only.

Input text:
{description_text[:1000]}
""".strip()


def build_skills_full_prompt(role_category: str, description: str, candidate_skills: list[str], allowed_skills: list[str]) -> str:
    allowed_skills_text = ", ".join(allowed_skills[:4000])
    candidate_skills_text = ", ".join(candidate_skills[:100])

    return f"""
Return valid JSON only:
{{
  "role_category": "T&P" or "NonT&P",
  "skills": ["..."]
}}

Rules:
- Echo role_category exactly
- Use only skills from the allowed list
- Prefer concrete, clearly evidenced skills from the description
- Candidate skills are provided as a primary hint
- Never output a skill not in the allowed list
- Do not use footer/legal/navigation/marketing boilerplate

role_category:
{role_category}

candidate_skills:
{candidate_skills_text}

Allowed skills:
{allowed_skills_text}

Job description:
{description[:12000]}
""".strip()


def build_skills_additional_prompt(
    role_category: str,
    description: str,
    existing_skills: list[str],
    candidate_skills: list[str],
    allowed_skills: list[str],
) -> str:
    allowed_skills_text = ", ".join(allowed_skills[:4000])
    existing_skills_text = ", ".join(existing_skills[:100])
    candidate_skills_text = ", ".join(candidate_skills[:100])

    return f"""
Return valid JSON only:
{{
  "role_category": "T&P" or "NonT&P",
  "additional_skills": ["..."]
}}

Rules:
- Echo role_category exactly
- Do not repeat any existing skill
- Use only skills from the allowed list
- Candidate skills are a primary hint
- Infer additional skills from the description only when clearly supported
- Do not use footer/legal/navigation/marketing boilerplate

role_category:
{role_category}

existing_skills:
{existing_skills_text}

candidate_skills:
{candidate_skills_text}

Allowed skills:
{allowed_skills_text}

Job description:
{description[:12000]}
""".strip()


def build_skills_prompt(role_category: str, description: str, exact_skills: list[str], allowed_skills: list[str]) -> str:
    return build_skills_full_prompt(
        role_category=role_category,
        description=description,
        candidate_skills=exact_skills,
        allowed_skills=allowed_skills,
    )
