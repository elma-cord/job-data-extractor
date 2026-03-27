def build_relevance_prompt(job_title: str, role_context_text: str) -> str:
    return f"""
Return valid JSON only:
{{
  "role_relevance": "Relevant" or "Not relevant",
  "job_category": "T&P" or "NonT&P",
  "role_relevance_reason": "short reason"
}}

You are the FIRST screening step for a recruiting workflow.

You will receive:
1. a position name
2. the CURRENT PROVIDED JOB DESCRIPTION / ROLE CONTEXT

Important:
- The provided description is the primary source of truth.
- Do not assume any information not supported by the provided title + description.
- Do not guess from company name alone.
- Ignore cookie banners, legal boilerplate, privacy text, marketing text, navigation text, partner lists, and unrelated company text.

Your tasks:
1. Decide whether the role is Relevant or Not relevant
2. Decide whether the role is T&P or NonT&P
3. Give a concise reason

If a role is Not relevant, the workflow will stop after this step.

RELEVANT ROLE FAMILIES
Relevant roles include these exact families and close synonyms/specialisations:

Account Director, Account Executive, AI Engineer, Automation Engineer, Back End, BI Developer, Big Data Engineer, Brand Marketing, Business Analyst, Business Development Manager, Business Operations, CDO, CFO, Chief of Staff, CIO, CLO, Cloud Engineer, CMO, Computer Vision Engineer, Content Marketing, COO, Copywriting, CPO, CRM Developer, CRM Manager, CRO, CSM / Account Manager, CSO, CTO, Customer Operations, Customer Service Representative, Customer Support, Data Architect, Data Engineer, Data Scientist, Data / Insight Analyst, Database Engineer, Deep Learning Engineer, Demand / Lead Generation, Developer in Test, DevOps Engineer, Digital Marketing, Embedded Developer, Engineering Manager, Events and Community, Executive Assistant, Finance / Accounting, Founder, FP&A, Front End, Full Stack, Games Designer, Games Developer, Generalist Marketing, Graphic Designer, Graphics Developer, Growth Marketing, Head of Customer, Head of Data, Head of Design, Head of Engineering, Head of Finance, Head of HR, Head of Infrastructure, Head of Marketing, Head of Operations, Head of Product, Head of QA, Head of Sales, Human Resources, Implementation Manager, Integration Developer, Legal, Machine Learning Engineer, Marketing Analyst, Mobile Developer, Network Engineer, Operations, Partnerships, Penetration Tester, People Ops, Performance Marketing, PR / Communications, Product Manager, Product Marketing, Product Owner, Project Manager, QA Automation Tester, QA Manual Tester, Quality Assurance, Quantitative Developer, Renewals Manager, Research Engineer, RevOps, Risk and Compliance, Sales Engineer, Sales Operations, Scrum Master, SDR / BDR, Security, Security Engineer, SEO Marketing, Site Reliability Engineer, Social Media Marketing, Solutions Engineer, Support Engineer, System Administrator, System Engineer, Talent Acquisition, Technical Architect, Technical Director, Technical Writer, Testing Manager, UI Designer, UI/UX Designer, UX Designer, UX Researcher, Videography, VP of Engineering.

ALSO TREAT THESE AS RELEVANT
- Internal recruitment / TA / HR / People roles
- IT support / infrastructure / support engineering roles
- Business operations / programme / PMO / project / transformation / change roles
- National Account Manager / Key Account Manager / Account Manager / Customer Success roles
- Customer operations / renewals / implementation / partnerships roles when clearly business-facing and relevant
- Finance, accounting, legal, risk, compliance, revops, sales ops roles
- Executive Assistant / Chief of Staff / Founder / C-level roles where clearly business/corporate and otherwise allowed

CLEARLY NOT RELEVANT
If the role is clearly outside tech or business functions, mark Not relevant even if some wording overlaps.

Examples:
teacher, nurse, waiter, chef, cleaner, warehouse operative, driver, construction worker, civil engineer, retail associate, electrician, mechanical engineer in plant/manufacturing context, microbiology roles, maritime roles, injection molding roles, beauty brand store roles.

EXCLUDED AREAS
Exclude any roles related to:
- construction
- civil engineering
- retail
- electrical
- mechanical
- manufacturing
- microbiology
- maritime
- injection molding
- beauty brands

LOCATION RULES
The role is Relevant only if the working location setup is allowed based on the PROVIDED DESCRIPTION.

Allowed:
a) United Kingdom: onsite, hybrid, or remote
b) Ireland: remote only
c) Europe: only if explicitly Remote Europe or Remote EMEA
d) Remote Global / Worldwide: allowed only if not restricted to excluded regions

Detailed rules:
- Accept UK, United Kingdom, Great Britain, England, Scotland, Wales, London, and other UK places as UK.
- For Ireland, if the role is onsite or hybrid, mark Not relevant.
- For Europe, roles based in a specific European country such as Germany, France, Spain, Netherlands, etc. are Not relevant unless explicitly Remote Europe or Remote EMEA.
- Remote Global / Worldwide is allowed unless the ad specifies or implies Asia-only, Africa-only, Remote APAC, Remote LATAM, USA-only, Canada-only, North America, or any clearly excluded region.
- If the role clearly mentions USA, Canada, or another non-allowed region and there is no evidence it can be done from an allowed region, mark Not relevant.
- If salary is stated in USD or CAD and there is no evidence the role can be worked from an allowed region, that is evidence for Not relevant.
- If multiple locations are listed and at least one is allowed while others are generic like remote/anywhere, treat as Relevant only if the contract clearly allows working fully from the allowed location.
- If location is missing or unclear in the provided description, do NOT mark Not relevant for that reason alone. Judge based on function first.

LANGUAGE RULE
- If the role clearly requires any language other than English, mark Not relevant.
- Jobs requiring English only, or not mentioning another mandatory language, can still be Relevant.
- Do not reject a job just because another language is described as a bonus/preferred unless it is clearly required.

T&P VS NonT&P
- T&P includes software development, engineering, product management, data science, data engineering, IT, UX/UI, QA, DevOps, infrastructure, network, security, support engineering, cloud, systems, solutions engineering, technical architecture, and closely related technical/product roles.
- NonT&P includes other relevant business roles such as HR, talent, people, operations, PMO, programme, project, transformation, finance, accounting, legal, risk, compliance, marketing, partnerships, executive assistant, chief of staff, account / customer / renewals / implementation roles.

DECISION ORDER
1. Identify likely role family from title and description.
2. Identify location and remote setup from the provided description if present.
3. Apply the location rules only where supported by the description.
4. Apply the language rule.
5. If Relevant, classify T&P or NonT&P.
6. Return final result.

REASON RULES
- Keep the reason short and concrete.
- Mention the main deciding factor only.
- Good examples:
  - "UK-based People role fits allowed functions."
  - "Germany onsite role is outside allowed locations."
  - "Role requires French language."
  - "Manufacturing mechanical role is outside target functions."
  - "Remote Ireland role is allowed and matches finance function."
  - "Relevant PMO / operations role."

Position name:
{job_title}

Current provided job description / role context:
{role_context_text[:16000]}
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

You are extracting fields from the CURRENT PROVIDED JOB DESCRIPTION.
The provided description is the main source of truth.
Be strict and grounded. Do not guess.

If a field is not clearly supported by the provided text, return an empty string for that field.
Exception:
- remote_preferences may return "not specified"
- remote_days may return "not specified"

1. Position Location
a) Read the provided description carefully to find the work location.
b) Focus on locations near keywords like:
   "location:", "based in", "office in", "work location", "remote in", "country", "city", "based at"
c) If multiple locations appear, select the most specific one that seems directly tied to the role.
d) Match exactly one entry from the provided list of acceptable normalized locations where possible.
e) If the extracted location does not exactly exist in the list, select the closest broader location from the list.
f) If no supported match is found, return "".
g) Do not invent a location from company HQ, company name, currency alone, or generic regional references.

2. Position Remote Preferences
a) Extract remote working preferences only if supported by the provided description.
b) Normalize all variants of onsite as exactly onsite.
c) Normalize all hybrid variants as exactly hybrid.
d) Normalize all remote variants as exactly remote.
e) List extracted preferences in this order only: onsite, hybrid, remote.
f) Separate multiple preferences with comma + space.
g) If none of these are clearly supported, return "not specified".
h) It is valid to return combinations such as:
   - onsite, hybrid
   - hybrid, remote
   - onsite, remote
   - onsite, hybrid, remote
i) Do not collapse multiple clearly supported preferences into one.

3. Remote Days
Return only:
- the highest number of remote work days in a 5-day week, as a string number, or
- "not specified"

Rules:
a) Return a number only if remote days or office days are explicitly stated.
b) If a range is given for remote days, return the highest number.
c) If office days are stated, calculate remote days as 5 - office days.
d) If office days are given as a range, return the highest possible remote days.
e) Examples:
   - 2 days in office => 3
   - 1-2 days in office => 4
   - 2-3 remote days => 3
   - 60% office based => 2
f) If text says fully remote / remote every day, return "not specified".
g) If no remote work is allowed, return "not specified".
h) If days are mentioned but unclear whether they are remote or office days, return "not specified".

4. Salary
a) Extract minimum and maximum salary numbers only if explicitly stated for this role.
b) If only one salary is present, treat it as both min and max.
c) Identify currency code from text.
d) salary_period must be one of: year, day, hour, month, or "".
e) Keep raw explicit numeric salary values only.
f) Do not use years of experience, dates, ids, percentages, bonus-only numbers, benefits budgets, employee counts, revenue, or unrelated numbers.
g) If salary is ambiguous or not clearly tied to this role, return empty strings.

5. Visa Sponsorship
- Return yes, no, or ""
- yes only if sponsorship is explicitly available
- no only if the ad explicitly says sponsorship is unavailable / right to work required / no sponsorship
- otherwise ""

6. Contract Type
Identify and normalize into exactly one of:
- Permanent
- FTC
- Part Time
- Freelance/Contract

Rules:
- If multiple types are mentioned, choose one using this priority:
  Permanent > FTC > Part Time > Freelance/Contract
- Map synonyms:
  - Permanent -> permanent, full time, full-time, standard
  - FTC -> temporary, fixed term, fixed-term, maternity cover
  - Part Time -> part time, part-time, job share, job-share
  - Freelance/Contract -> freelance, contract, contracting
- If no valid type is found, return ""

7. job_category
- If the category is obvious from the role, return T&P or NonT&P.
- If unclear, return "".
- Do not force a category.

Allowed normalized locations:
{location_list_text}

Position name:
{job_title}

Header/meta text:
{header_text[:5000]}

Current provided job description:
{role_body_text[:14000]}
""".strip()


def build_location_normalization_prompt(input_location: str, allowed_locations: list[str]) -> str:
    location_list_text = "\n".join(allowed_locations[:2500])

    return f"""
Output only the chosen normalized location exactly as it appears in the acceptable locations list, or output Unknown.

You are given an input location string and a list of acceptable normalized locations.

Your task:
1. Identify the best matching normalized location from the list.
2. If multiple acceptable locations match, choose the most specific one.
3. If the input location is a small town or village not in the list, choose the closest broader matching location from the list.
4. If none match, output Unknown.
5. Output only the chosen normalized location, with no extra text.

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

You are given a salary as free text and a predefined list of valid salary amounts.

Tasks:
1. Extract minimum and maximum salary values from the input text.
2. Round each extracted value to the closest value from the predefined salary list.
3. If only one salary number is present, treat it as both minimum and maximum.
4. Identify the currency code from the salary text.
5. Never output a salary that is not in the predefined list.
6. If you cannot identify a valid salary, output:
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

Extract salary only when it is explicitly stated for this role in the CURRENT PROVIDED JOB DESCRIPTION.

Rules:
- salary_period must be one of: year, day, hour, month, or ""
- If one amount only, use the same value for min and max
- If a range is given, extract the real minimum and maximum
- Keep raw explicit numeric salary values
- Do not round
- Do not normalize
- Do NOT use:
  - years of experience
  - dates
  - ids
  - percentages
  - office attendance percentages unless they are clearly salary-related
  - employee counts
  - revenue figures
  - bonus only figures
  - equity only figures
  - benefits budgets
  - unrelated numbers from page chrome
- If salary is ambiguous or not clearly tied to this role, return empty strings
- If multiple currencies are mentioned, extract only the one clearly tied to compensation

Position name:
{job_title}

Header/meta text:
{header_text[:3000]}

Current provided job description:
{role_body_text[:10000]}
""".strip()


def build_job_titles_prompt(position_name: str, description: str, allowed_job_titles: list[str]) -> str:
    job_titles_text = ", ".join(allowed_job_titles[:3000])

    return f"""
Return valid JSON only:
{{
  "job_titles": ["..."]
}}

You will receive:
1. a position name
2. the CURRENT PROVIDED JOB DESCRIPTION

Task:
Choose up to 3 best matching job titles from the predefined list.

Rules:
- Use only job titles from the predefined list.
- If the position name exactly matches one predefined title, return only that one.
- If the position name is clear and strongly points to one allowed title, prefer just one result.
- If the position name is unclear or ambiguous, choose up to the top 3 most appropriate job titles from the predefined list.
- Prefer functional fit over literal wording.
- Do not force 3 titles if 1 or 2 are clearly enough.
- Do not return loosely related titles.
- Do not guess from company background.
- Output titles ordered from most to least appropriate.
- If no suitable match exists, return an empty array.

Important mappings:
- National Account Manager / Key Account Manager / Account Manager / Customer Success style roles should include "CSM / Account Manager" when available
- Systems / infrastructure / Kubernetes / Linux / OpenShift / virtualisation-heavy roles should prefer "System Engineer"
- CRM / loyalty / campaign / customer analytics roles should prefer "Marketing Analyst" and/or "Data / Insight Analyst"
- Do not force "Account Executive" unless clearly new-business / hunter / AE style
- PMO / Programme / Change / Transformation roles should prefer the closest operations/project/programme title available
- People Ops / Talent / HR / Recruiter roles should map to the closest HR / TA title available
- Finance / FP&A / Accounting roles should map to the closest finance title available

Predefined job titles:
{job_titles_text}

Position name:
{position_name}

Current provided job description:
{description[:12000]}
""".strip()


def build_seniority_prompt(position_name: str, description: str) -> str:
    return f"""
Return valid JSON only:
{{
  "seniorities": ["entry|junior|mid|senior|lead|leadership"]
}}

You will receive:
1. a position name
2. the CURRENT PROVIDED JOB DESCRIPTION

Task:
Choose up to 3 seniority values using only:
entry, junior, mid, senior, lead, leadership

Rules:
- Keep order exactly as: entry, junior, mid, senior, lead, leadership
- Seniority values must be lowercase
- Use only values strongly supported by the title and description
- Do not over-tag
- Do not add levels just to fill space

Strong title rules:
- If title includes Head, Director, VP, Chief, C-level => leadership
- If title includes Engineering Manager or similar strong people-management title => leadership
- Plain manager titles usually mean senior and/or lead, not mid
- Do not include mid for manager titles unless the role is clearly junior/assistant/associate-manager style

Role-specific rules:
- PMO / Programme / Project / Operations / Account Manager / Customer Success / People roles:
  - do not include mid unless clearly supported
  - include senior for broad ownership, governance, complex stakeholder management, forecasting, end-to-end responsibility, or strategic oversight
  - include lead for team leadership, workstream ownership, cross-functional leadership, or major stakeholder coordination
- IC technical roles:
  - junior titles => junior
  - senior titles => senior
  - staff/principal/lead titles often => senior and/or lead
- Assistant / Associate / Graduate / Intern => entry and/or junior depending on wording

Experience guidance:
- 0-1 years => entry, junior
- 2 years => junior, mid
- 3-5 years => senior, lead
- 5+ years => senior, lead
- If the experience is a range, include all seniority levels that apply across that range
- Only use experience guidance when title is not already clearer

Other rules:
- If title explicitly says junior / senior / lead, include that
- If unclear, return empty array
- Avoid adding mid just to fill space

Position name:
{position_name}

Current provided job description:
{description[:12000]}
""".strip()


def build_contract_type_prompt(text: str) -> str:
    return f"""
Output only one of these exact values:
Permanent
FTC
Part Time
Freelance/Contract

Or output an empty string if no valid type is found.

Task:
Identify and normalize the contract type.

Rules:
1. If multiple types are mentioned, choose only one using this priority:
   Permanent > FTC > Part Time > Freelance/Contract
2. Map synonyms:
   - Permanent -> permanent, full time, full-time, standard
   - FTC -> temporary, fixed term, fixed-term, maternity cover
   - Part Time -> part time, part-time, job share, job-share
   - Freelance/Contract -> freelance, contract, contracting
3. Output only one category, with no extra words.

Input text:
{text[:10000]}
""".strip()


def build_job_description_prompt(description_text: str) -> str:
    return f"""
Return valid JSON only:
{{
  "status": "unused"
}}

This prompt is kept only for compatibility with older imports.
Do not transform, rewrite, clean, format, or summarize the job description.
The current workflow no longer outputs a generated job_description column.

Input text:
{description_text[:2000]}
""".strip()


def build_skills_full_prompt(role_category: str, description: str, candidate_skills: list[str], allowed_skills: list[str]) -> str:
    allowed_skills_text = ", ".join(allowed_skills[:4000])
    candidate_skills_text = ", ".join(candidate_skills[:100])

    return f"""
Return valid JSON only. No markdown. No commentary.

Hard rules:
- role_category is PROVIDED as input. You MUST echo it exactly as given ("T&P" or "NonT&P"). Do not change it.
- skills MUST be an array with 2 to 10 items. Try hard to return at least 2 when clearly supported.
- skills MUST be chosen ONLY from the correct allowed skills list.
- Prefer concrete, clearly evidenced skills from the CURRENT PROVIDED JOB DESCRIPTION.
- candidate_skills is provided: use it as the PRIMARY source, but if it has fewer than 2 items, infer additional skills from the description.
- NEVER output a skill not present in the allowed list.
- Before finalizing, verify each returned skill appears exactly in the allowed list. If not, remove it.
- Use only skills supported by the description or candidate_skills.
- Do not use footer text, navigation text, legal text, marketing text, partner lists, or unrelated company content.
- Do not invent broad generic skills if the wording is not supported.

Output schema:
{{
  "role_category": "T&P" or "NonT&P",
  "skills": ["..."]
}}

role_category:
{role_category}

candidate_skills:
{candidate_skills_text}

Allowed skills:
{allowed_skills_text}

Current provided job description:
{description[:14000]}
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
Return valid JSON only. No markdown. No commentary.

Hard rules:
- role_category is PROVIDED as input. You MUST echo it exactly as given ("T&P" or "NonT&P"). Do not change it.
- You MUST return additional_skills as an array with 2 to 5 items. Try hard to return at least 2 when strongly supported.
- You MUST NOT repeat any skills that are already present in existing_skills.
- You MUST choose skills ONLY from the correct allowed skills list.
- candidate_skills is provided: use it as the primary source when relevant.
- If candidate_skills is insufficient, infer from the CURRENT PROVIDED JOB DESCRIPTION but still only from the allowed list.
- Before finalizing, verify each returned skill exists exactly in the allowed list. Remove any that do not.
- Do not output existing_skills again.
- Do not use footer text, navigation text, legal text, marketing text, partner lists, or unrelated company content.
- Do not invent broad generic skills if the wording is not supported.

Output schema:
{{
  "role_category": "T&P" or "NonT&P",
  "additional_skills": ["..."]
}}

role_category:
{role_category}

existing_skills:
{existing_skills_text}

candidate_skills:
{candidate_skills_text}

Allowed skills:
{allowed_skills_text}

Current provided job description:
{description[:14000]}
""".strip()


def build_skills_prompt(role_category: str, description: str, exact_skills: list[str], allowed_skills: list[str]) -> str:
    """
    Compatibility wrapper for the current pipeline.
    If exact_skills is empty, the workflow should ideally use build_skills_full_prompt.
    If exact_skills already has one or more items, the workflow should ideally use build_skills_additional_prompt.
    This wrapper remains so current imports do not break immediately.
    """
    return build_skills_full_prompt(
        role_category=role_category,
        description=description,
        candidate_skills=exact_skills,
        allowed_skills=allowed_skills,
    )
