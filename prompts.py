def build_relevance_prompt(job_title: str, role_context_text: str) -> str:
    return f"""
Return valid JSON only:
{{
  "role_relevance": "Relevant" or "Not relevant",
  "role_relevance_reason": "short reason"
}}

You are classifying whether a job should be kept for further processing.

Your decision must be strict, accurate, and grounded in the actual job title + job description.

IMPORTANT:
- This is the FIRST screening step.
- Decide only whether the job is Relevant or Not relevant.
- If uncertain, prefer the most evidence-based decision from the title and description.
- Do not guess based on company type alone.
- Do not use footer text, navigation text, legal text, cookie text, or unrelated company marketing text.

RELEVANT ROLES
Relevant roles match any in this list or close synonyms/specializations:

Account Director, Account Executive, AI Engineer, Automation Engineer, Back End, BI Developer, Big Data Engineer, Brand Marketing, Business Analyst, Business Development Manager, Business Operations, CDO, CFO, Chief of Staff, CIO, CLO, Cloud Engineer, CMO, Computer Vision Engineer, Content Marketing, COO, Copywriting, CPO, CRM Developer, CRM Manager, CRO, CSM / Account Manager, CSO, CTO, Customer Operations, Customer Service Rep, Customer Support, Data Architect, Data Engineer, Data Scientist, Data / Insight Analyst, Database Engineer, Deep Learning Engineer, Demand / Lead Generation, Developer in Test, DevOps Engineer, Digital Marketing, Embedded Developer, Engineering Manager, Events & Community, Executive Assistant, Finance / Accounting, Founder, FP&A, Front End, Full Stack, Games Designer, Games Developer, Generalist Marketing, Graphic Designer, Graphics Developer, Growth Marketing, Head of Customer, Head of Data, Head of Design, Head of Engineering, Head of Finance, Head of HR, Head of Infrastructure, Head of Marketing, Head of Operations, Head of Product, Head of QA, Head of Sales, Human Resources, Implementation Manager, Integration Developer, Legal, Machine Learning Engineer, Marketing Analyst, Mobile Developer, Network Engineer, Operations, Partnerships, Penetration Tester, People Ops, Performance Marketing, PR / Communications, Product Manager, Product Marketing, Product Owner, Project Manager, QA Automation Tester, QA Manual Tester, Quality Assurance, Quantitative Developer, Renewals Manager, Research Engineer, RevOps, Risk & Compliance, Sales Engineer, Sales Operations, Scrum Master, SDR / BDR, Security, Security Engineer, SEO Marketing, Site Reliability Engineer, Social Media Marketing, Solutions Engineer, Support Engineer, System Administrator, System Engineer, Talent Acquisition, Technical Architect, Technical Director, Technical Writer, Testing Manager, UI Designer, UI/UX Designer, UX Designer, UX Researcher, Videography, VP of Engineering.

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
The role is Relevant only if the working location setup is allowed.

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

LANGUAGE RULE
- If the role clearly requires any language other than English, mark Not relevant.
- Jobs requiring English only, or not mentioning another mandatory language, can still be Relevant.
- Do not reject a job just because another language is described as a bonus/preferred unless it is clearly required.

DECISION ORDER
1. Identify likely role family from title and description.
2. Identify location and remote setup from the description.
3. Apply the location rules.
4. Apply the language rule.
5. Return final relevance result.

REASON RULES
- Keep the reason short and concrete.
- Mention the main deciding factor only.
- Good examples:
  - "UK-based People role fits allowed functions."
  - "Germany onsite role is outside allowed locations."
  - "Role requires French language."
  - "Manufacturing mechanical role is outside target functions."
  - "Remote Ireland role is allowed and matches finance function."

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

You will receive a relevant job only. Extract the core structured fields carefully.

Rules:

1. job_category
- T&P = software, engineering, data, AI, ML, IT, infrastructure, cloud, network, devops, security, QA, product, UX/UI, systems, technical support, solutions engineering, technical architecture
- NonT&P = other relevant business roles such as HR, people, talent, operations, finance, accounting, legal, risk, compliance, marketing, partnerships, customer/account roles, PMO/project/programme/change roles, executive assistant, chief of staff
- IT support / infrastructure / support engineering roles should be T&P
- PMO / programme / operations / people / finance / legal / account / customer roles should usually be NonT&P

2. job_location
- Prefer explicit header/location labels first
- Prefer the most specific visible location
- If multiple valid locations exist, return the best primary one
- If the ad only clearly states Remote UK / UK Remote / remote in the UK, return the matching UK location only if a specific valid location is explicitly shown; otherwise return ""
- Return "" if unclear

3. remote_preferences
- Allowed only: onsite, hybrid, remote, or ""
- home based = remote
- office attendance wording usually = hybrid
- do not let generic company-wide flexibility override role-specific wording
- if role says remote with occasional office travel but still fundamentally remote, use remote
- if role says x days in office / weekly office attendance / split office-home, use hybrid

4. remote_days
- Return number as string or ""
- Only when office/remote day evidence exists
- Examples:
  - 1 office day per week => 4
  - 2 office days per week => 3
  - 1-2 office days per week => 3
  - 3 office days per week => 2
  - 60% office based => 2
- Do NOT infer from generic flexible working language, benefits, or vague hybrid wording

5. visa_sponsorship
- yes, no, or ""
- yes only if explicit sponsorship evidence exists
- no only if explicit no-sponsorship/right-to-work evidence exists
- otherwise ""

6. job_type
- Allowed values: Permanent, FTC, Part Time, Freelance/Contract, or ""
- FTC includes fixed term contract / maternity cover / temporary contract
- Freelance/Contract includes day-rate contract, contractor, freelance, interim
- Priority if multiple phrases appear: Permanent > FTC > Part Time > Freelance/Contract only when clearly role-specific

Allowed locations list for normalization support:
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

Extract salary only when it is explicitly stated for this role.

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
- Do not force 3 titles if 1 or 2 are clearly enough
- Do not return titles that are only loosely related

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

Description:
{description[:12000]}
""".strip()


def build_seniority_prompt(position_name: str, description: str) -> str:
    return f"""
Return valid JSON only:
{{
  "seniorities": ["entry|junior|mid|senior|lead|leadership"]
}}

Task:
Choose up to 3 seniority values using only:
entry, junior, mid, senior, lead, leadership

Rules:
- Keep order exactly as: entry, junior, mid, senior, lead, leadership
- Use only values strongly supported by the title and description
- Do not over-tag

Strong title rules:
- If title includes Head, Director, VP, Chief, C-level => leadership
- If title includes Engineering Manager / Manager with clear people-management responsibility, leadership can apply only when it is genuinely managerial at department/team level
- Plain manager titles usually mean senior and/or lead, not mid
- Do not include mid for manager titles unless the role is clearly junior/assistant/associate-manager style, which is rare

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
- 3-5 years => senior
- 5+ years => senior, lead
- Only use experience guidance when title is not already clearer

Other rules:
- If title explicitly says junior / senior / lead, include that
- Return empty array if genuinely unclear
- Avoid adding mid just to fill space

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

Task:
Return up to 10 skills total using only the allowed list.

Rules:
- Keep exact_skills first whenever they are valid and relevant
- Add only clearly evidenced missing skills from the description
- Use only skills from the allowed list
- Do not infer adjacent tools or technologies
- Do not infer broad umbrella skills unless directly supported
- Exclude weak guesses
- Do not use footer text, navigation text, marketing text, legal text, company boilerplate, partner lists, or unrelated content
- Prefer skills that are central to the actual responsibilities/requirements
- If the role is NonT&P, prefer business-function skills from the allowed list
- If the role is T&P, prefer technical/product/data skills from the allowed list
- If unsure, exclude rather than guess

role_category:
{role_category}

exact_skills:
{exact_skills_text}

Allowed skills:
{allowed_skills_text}

Description:
{description[:12000]}
""".strip()
