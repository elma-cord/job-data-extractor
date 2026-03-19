def build_relevance_prompt(job_title: str, role_context_text: str) -> str:
    return f"""
You will receive two inputs: position name and job description.

Return ONLY valid JSON with exactly these keys:
role_relevance
role_relevance_reason

Rules:
- role_relevance must be exactly "Relevant" or "Not relevant"
- role_relevance_reason must be concise and specific

Relevant roles match any in this list or close synonyms/specializations:

Account Director, Account Executive, AI Engineer, Automation Engineer, Back End, BI Developer, Big Data Engineer, Brand Marketing, Business Analyst, Business Development Manager, Business Operations, CDO, CFO, Chief of Staff, CIO, CLO, Cloud Engineer, CMO, Computer Vision Engineer, Content Marketing, COO, Copywriting, CPO, CRM Developer, CRM Manager, CRO, CSM / Account Manager, CSO, CTO, Customer Operations, Customer Service Rep, Customer Support, Data Architect, Data Engineer, Data Scientist, Data / Insight Analyst, Database Engineer, Deep Learning Engineer, Demand / Lead Generation, Developer in Test, DevOps Engineer, Digital Marketing, Embedded Developer, Engineering Manager, Events & Community, Executive Assistant, Finance / Accounting, Founder, FP&A, Front End, Full Stack, Games Designer, Games Developer, Generalist Marketing, Graphic Designer, Graphics Developer, Growth Marketing, Head of Customer, Head of Data, Head of Design, Head of Engineering, Head of Finance, Head of HR, Head of Infrastructure, Head of Marketing, Head of Operations, Head of Product, Head of QA, Head of Sales, Human Resources, Implementation Manager, Integration Developer, Legal, Machine Learning Engineer, Marketing Analyst, Mobile Developer, Network Engineer, Operations, Partnerships, Penetration Tester, People Ops, Performance Marketing, PR / Communications, Product Manager, Product Marketing, Product Owner, Project Manager, QA Automation Tester, QA Manual Tester, Quality Assurance, Quantitative Developer, Renewals Manager, Research Engineer, RevOps, Risk & Compliance, Sales Engineer, Sales Operations, Scrum Master, SDR / BDR, Security, Security Engineer, SEO Marketing, Site Reliability Engineer, Social Media Marketing, Solutions Engineer, Support Engineer, System Administrator, System Engineer, Talent Acquisition, Technical Architect, Technical Director, Technical Writer, Testing Manager, UI Designer, UI/UX Designer, UX Designer, UX Researcher, Videography, VP of Engineering.

Important allow rules:
- Talent Acquisition roles ARE relevant.
- Human Resources roles ARE relevant.
- People Ops roles ARE relevant.
- Recruitment roles are relevant when they are internal/corporate talent acquisition, recruiting, people, HR, or employer-branding functions.
- IT support / infrastructure / support engineering roles ARE relevant and should not be excluded just because they are support-oriented or customer-facing.
- Business/program/change/transformation/PMO/operations roles can be relevant under Business Operations / Project Manager / Operations families.
- National Account Manager / Key Account Manager / Account Manager / Customer Success Manager style roles can be relevant under CSM / Account Manager.

Reject only when the role is clearly outside allowed business/tech functions, such as:
teacher, nurse, waiter, chef, construction worker, civil engineer, electrician, mechanical engineer in manufacturing/plant context, manufacturing operator, maritime crew, microbiology lab role, beauty retail staff, injection molding technician, warehouse operative, driver, cleaner.

Location rules:
- Relevant only if the working location is allowed:
  a) United Kingdom: onsite, hybrid, or remote allowed
  b) Ireland: only remote allowed
  c) Europe: only explicitly Remote Europe or Remote EMEA allowed
  d) Remote Global allowed unless explicitly limited to excluded regions
  e) If the role clearly points to USA/Canada/other non-allowed region with no evidence of UK/allowed-region work, mark Not relevant
- Accept UK cities/regions as UK
- If multiple locations are listed, use the stated working arrangement and whether at least one valid allowed working option clearly exists

Language rules:
- If the role clearly requires a non-English language, mark Not relevant
- English-only roles are fine

Decision priority:
1. First determine whether the role family is in the allowed list or a close synonym
2. Then determine whether the location/remote setup is allowed
3. Only then decide final relevance

Be careful:
- Do NOT mark a role Not relevant simply because it is recruitment/HR if it is clearly Talent Acquisition / Human Resources / People Ops
- Do NOT mark a role Not relevant simply because it is support/infrastructure if it is clearly IT / systems / cloud / network / support engineering
- Do NOT mark a role Not relevant as “non-informative” if the extracted text clearly contains an actual engineering/business role title or responsibilities
- Use the actual job title and role duties, not generic company overview text

Input position name:
{job_title}

Input description:
{role_context_text[:22000]}
""".strip()


def build_core_fields_prompt(
    job_title: str,
    header_text: str,
    role_body_text: str,
    allowed_locations: list[str],
) -> str:
    location_list_text = "\n".join(allowed_locations[:3000])

    return f"""
You will receive:
1. position name
2. header/meta text
3. role description text

Return ONLY valid JSON with exactly these keys:
job_category
job_location
remote_preferences
remote_days
visa_sponsorship
job_type

Rules for job_category:
- Output exactly one of: "T&P", "NonT&P"
- T&P includes software development, engineering, product, data, IT, UX/UI, QA, DevOps, infrastructure, system administration, support engineering, network engineering, solutions engineering, cloud/platform engineering and similar technical roles.
- T&P ALSO includes IT support, helpdesk, desktop support, 2nd line / 3rd line, infrastructure support, MSP engineering, field engineering, and technical customer-environment support roles.
- NonT&P is everything else that is still relevant.

Rules for job_location:
- Use header/meta text first, then role description.
- Prefer the most specific visible location over a broad country fallback.
- If multiple valid locations are listed, choose the most specific clear normalized location from the allowed list.
- If the page says “Remote working within the UK”, “UK remote”, or similar, return the correct UK location from the allowed list, not blank.
- Only return "" if there is truly no clear location evidence.
- Prefer true header/location labels over incidental mentions elsewhere in the body.

Rules for remote_preferences:
- Allowed values only: onsite, hybrid, remote, or ""
- Use role-specific/header evidence first.
- home based counts as remote.
- daily field travel does NOT mean onsite only.
- Do not let generic company-wide smart-working text override a clearer role-specific line.
- Wording like “office attendance requirement”, “1-2 days per week in office”, or “60% office based” means the role is hybrid unless the role also clearly says fully onsite.
- If the role-specific line says remote in UK / remote working within the UK, return remote only.

Rules for remote_days:
- Return only a single number as a string, or ""
- Only return a number when explicit office/remote day evidence is present.
- For office ranges such as 1-2 days in office per week, return the minimum guaranteed remote-days value, which is 3.
- For office percentages like 60% office based, convert approximately to a 5-day week and return the remote-day remainder.
- Never infer a number from hybrid, flexible working, compressed hours, Friday half-day, or generic work/life balance text.
- Do not infer from company-wide smart-working statements unless they are explicitly role-specific.

Rules for visa_sponsorship:
- Return only yes, no, or ""

Rules for job_type:
- Return only one of:
  Permanent
  FTC
  Part Time
  Freelance/Contract
  or ""
- Priority: Permanent > FTC > Part Time > Freelance/Contract

Allowed normalized locations:
{location_list_text}

Input position name:
{job_title}

Header/meta text:
{header_text[:6000]}

Role description:
{role_body_text[:24000]}
""".strip()


def build_salary_prompt(job_title: str, header_text: str, role_body_text: str) -> str:
    return f"""
You will receive:
1. position name
2. header/meta text
3. role description text

Return ONLY valid JSON with exactly these keys:
salary_min
salary_max
salary_currency
salary_period

Rules:
- Extract salary ONLY when compensation/rate is explicitly stated for this role.
- salary_period must be exactly one of: year, day, hour, month, or ""
- If a range is stated, keep both min and max.
- If only one amount is stated, put the same value in both salary_min and salary_max.
- Do not invent salary from unrelated numbers.
- Do not use years of experience, employee counts, dates, percentages, office attendance percentages, standards, ids, revenue, headcount, or benefits.
- Do not round or normalize the numeric value.
- Keep only raw explicit salary numbers, without commas if possible.
- If no clear salary is present, return empty strings for all four keys.

Input position name:
{job_title}

Header/meta text:
{header_text[:6000]}

Role description:
{role_body_text[:24000]}
""".strip()


def build_job_titles_prompt(position_name: str, description: str, allowed_job_titles: list[str]) -> str:
    job_titles_text = ", ".join(allowed_job_titles[:3000])

    return f"""
You will receive two inputs: position name and description.

Task:
Choose up to 3 best matching job titles from the predefined list.

Rules:
- Analyze both the position name and description together.
- Use ONLY job titles from the predefined list.
- If the position name exactly matches one predefined job title, return only that single title.
- If the position name is unclear or broader than the predefined list, choose up to the top 3 most appropriate job titles from the predefined list.
- Important mapping rule:
  - National Account Manager / Key Account Manager / Account Manager / Customer Success Manager style roles should include "CSM / Account Manager" when that exists in the predefined list.
  - DTN Software Engineer / Systems Engineer / infrastructure-heavy Kubernetes/Linux/virtualisation roles should prefer "System Engineer" and can also include "DevOps Engineer" if strongly supported.
  - CRM / loyalty / campaign / customer analytics roles should prefer "Marketing Analyst" and/or "Data / Insight Analyst" over "Business Analyst" or "Data Scientist" unless those are clearly better fits.
  - Do not force "Account Executive" unless the role clearly looks AE / sales-hunter / new-business focused.
- Return a JSON object with one key only:
  "job_titles": ["...", "..."]
- job_titles must contain exact strings from the predefined list only.
- Order from most appropriate to least appropriate.
- If no suitable match exists, return an empty array.
- Do NOT leave job_titles empty if there is a clear best-fit title in the predefined list.

Predefined job titles:
{job_titles_text}

Position name:
{position_name}

Description:
{description[:18000]}
""".strip()


def build_seniority_prompt(position_name: str, description: str) -> str:
    return f"""
You will receive two inputs: position name and description.

Determine seniority using only this allowed list:
entry, junior, mid, senior, lead, leadership

Rules:
1. First analyze the position name.
2. If title includes leadership indicators such as "head of", "director", "technical director", "engineering manager", "vp", "chief", return leadership only.
3. Plain manager titles usually indicate senior and/or lead, not mid.
4. For manager titles such as PMO Manager, Programme Manager, Project Manager, Operations Manager, Account Manager:
   - do NOT include mid unless the title/description clearly indicates junior/associate/assistant level.
   - include lead when ownership / stakeholder management / coordination / governance / team leadership is present.
   - include senior when the role spans broad ownership, full lifecycle responsibility, governance, forecasting, oversight, or complex stakeholder management.
5. If the title clearly contains seniority like junior, senior, lead, return the appropriate value(s).
6. If ambiguous, analyze both title and description together and return up to 3 most appropriate seniority levels.
7. Output must be JSON with one key only:
   "seniorities": ["...", "..."]
8. Use only these exact lowercase values:
   entry, junior, mid, senior, lead, leadership
9. Keep order exactly:
   entry, junior, mid, senior, lead, leadership
10. If experience is a range, include all applicable seniorities across the range:
   - 0-1 years -> entry, junior
   - 2 years -> junior, mid
   - 3-5 years -> senior
   - 5+ years -> senior, lead
11. If managerial/team-management/coaching/escalation-point/ownership responsibilities are clearly present, include lead.
12. If nothing suitable is identified, return an empty array.

Position name:
{position_name}

Description:
{description[:18000]}
""".strip()


def build_skills_prompt(role_category: str, description: str, exact_skills: list[str], allowed_skills: list[str]) -> str:
    allowed_skills_text = ", ".join(allowed_skills[:5000])
    exact_skills_text = ", ".join(exact_skills)

    return f"""
You are a strict skills tagger for a recruiting workflow.

Return valid json only. No markdown. No commentary.

Goal:
- Use ONLY the supplied role-focused description text.
- Keep the exact-match skills already found.
- Add only clearly evidenced missing skills from the description.
- Do not use page chrome, partner lists, footer text, platform names, legal text, company marketing, or brand badges.
- Do not infer adjacent technologies unless they are clearly mentioned in the role-focused description.

Hard rules:
- role_category is PROVIDED as input. You MUST echo it exactly as given ("T&P" or "NonT&P"). Do not change it.
- skills MUST be chosen ONLY from the correct Allowed skills list (exact string match).
- exact_skills is the PRIMARY base set. Preserve them.
- NEVER output a skill not present in the allowed list.
- Before finalizing, verify every returned skill appears exactly in the allowed list.
- Keep exact_skills first, then add missing clearly evidenced skills.
- Return up to 10 skills total.
- If you are not sure a skill is evidenced in the text, exclude it.

Output schema:
{{
  "role_category": "T&P" or "NonT&P",
  "skills": ["..."]
}}

role_category:
{role_category}

exact_skills:
{exact_skills_text}

Allowed skills:
{allowed_skills_text}

role_focused_description:
{description[:18000]}
""".strip()
