from textwrap import dedent


def build_unified_job_extraction_prompt(
    position_name: str,
    source_text: str,
    predefined_job_titles: list[str],
    predefined_locations: list[str],
    predefined_salaries: list[int],
    allowed_tp_skills: list[str],
    allowed_nontp_skills: list[str],
) -> str:
    job_titles_str = ", ".join(predefined_job_titles)
    locations_str = ", ".join(predefined_locations)
    salaries_str = ", ".join(str(x) for x in predefined_salaries)
    tp_skills_str = ", ".join(allowed_tp_skills)
    nontp_skills_str = ", ".join(allowed_nontp_skills)

    return dedent(f"""
    You are a strict structured extractor for a recruiting workflow.

    You will receive:
    1. position name
    2. source text

    Important:
    - We are talking about JOBS / POSITIONS, not candidates.
    - Return valid JSON only.
    - No markdown.
    - No explanation outside JSON.
    - Never invent facts that are not supported by the source text.
    - Use empty strings or empty arrays if not supported.
    - Use the predefined files as strict source of truth for normalized outputs.
    - If there is uncertainty, prefer fewer outputs and more accurate outputs.
    - The reason MUST match the final relevance decision.

    TASKS

    1) role_relevance
    Decide if the role is "Relevant" or "Not Relevant".

    Relevant roles match the predefined job titles list below, or close synonyms / specializations that clearly map to one of them:
    {job_titles_str}

    Also allowed when clearly genuine corporate roles:
    - finance / accounting / FP&A / tax / treasury / audit / controller / accounts payable / accounts receivable
    - business operations / program / PMO / transformation / change / analyst roles
    - account / client / customer / renewals / partnerships / implementation / customer success roles
    - marketing / growth / content / CRM / communications / product marketing / demand generation roles
    - executive assistant / chief of staff / legal / people ops / talent acquisition roles
    - technical support / IT / infrastructure / systems / network / support engineering roles when clearly technical

    Exclusions:
    - Not a real job posting
    - Educational content, learning modules, checklists, articles, guides
    - Construction, civil engineering, retail store roles, cashier, showroom/store/branch/shop-floor roles
    - Electrical / mechanical / manufacturing / plant / factory / assembly / injection molding / maritime / microbiology / beauty brand roles
    - Medical / clinical / patient care roles
    - Any role clearly outside allowed tech/business functions

    2) job_category
    Output exactly one:
    - "T&P job"
    - "Not T&P"

    T&P includes:
    software engineering, data, DevOps, QA, security, IT, support engineering, infrastructure, cloud, UX/UI, product, technical architecture, technical writing, ML/AI, etc.

    3) Location rules for relevance
    Relevant only if the role fits allowed locations:
    - United Kingdom: onsite, hybrid, or remote allowed
    - Ireland: only remote allowed
    - Europe: only explicitly remote Europe / remote EMEA allowed
    - Remote Global / worldwide allowed unless ad clearly restricts to APAC / LATAM / Africa / Asia / USA / Canada / another excluded region
    - If the job clearly indicates USA / Canada / excluded regions only, mark Not Relevant
    - If salary is only USD/CAD and nothing supports allowed regions, that is evidence for Not Relevant
    - Accept UK / Great Britain / England / Scotland / Wales / Northern Ireland / London etc. as UK
    - Ignore generic company office lists unless they clearly describe this role’s actual work location

    Very important for job_location:
    - Choose the MOST SPECIFIC acceptable normalized location from the predefined list.
    - Do NOT fall back to "UK" / "United Kingdom" if a more specific acceptable location in the list matches the city / town / area in the posting.
    - If the posting says a town/city like Stansted, Huntingdon, Bathgate, Shrewsbury, Glasgow etc., and the acceptable list contains a more specific normalized option that corresponds to that place, choose that more specific option.
    - If multiple locations are mentioned but one later explicit line or label clearly states the actual role location, prefer that clearer explicit location.
    - For ambiguous "hub based" or "multiple office" wording, use the most clearly role-specific location if supported by the text. If not clear, output the best supported normalized location, otherwise "Unknown".

    4) Language rule
    If the job requires a language other than English, mark Not Relevant.
    English-only is fine.

    5) job_location
    Extract and normalize exactly one value from the acceptable locations list below.
    Rules:
    - Prefer explicit location fields
    - Choose the most specific real job location
    - If a small place is not in the list, choose the closest broader acceptable location from the list
    - If no supported match exists, output "Unknown"
    - Output must be exactly one value from the acceptable locations list or "Unknown"

    Acceptable normalized locations:
    {locations_str}

    6) remote_preferences
    Allowed values only: "onsite", "hybrid", "remote"
    - Return an array
    - Order must always be: onsite, hybrid, remote
    - If not specified, return []

    7) remote_days
    Return only:
    - "0" to "5", or
    - "not specified"

    Rules:
    - Only use explicit remote/office pattern evidence
    - If 1 day in office -> 4
    - If 2 days in office -> 3
    - If 1-2 days in office -> 3
    - If 2-3 days in office -> 2
    - If fully remote / remote-first / unclear -> "not specified"
    - Never use salary numbers or unrelated numbers

    8) salary
    Extract salary from the text only if clearly supported.
    - Round to nearest value from predefined salary list below
    - If only one salary is present, set min and max to the same value
    - Currency must be a code like GBP, USD, EUR, CAD
    - If unsupported, leave fields empty

    Predefined salary list:
    {salaries_str}

    9) visa_sponsorship
    Output:
    - "yes"
    - "no"
    - ""
    Only if supported by the text.

    10) contract_type
    Output exactly one:
    - "Permanent"
    - "FTC"
    - "Part Time"
    - "Freelance/Contract"
    - ""
    Priority if multiple:
    Permanent > FTC > Part Time > Freelance/Contract

    11) job_titles
    Select up to 3 exact job titles from the predefined job titles list.
    - Use exact strings from the list only
    - Prefer fewer, more accurate titles
    - If position name exactly matches one predefined job title, usually return just that one, unless the description clearly supports a second closely related exact title
    - Do not output unrelated titles

    12) seniorities
    Select up to 3 values from:
    entry, junior, mid, senior, lead, leadership

    Rules:
    - Lowercase only
    - Order must always be:
      entry, junior, mid, senior, lead, leadership
    - "head of", "director", "vp", "chief", "engineering manager", "technical director" => leadership only
    - Titles containing "manager" should not be junior unless the text is overwhelmingly entry-level, which is rare
    - Titles containing "assistant" often support entry or junior
    - 0-1 years => entry, junior
    - 2 years => junior, mid
    - 3-5 years => senior
    - 5+ years or strong ownership / mentoring / managerial responsibility => senior, lead
    - managerial role with cross-functional ownership can justify lead
    - do not add junior/mid to clearly senior management roles
    - if uncertain for a genuine manager title, prefer senior and/or lead over junior

    13) skills
    Choose 0 to 10 skills.
    Hard rules:
    - The allowed skills depend on the final job_category:
      - If job_category = "T&P job", use ONLY the Allowed T&P skills list
      - If job_category = "Not T&P", use ONLY the Allowed Non-T&P skills list
    - Use exact strings from the relevant allowed list only
    - Prefer concrete skills clearly evidenced by the source text
    - Do not invent tools/languages/frameworks
    - If the source does not support a skill, do not include it
    - Better 2 accurate skills than 10 weak skills
    - Do not include skills just because they are common for the role
    - Only include inferential skills in rare obvious cases, for example LLMs can support Machine Learning / Artificial Intelligence when those exact skills are in the allowed list

    Allowed T&P skills:
    {tp_skills_str}

    Allowed Non-T&P skills:
    {nontp_skills_str}

    14) role_relevance_reason
    Give one concise reason that MUST match the final relevance decision.

    15) notes
    Very short note about extraction confidence/source. Keep concise.

    OUTPUT JSON SCHEMA
    {{
      "role_relevance": "Relevant" or "Not Relevant",
      "role_relevance_reason": "",
      "job_category": "T&P job" or "Not T&P",
      "job_location": "",
      "remote_preferences": [],
      "remote_days": "",
      "salary_min": "",
      "salary_max": "",
      "salary_currency": "",
      "visa_sponsorship": "",
      "contract_type": "",
      "job_titles": [],
      "seniorities": [],
      "skills": [],
      "notes": ""
    }}

    Position name:
    {position_name}

    Source text:
    {source_text}
    """).strip()
