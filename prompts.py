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

    return dedent(
        f"""
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
        - If the explanation sounds Not Relevant, do not output Relevant.
        - If the role requires any language other than English, output Not Relevant.
        - Do not map a foreign location to a UK location just because of generic words like city, centre, road, or united kingdom.
        - If an explicit location line says a foreign place such as Bonifacio Global City / Makati / Metro Manila / Philippines, output job_location = "Unknown" and role_relevance = "Not Relevant".

        TASKS

        1) role_relevance
        Decide if the role is "Relevant" or "Not Relevant".

        Relevant roles match the predefined job titles list below, or close synonyms / specializations that clearly map to one of them:
        {job_titles_str}

        Also allowed when clearly genuine corporate roles:
        - finance / accounting / FP&A / tax / treasury / audit / controller / accounts payable / accounts receivable / billing / credit control / investment operations / fund accounting / transfer pricing
        - risk / compliance / credit risk / financial crime / regulatory compliance / AML / KYC roles
        - business operations / program / PMO / transformation / change / analyst / business support / data administrator roles
        - account / client / customer / renewals / partnerships / implementation / customer success / customer support roles
        - marketing / growth / content / CRM / communications / PR / product marketing / demand generation / brand marketing / influencer marketing / events / community roles
        - brand design / visual design / graphic design / brand designer / brand design lead roles
        - assistant brand manager / brand manager / brand marketing roles when they are genuine corporate marketing roles
        - executive assistant / personal assistant / chief of staff / legal / commercial counsel / paralegal / company secretary roles
        - HR / human resources / people ops / people operations / talent acquisition / recruitment / employee relations / total rewards / compensation and benefits roles
        - procurement / sourcing / supply chain analyst / buyer roles when they are office/corporate roles
        - quality assurance / QA analyst / quality engineer roles when they are not shop-floor/manufacturing inspection roles
        - technical support / IT / infrastructure / systems / network / support engineering roles when clearly technical
        - business development / BDR / SDR / account executive / sales consultant / commercial roles when clearly genuine business or B2B roles and not retail store sales

        Important industry-context rule:
        - Do NOT mark a corporate business role Not Relevant only because the company, customer base, or industry mentions construction, property, manufacturing, hospitality, healthcare, maritime, automotive, retail, or similar sectors.
        - Judge the ACTUAL JOB FUNCTION, not only the company industry.
        - Example: Account Executive selling to construction/property clients can still be Relevant if the role itself is sales/business development.
        - Example: Finance Analyst at a manufacturing company can still be Relevant if the role itself is finance/accounting.
        - Example: Legal Counsel at a healthcare company can still be Relevant if the role itself is legal/commercial.
        - Example: HR Advisor at a retail company can still be Relevant if the role itself is HR.
        - Example: Procurement Manager at a manufacturing company can still be Relevant if the role itself is corporate procurement.
        - Only exclude when the actual role duties are construction/civil engineering, shop-floor, plant, factory, mechanical/electrical technician, medical/clinical/patient-care, retail-store, hospitality service, etc.

        Exclusions:
        - Not a real job posting
        - Educational content, learning modules, checklists, articles, guides
        - Actual construction, civil engineering, site manager, quantity surveyor, building-site roles
        - Actual retail store roles, cashier, showroom/store/branch/shop-floor roles
        - Actual electrical / mechanical / manufacturing / plant / factory / assembly / injection molding / maritime / microbiology roles
        - Robotics technician / electro-mechanical / hands-on hardware build and maintenance roles
        - Medical / clinical / patient care roles
        - Beauty therapist / salon / cosmetology / in-store beauty advisor roles
        - Hospitality service roles such as waiter, waitress, chef, kitchen roles
        - Any role clearly outside allowed tech/business functions

        2) job_category
        Output exactly one:
        - "T&P job"
        - "Not T&P"

        T&P includes:
        software engineering, data, DevOps, QA, security, IT, support engineering, infrastructure, cloud, UX/UI, product, technical architecture, technical writing, ML/AI, systems, network, application engineering, and similar technical roles.

        Most finance/accounting, risk/compliance, sales, marketing, legal, HR, customer success, procurement, and operations roles are "Not T&P" unless they are clearly technical/product/data roles.

        3) Location rules for relevance
        Relevant only if the role fits allowed locations:
        - United Kingdom: onsite, hybrid, or remote allowed
        - Ireland: only remote allowed
        - Europe: only explicitly remote Europe / remote EMEA allowed
        - Remote Global / worldwide allowed unless ad clearly restricts to APAC / LATAM / Africa / Asia / USA / Canada / another excluded region
        - If the job clearly indicates USA / Canada / Philippines / another excluded region as the actual job location, mark Not Relevant
        - If salary is only USD/CAD and nothing supports allowed regions, that is evidence for Not Relevant
        - Accept UK / Great Britain / England / Scotland / Wales / Northern Ireland / London etc. as UK
        - Ignore generic company office lists unless they clearly describe this role’s actual work location
        - Do not mark Not Relevant just because boilerplate mentions global offices, US headquarters, customers in other countries, or international company presence.

        Very important for job_location:
        - Choose the MOST SPECIFIC acceptable normalized location from the predefined list.
        - Do NOT fall back to a more specific neighborhood or office if the text only says a broader place.
        - Example: if the text says London, UK and that exact normalized location exists, choose London, UK, not Abbey Road, London, United Kingdom.
        - If the posting says a town/city like Stansted, Huntingdon, Bathgate, Shrewsbury, Glasgow, Worthing etc., and the acceptable list contains a more specific normalized option that corresponds to that place, choose that more specific option.
        - If the description has a clear explicit location, prefer that over generic office-list text from later page text.
        - If multiple locations are mentioned but one later explicit line or label clearly states the actual role location, prefer that clearer explicit location.
        - If there is an ambiguous hub-based / multiple-office statement AND a separate explicit location field elsewhere in the text, prefer the explicit location field.
        - For ambiguous "hub based" or "multiple office" wording, use the most clearly role-specific location if supported by the text. If not clear, output "Unknown".
        - If the explicit location is outside the allowed set and no acceptable normalized match exists, output "Unknown". Do not guess a UK location.

        4) Language rule
        - If the job description is mainly written in another language, mark Not Relevant.
        - If the role requires any language other than English, mark Not Relevant.
        - Example: "Fluency in French and English" => Not Relevant.
        - Example: "German required" => Not Relevant.
        - Example: "Business-level Dutch preferred" => Not Relevant.
        - English-only is fine.

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

        Important:
        - If the posting clearly says "Remote", "fully remote", "remote only", or "mostly async from anywhere", prefer ["remote"].
        - Do not add onsite or hybrid unless explicitly supported.
        - If the posting clearly says on-site / onsite / office-based / F/T Site, include onsite.
        - If the posting clearly says hybrid / occasional home working / one day work from home / flexibility for occasional home working, include hybrid.
        - Do not output onsite and remote together unless the text explicitly supports both.

        7) remote_days
        Return only:
        - "0" to "5", or
        - "not specified"

        Rules:
        - Only use explicit remote/office pattern evidence
        - If 1 day in office -> 4
        - If 2 days in office -> 3
        - If 3 days in office -> 2
        - If 4 days in office -> 1
        - If 1-2 days in office -> 3
        - If 2-3 days in office -> 2
        - If one day work from home / WFH -> 1
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
        - For designer roles, prioritize the designer title over broader marketing labels if both are supported
        - Brand design / brand designer / brand design lead roles should include the best matching designer title if it exists in the predefined list, not only Brand Marketing
        - Assistant Brand Manager roles should prefer the closest brand/marketing title, not be treated as irrelevant
        - 1st line / IT support / technical support / application engineer roles should map to the most appropriate technical title, not customer service
        - Do not output unrelated titles

        12) seniorities
        Select up to 3 values from:
        entry, junior, mid, senior, lead, leadership

        Rules:
        - Lowercase only
        - Order must always be:
          entry, junior, mid, senior, lead, leadership
        - "head of", "director", "vp", "chief", "engineering manager", "technical director" => leadership only
        - "assistant manager" is usually junior, mid
        - Other assistant roles are usually junior, mid unless the text clearly proves something else
        - Generic "manager" should usually be senior, lead
        - Generic non-manager professional roles without clear years can be junior, mid, senior
        - Do not use lead for non-manager roles unless the text clearly shows strong ownership / mentoring / cross-functional leadership
        - 0-1 years => entry, junior
        - 2 years => junior, mid
        - 3-5 years => senior
        - 5+ years or strong ownership / mentoring / managerial responsibility => senior, lead
        - avoid entry unless the posting truly looks early-career
        - do not add junior/mid to clearly senior management roles
        - assistant brand manager should usually NOT be senior, lead
        - if unsure for a non-manager professional role, junior, mid, senior is safer than lead
        - do not overuse lead

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
        - Be careful with false positives:
          - do not infer "R" from words like New Relic
          - do not infer "Flutter" from company names unless the framework is clearly referenced
        - Only include inferential skills in rare obvious cases
        - If the description is thin and only a few skills are obvious, return just the most appropriate 2-4 supported skills

        Allowed T&P skills:
        {tp_skills_str}

        Allowed Non-T&P skills:
        {nontp_skills_str}

        14) role_relevance_reason
        Give one concise reason that MUST match the final relevance decision.
        - If Relevant, explain the allowed job function and allowed location/work-pattern basis.
        - If Not Relevant, explain the actual exclusion reason.

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
        """
    ).strip()
