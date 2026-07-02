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
        - If the actual job location is North Ryde / NSW / Australia, output job_location = "Unknown" and role_relevance = "Not Relevant".
        - Be careful not to use locations from unrelated job cards, footer links, "More jobs", "Similar jobs", or "Related jobs" sections. Only use the actual role location.

        TASKS

        1) role_relevance
        Decide if the role is "Relevant" or "Not Relevant".

        Relevant roles match the predefined job titles list below, or close synonyms / specializations that clearly map to one of them:
        {job_titles_str}

        Also allowed when clearly genuine corporate roles:
        - finance / accounting / financial analyst / finance analyst / FP&A / tax / treasury / audit / controller / accounts payable / accounts receivable / billing / credit control / investment operations / fund accounting / transfer pricing
        - risk / compliance / credit risk / financial crime / regulatory compliance / AML / KYC / governance risk compliance / GRC roles
        - legal / legal specialist / AI legal specialist / privacy / data protection / commercial counsel / legal counsel / paralegal / company secretary / contracts roles
        - HR / human resources / HR generalist / people ops / people operations / talent acquisition / recruitment / employee relations / total rewards / compensation and benefits roles
        - communications / corporate communications / internal communications / external communications / PR / public relations / public affairs roles
        - business operations / program / PMO / transformation / change / analyst / business support / data administrator roles
        - account / client / customer / renewals / partnerships / implementation / customer success / customer support roles
        - marketing / growth / content / CRM / product marketing / demand generation / brand marketing / influencer marketing / events / community roles
        - brand design / visual design / graphic design / brand designer / brand design lead roles
        - assistant brand manager / brand manager / brand marketing roles when they are genuine corporate marketing roles
        - executive assistant / personal assistant / chief of staff roles
        - procurement / sourcing / supply chain analyst / buyer roles when they are office/corporate roles
        - quality assurance / QA analyst / quality engineer roles when they are not shop-floor/manufacturing inspection roles
        - technical support / IT / infrastructure / systems / network / support engineering roles when clearly technical
        - business development / BDR / SDR / account executive / sales consultant / commercial roles when clearly genuine business or B2B roles and not retail store sales

        Important industry-context rule:
        - Do NOT mark a corporate business role Not Relevant only because the company, customer base, or industry mentions construction, property, manufacturing, hospitality, healthcare, maritime, automotive, retail, or similar sectors.
        - Judge the ACTUAL JOB FUNCTION, not only the company industry.
        - Example: Account Executive selling to construction/property clients can still be Relevant if the role itself is sales/business development.
        - Example: Finance Analyst at a manufacturing company can still be Relevant if the role itself is finance/accounting.
        - Example: Financial Analyst / Finance Analyst / Accountant can still be Relevant if it is a genuine finance/accounting role.
        - Example: Legal Counsel at a healthcare company can still be Relevant if the role itself is legal/commercial.
        - Example: AI Legal Specialist / Privacy Counsel / Compliance Specialist can still be Relevant if it is a genuine legal/compliance/privacy role.
        - Example: HR Advisor / HR Generalist / People Advisor / Talent Acquisition role can still be Relevant if it is a genuine HR/People role.
        - Example: Communications Lead / PR Lead / Corporate Communications role can still be Relevant if it is a genuine communications/PR corporate role.
        - Example: Procurement Manager at a manufacturing company can still be Relevant if the role itself is corporate procurement.
        - Only exclude when the actual role duties are construction/civil engineering, shop-floor, plant, factory, mechanical/electrical technician, medical/clinical/patient-care, retail-store, hospitality service, etc.

        Exclusions:
        - Not a real job posting
        - Volunteer / voluntary / unpaid volunteer roles
        - Educational content, learning modules, checklists, articles, guides
        - Actual construction, civil engineering, structural engineering, site manager, quantity surveyor, building-site roles
        - Actual retail store roles, cashier, showroom/store/branch/shop-floor roles, branch and yard duties, trade counter roles
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

        Most finance/accounting, risk/compliance, sales, marketing, communications, PR, legal, HR, customer success, procurement, and operations roles are "Not T&P" unless they are clearly technical/product/data roles.

        3) Location rules for relevance
        Relevant only if the role fits allowed locations:
        - United Kingdom: onsite, hybrid, or remote allowed
        - Ireland: only remote allowed
        - Europe: only explicitly remote Europe / remote EMEA allowed
        - Remote Global / worldwide allowed unless ad clearly restricts to APAC / LATAM / Africa / Asia / USA / Canada / Australia / another excluded region
        - If the job clearly indicates USA / Canada / Philippines / Australia / another excluded region as the actual job location, mark Not Relevant
        - If the actual job location is North Ryde / NSW / Australia, mark Not Relevant
        - If salary is only USD/CAD/AUD and nothing supports allowed regions, that is evidence for Not Relevant
        - Accept UK / Great Britain / England / Scotland / Wales / Northern Ireland / London etc. as UK
        - Ignore generic company office lists unless they clearly describe this role's actual work location
        - Do not mark Not Relevant just because boilerplate mentions global offices, US headquarters, customers in other countries, or international company presence
        - Be careful not to use locations from unrelated job cards, footer links, "More jobs", "Similar jobs", or "Related jobs" sections. Only use the actual role location.

        Very important for job_location:
        - Report the location exactly as the posting states it for this role. Do not invent a more specific neighbourhood, street, or office than the text supports.
        - Do NOT fall back to a more specific neighborhood or office if the text only says a broader place. Example: if the text only says London, UK, output "London, UK", not a specific street or office in London.
        - If the description has a clear explicit location, prefer that over generic office-list text from later page text.
        - If multiple locations are mentioned but one later explicit line or label clearly states the actual role location, prefer that clearer explicit location.
        - If there is an ambiguous hub-based / multiple-office statement AND a separate explicit location field elsewhere in the text, prefer the explicit location field.
        - For ambiguous "hub based" or "multiple office" wording, use the most clearly role-specific location if supported by the text. If it is clearly a UK role but no single city is clear, output "United Kingdom". Only output "Unknown" when the location is genuinely unclear or not an allowed region.
        - If the actual location is outside the allowed regions, output "Unknown". Do not guess a UK location for a role that is really elsewhere.

        4) Language rule
        - If the job description is mainly written in another language, mark Not Relevant.
        - If the role requires any language other than English, mark Not Relevant.
        - Example: "Fluency in French and English" => Not Relevant.
        - Example: "German required" => Not Relevant.
        - Example: "Business-level Dutch preferred" => Not Relevant.
        - English-only is fine.

        5) job_location
        Extract the single real work location for THIS role as plain text.
        Rules:
        - The location may be a labelled field ("Location:", "based in", office/city line) OR unlabelled and inline, often near the top - read the WHOLE posting, do not only look for a "Location" label.
        - Write it in plain "City, Country" form when possible (e.g. "Manchester, UK"); it will be normalized to the canonical location afterwards.
        - If the role is clearly in the UK but NO single city is named (or several UK cities are listed), output "United Kingdom". Do NOT output "Unknown" just because a specific city is missing.
        - LONDON: if the role is anywhere in London (central London, a London borough, or a London suburb such as Shoreditch/Croydon/Wimbledon), output exactly "London, UK" - do not output the specific neighbourhood.
        - Do not invent a more specific neighbourhood, street, or office than the text supports.
        - Do NOT use locations from unrelated job cards, footers, "More jobs", "Similar jobs", or "Related jobs" sections.
        - Output "Unknown" ONLY when the role is genuinely remote-anywhere with no allowed region, or the actual location is a foreign/excluded place, or there is truly no supported location at all.

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
        - A Relevant role must ALWAYS get at least one job title. Never leave job_titles empty for a Relevant role - pick the single closest predefined title.
        - An engineering/technical role (anything "... Engineer" or "... Developer", including "Product Engineer") is a SOFTWARE/TECHNICAL ENGINEER. NEVER map it to "Product Manager", "Product Owner", or "Project Manager". (A "Product Engineer" is an engineer, not a Product Manager.)
        - There is no generic "Software Engineer"/"Developer" in the list, so decide what KIND of engineer this specific role is from the description and map to the most fitting predefined title. Examples:
          - AI / ML / deep learning / LLM work => "AI Engineer" or "Machine Learning Engineer"
          - back-end / server / API work => "Back End"
          - front-end / UI work => "Front End"
          - both front and back end => "Full Stack"
          - data pipelines / warehousing => "Data Engineer"; data science => "Data Scientist"
          - cloud / infrastructure => "Cloud Engineer"; devops / CI-CD => "DevOps Engineer"
          - embedded / firmware / hardware-adjacent => "Embedded Developer"
          - security => "Security Engineer"; QA / test automation => "QA Automation Tester" or "Developer in Test"
          - if the specialization is genuinely unclear, choose "Full Stack" or "Back End" as the closest general software title
        - If position name exactly matches one predefined job title, usually return just that one, unless the description clearly supports a second closely related exact title
        - For designer roles, prioritize the designer title over broader marketing labels if both are supported
        - Brand design / brand designer / brand design lead roles should include the best matching designer title if it exists in the predefined list, not only Brand Marketing
        - Assistant Brand Manager roles should prefer the closest brand/marketing title, not be treated as irrelevant
        - Communications / PR roles should map to the closest communications, marketing, PR, brand, or corporate role available in the predefined list
        - HR / People roles should map to the closest HR, People, Recruiting, Talent, Operations, or corporate role available in the predefined list
        - Legal / Privacy / Compliance roles should map to the closest Legal, Compliance, Operations, or corporate role available in the predefined list
        - Finance / Accounting / Financial Analyst roles should map to the closest Finance/Accounting, Analyst, or Operations title available in the predefined list
        - 1st line / IT support / technical support / application engineer roles should map to the most appropriate technical title, not customer service
        - ALWAYS output the title EXACTLY as it appears in the predefined list. If the natural title is not literally in the list, pick the CLOSEST predefined equivalent rather than leaving it blank. Examples:
          - Customer Success Manager => "CSM/Account Manager"
          - Business Intelligence Developer / BI Analyst => "BI Developer"
          - IT Support Analyst / Service Desk / Desktop Support => "Support Engineer" or "System Administrator"
          - Risk Manager / Compliance Analyst / Regulatory / AML / KYC => "Risk and Compliance"
          - Sales Manager / Sales Executive / Commercial Manager => "Business Development Manager" or "Account Executive"
          - Programme Manager / Delivery Manager / PMO / Project Coordinator => "Project Manager"
          - Office Manager / Personal Assistant / Executive Assistant => "Executive Assistant" or "Operations"
          - Payroll / Pensions / Accounts Assistant => "Finance/Accounting" or "Operations"
          - Marketing Manager / Digital Marketing / Brand Manager => the closest Marketing title (e.g. "Generalist Marketing", "Digital Marketing", "Content Marketing", "Brand Marketing")
          - HR Advisor / People Partner / Recruiter / Talent => "Human Resources" or "Talent Acquisition"
          - Communications / PR Manager / Public Relations => "PR/Communications"
        - Do not output unrelated titles

        12) seniorities
        Select up to 3 values from: entry, junior, mid, senior, lead, leadership
        - Lowercase only.
        - Always order them as: entry, junior, mid, senior, lead
        - "leadership" is EXCLUSIVE: if it applies, output ONLY ["leadership"] (never combine it with senior, lead, etc.)

        Decide using the FIRST rule below that clearly applies:

        a) LEADERSHIP => ["leadership"] only.
           Leadership means: any "Head of ..." role, ANY "... Director" title
           (e.g. Account Director, Sales Director, Client Director, Technical Director,
           Creative Director), a C-level role (CEO/CTO/CFO/COO/CIO/CMO/CRO/CPO/CDO/CLO/CSO),
           VP of Engineering, Engineering Manager, Founder, or Chief of Staff.
           NOT leadership: "Account Manager", "Account Coordinator", "Product Manager",
           "Project Manager", "CSM" and similar (note: a "Manager"/"Coordinator" without
           "Director"/"Head of"/C-level is NOT leadership). Decide their seniority from the
           experience / explicit-level / ownership rules below (b, c, d) - do not force a
           level. For example a Product Manager may be junior, mid, senior, or lead.

        b) EXPLICIT LEVEL IN THE POSITION NAME => output ONLY that level.
           If the position name itself states the level, that IS the seniority:
           - "graduate" / "intern" / "trainee" / "apprentice" / "entry-level"  => ["entry"]
           - "junior" / "jr"                                                   => ["junior"]
           - "mid" / "mid-level"                                               => ["mid"]
             (but IGNORE "mid-market" and similar - that is a market segment, NOT a seniority)
           - "senior" / "snr" / "sr"                                           => ["senior"]
           - "lead" / "principal" / "staff engineer"                           => ["lead"]
             (but IGNORE "lead generation" - that is sales, NOT a seniority)

        c) EXPERIENCE-BASED (when the text states years of experience):
           - less than 1 year / 0-1 years => ["entry"]
           - 1-2 years                    => ["junior", "mid"]
           - 3-4 years                    => ["senior"]
           - 4+ years                     => ["senior", "lead"] if it is NOT really a manager role;
                                             ["lead"] if it IS a manager role

        d) Otherwise use responsibility / ownership signals to choose the MOST ACCURATE level(s):
           - individual contributor  => entry / junior / mid / senior (pick the best fit)
           - owns a team             => ["lead"]
           - owns an org or function => ["leadership"]

        Be smart and pick the most precise level you can justify. Only fall back to
        ["junior", "mid", "senior"] when you genuinely cannot tell - it is a last resort, not a default.

        13) skills
        Choose the skills that are MOST APPROPRIATE for THIS specific role (up to 10).
        - The allowed skills depend on the final job_category:
          - If job_category = "T&P job", use ONLY the Allowed T&P skills list
          - If job_category = "Not T&P", use ONLY the Allowed Non-T&P skills list
        - Use exact strings from the relevant allowed list only.
        - Prefer skills clearly evidenced by the source text.
        - If the description is thin or does not explicitly list skills, INFER the skills most typical and appropriate for this kind of role (still only from the allowed list). A Relevant job should normally receive at least 3-5 appropriate skills - do not leave skills empty for a relevant role.
        - Every skill you choose MUST genuinely fit this specific role. Do NOT add skills unrelated to the role, and do not invent tools/languages/frameworks that would not plausibly apply.
        - Be careful with false positives when reading the text:
          - do not infer "R" from words like New Relic
          - do not infer "Flutter" from company names unless the framework is clearly referenced

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

        16) clean_job_title
        Output a cleaned version of the ORIGINAL position name: keep only the core
        job title, stripped of trailing company names, locations, contract types,
        and separators.
        - Keep only the core title at the start, stopping at the first separator or
          trailing detail. Separators include "\u00b7", "-", "|", commas that begin
          trailing info, and words/markers like Full-Time, Part-Time, Contract,
          Permanent, Remote, Hybrid, Location, Office, Ltd, Limited, Inc, or a
          city/country name.
        - Preserve the exact wording and order of the title portion. Do not rephrase,
          translate, expand, or add words. Preserve commas/hyphens that are genuinely
          inside the core title (e.g. "Analyst, Commercial Analytics").
        - Examples:
          - "Account Director Event Concept Ltd Full-Time Contract" => "Account Director"
          - "Analyst, Commercial Analytics - NORAM & LATAM KAYAK \u00b7 Commercial" => "Analyst, Commercial Analytics"
          - "Senior Software Engineer - New York, NY" => "Senior Software Engineer"
          - "Marketing Manager | Remote" => "Marketing Manager"
          - "Business Analyst, San Francisco" => "Business Analyst"
        - If the position name is already clean, return it unchanged.

        OUTPUT JSON SCHEMA
        {{
          "role_relevance": "Relevant" or "Not Relevant",
          "role_relevance_reason": "",
          "job_category": "T&P job" or "Not T&P",
          "job_location": "",
          "clean_job_title": "",
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


def build_relevant_description_prompt(description_text: str) -> str:
    return dedent(
        f"""
        You are given a raw job description. Extract only the parts that describe
        the job itself, clean them up, and return them as simple HTML.

        INCLUDE
        - The role's purpose and responsibilities ("About the Role", "Responsibilities", "Key Duties", "What you'll do").
        - Candidate requirements, qualifications, skills and experience ("About You", "Requirements", "What you'll bring", "Ideally you'll have").
        - Any sentence that directly explains what the job involves or what the candidate must bring.

        EXCLUDE
        - Company background / "about us" marketing.
        - Benefits, perks, salary-sell, culture fluff.
        - Legal disclaimers, equal-opportunity statements, application instructions, "how to apply".
        - Recruiter/agency boilerplate and unrelated content.

        CHARACTER & ENCODING RULES (important)
        - Output clean, correctly-encoded text.
        - Decode any HTML entities to their real character (&amp;->&, &#39;/&#x27;/&rsquo;->', &quot;->").
        - Repair corrupted punctuation. In this data an apostrophe is often replaced by a stray digit - most commonly a 7, sometimes 2, 9, 27, 39, 92 or 99 - sitting between a word and a contraction or possessive. Restore the apostrophe whenever a digit appears where one grammatically belongs:
          "We 7re" -> "We're", "You 7ll" -> "You'll", "don 7t" / "don7t" -> "don't", "company7s" -> "company's", "we 27re" -> "we're".
        - ONLY fix a digit that is standing in for an apostrophe (directly before a contraction ending ll/re/ve/s/t/d/m, or a possessive 's). NEVER change a digit that is a real number - leave "7 years", "7am", "\u00a370,000", "level 3", "24/7" exactly as they are.
        - Normalise curly quotes/apostrophes and dashes to plain forms; do not otherwise alter wording.

        FORMATTING RULES
        - Use ONLY these tags: <p>, <strong>, <ul>, <ol>, <li>, <br>. No other tags, no markdown, no code fences.
        - Section titles => <p><strong>Title</strong></p>, keeping the original wording (convert ALL-CAPS headers to Title Case). If the header is "About the Role" / "The Role" / "About this Role", drop the header and keep its content.
        - Bullets => <ul><li>...</li></ul>: one point per <li>, no blank lines between items, no <p> inside <li>. Only make bullets where the source actually had them; do not turn prose into bullets.
        - If bullets appear with no header above them: duties => <p><strong>Responsibilities</strong></p>; qualifications/skills => <p><strong>Requirements</strong></p>. Do not invent headers for plain prose.

        PRESERVATION
        - Keep included sentences word-for-word apart from the character/encoding repairs above. Do not rephrase, summarise, or add anything.
        - If nothing meaningful remains after removing excluded content, output an empty string.

        Return only the HTML. No explanations.

        RAW JOB DESCRIPTION:
        {description_text}
        """
    ).strip()
