from textwrap import dedent


def build_role_relevance_prompt(position_name: str, job_description: str, predefined_job_titles: list[str]) -> str:
    titles_str = ", ".join(predefined_job_titles)

    return dedent(f"""
    You will receive two inputs: position name and job description.

    Perform the following tasks carefully and output the results in the exact format described below.

    1. Role Relevance
    Decide if the role is Relevant or Not Relevant according to these criteria:

    Relevant roles match any in this list or close synonyms/specializations. Use the predefined list as the main source of truth:
    {titles_str}

    If the role is clearly outside tech or business functions (for example teacher, nurse, waiter), mark Not Relevant, even if some criteria partially match.

    Exclude any roles related to construction, civil engineering, retail, electrical, mechanical, manufacturing, microbiology, maritime, injection molding, and beauty brands.

    Location rules:
    a) United Kingdom: allowed for onsite, hybrid, or remote
    b) Ireland: only remote roles are allowed. If location mentions onsite or hybrid work, mark Not Relevant.
    c) Europe: only roles explicitly marked as Remote Europe or Remote EMEA are allowed. Specific European countries are Not Relevant unless explicitly remote in Europe/EMEA.
    d) Remote Global/worldwide is allowed unless the ad specifies or implies APAC, LATAM, Africa, Asia, USA, Canada, or another excluded region.
    e) If the role mentions a location outside allowed regions, or salary is only in USD/CAD and there is no evidence the role can be done from an allowed region, mark Not Relevant.
    f) Accept UK, Great Britain, London, England, Scotland, Wales, Northern Ireland as United Kingdom when appropriate.
    g) If multiple locations are listed, treat as Relevant only if the role can be done fully from an allowed location.

    Language rule:
    If the job requires any language other than English, mark Not Relevant.
    If the role requires English only, that is acceptable.

    2. Position Category
    Determine whether the job is:
    - T&P job
    - Not T&P

    T&P includes software development, engineering, product management, data, IT, UX/UI, QA, DevOps, infrastructure, security, technical support, technical architecture, and similar roles.

    Output format:
    Return exactly one single line with exactly three fields separated by " | ":

    1. Relevant or Not Relevant
    2. T&P job or Not T&P
    3. A concise explanation

    Example:
    Relevant | T&P job | Role fits allowed locations and matches tech product roles.
    Not Relevant | Not T&P | Location is Germany onsite, which is not allowed.

    Do not add anything beyond this format.

    Position name:
    {position_name}

    Job description:
    {job_description}
    """).strip()


def build_job_titles_prompt(position_name: str, job_description: str, predefined_job_titles: list[str]) -> str:
    titles_str = ", ".join(predefined_job_titles)

    return dedent(f"""
    You will receive two inputs: position name and job description.

    Perform the following task carefully and output the results exactly as described below.

    Job Title Identification

    a) Analyze both the position name and job description together.
    b) If the position name exactly matches one job title from the predefined list, output only that single job title.
    c) If the position name is unclear or ambiguous, select up to the top three most appropriate job titles from the predefined list that best fit the role.
    d) Only select job titles that exactly exist in the predefined list.
    e) Output the selected job titles as a comma-separated list, ordered from most to least appropriate.
    f) If fewer than three suitable job titles are found, output only those.
    g) If no suitable job title matches, output an empty string.

    Important:
    - Prefer the most precise business/technical title.
    - Do not guess unrelated titles.
    - Do not output titles that do not exist exactly in the predefined list.

    Output examples:
    Back End, Full Stack
    Product Manager
    Data Engineer, Data Scientist

    Predefined list of job titles:
    {titles_str}

    Position name:
    {position_name}

    Job description:
    {job_description}
    """).strip()


def build_seniority_prompt(position_name: str, job_description: str) -> str:
    return dedent(f"""
    You will receive two inputs: position name and job description.

    Perform the following tasks carefully and output the results exactly as described below.

    Seniority Level Determination

    First analyze the position name carefully.

    a) If the position name clearly indicates one seniority level from this predefined list, output only that level:
    entry, junior, mid, senior, lead, leadership

    b) If the title contains leadership indicators such as head of, director, VP, vice president, chief, C-level, or similar, output leadership only.

    c) If the title contains engineering manager, product manager with people management, team lead, lead engineer, or similar people-management responsibility, prefer lead or leadership as appropriate.

    If seniority is not clearly indicated by the title, analyze both title and description together.

    Rules:
    - Output up to three most appropriate seniority levels.
    - Output them only from this list:
      entry, junior, mid, senior, lead, leadership
    - Output must always be lowercase.
    - Output order must always be:
      entry, junior, mid, senior, lead, leadership

    Experience rules:
    - 0-1 years: entry, junior
    - 2 years: junior, mid
    - 3-5 years: senior, lead
    - Team management or managerial responsibility: include lead
    - Director, head of, VP, chief, C-level, engineering manager and similar leadership terms: leadership only

    Important:
    - Do not include mid if the role is clearly senior managerial level.
    - For PMO Manager, People Operations Manager, Engineering Manager, Head of X, Director roles, avoid junior or mid unless the description clearly supports that.
    - If the role is clearly managerial with ownership and cross-functional leadership, prefer senior/lead or leadership.

    Output examples:
    senior
    mid, senior
    lead
    junior, mid, senior

    Position name:
    {position_name}

    Job description:
    {job_description}
    """).strip()


def build_contract_type_prompt(job_description: str) -> str:
    return dedent(f"""
    You will receive a job description as input.

    Task:
    Identify and normalize the contract type into exactly one of these categories:

    Permanent
    FTC
    Part Time
    Freelance/Contract

    Rules:
    1. If multiple types are mentioned, choose only one using this priority:
       Permanent > FTC > Part Time > Freelance/Contract

    2. Map synonyms:
       - Permanent: permanent, full time, full-time, standard
       - FTC: temporary, fixed term, fixed-term, maternity cover, maternity leave
       - Part Time: part time, part-time, job share, job-share
       - Freelance/Contract: freelance, contract, contracting, contractor

    3. Output must be only one category name.
    4. If no valid type is found, output an empty string.

    Job description:
    {job_description}
    """).strip()


def build_skills_prompt(
    position_name: str,
    job_description: str,
    role_category_label: str,
    allowed_skills: list[str],
) -> str:
    skills_str = ", ".join(allowed_skills)

    return dedent(f"""
    You will receive:
    - position name
    - job description
    - role category
    - allowed skills list

    Task:
    Select up to 10 skills that are clearly supported by the position name and job description.

    Rules:
    - Only select skills that exactly exist in the allowed skills list.
    - Only select skills that are clearly evidenced by the text.
    - Do not guess.
    - Do not include soft skills unless they exist as exact allowed skills and are clearly supported.
    - Order skills from most relevant to least relevant.
    - If fewer than 10 fit, output only those.
    - If none fit, output an empty string.

    Role category:
    {role_category_label}

    Allowed skills list:
    {skills_str}

    Output:
    Return a comma-separated list only.

    Position name:
    {position_name}

    Job description:
    {job_description}
    """).strip()
