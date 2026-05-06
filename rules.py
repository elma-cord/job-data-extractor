import csv
import re
from pathlib import Path


GENERIC_LOCATION_TOKENS = {
    "uk",
    "united",
    "kingdom",
    "great",
    "britain",
    "england",
    "scotland",
    "wales",
    "northern",
    "ireland",
    "unitedstates",
    "states",
    "usa",
    "us",
    "city",
    "centre",
    "center",
    "county",
    "region",
    "remote",
    "hybrid",
    "onsite",
    "on",
    "site",
    "office",
}

DISALLOWED_LOCATION_TERMS = [
    "philippines",
    "metro manila",
    "makati",
    "bonifacio global city",
    "taguig",
    "united states",
    "usa",
    "us only",
    "canada",
    "germany",
    "france",
    "spain",
    "italy",
    "netherlands",
    "belgium",
    "sweden",
    "norway",
    "denmark",
    "finland",
    "switzerland",
    "austria",
    "poland",
    "portugal",
    "india",
    "singapore",
    "japan",
    "china",
    "australia",
    "new south wales",
    "nsw",
    "north ryde",
    "apac",
    "latam",
    "africa",
]


def load_single_column_csv(path: Path) -> list[str]:
    values = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            value = (row[0] or "").strip()
            if value and value.lower() not in {"job title", "location", "skill", "salary"}:
                values.append(value)
    return values


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def lower_text(text: str) -> str:
    return normalize_text(text).lower()


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def get_primary_text_window(text: str, max_chars: int = 12000) -> str:
    text = text or ""

    cut_markers = [
        r"(?i)\bour hiring process\b",
        r"(?i)\bother jobs\b",
        r"(?i)\bsimilar jobs\b",
        r"(?i)\byou may also like\b",
        r"(?i)\bmore jobs\b",
        r"(?i)\brelated jobs\b",
        r"(?i)\bsee more jobs\b",
        r"(?i)\bview all jobs\b",
    ]
    for pattern in cut_markers:
        m = re.search(pattern, text)
        if m:
            text = text[:m.start()]
            break

    if len(text) > max_chars:
        text = text[:max_chars]

    return text.strip()


def looks_like_non_job_content(position_name: str, description: str) -> bool:
    text = lower_text(f"{position_name}\n{description}")
    if not text:
        return True

    hard_non_job_terms = [
        "course module",
        "learning module",
        "training module",
        "lesson",
        "curriculum",
        "course content",
        "study guide",
        "career guide",
        "salary guide",
        "news article",
        "blog post",
        "learning objectives",
        "administrative checklist",
        "checklist only",
    ]
    job_signals = [
        "responsibilities",
        "requirements",
        "experience",
        "apply",
        "job type",
        "salary",
        "location",
        "benefits",
        "we are looking for",
        "candidate",
        "team",
        "reporting to",
        "full time",
        "part time",
        "role purpose",
        "job description",
    ]

    if any(term in text for term in hard_non_job_terms) and not any(term in text for term in job_signals):
        return True

    if "learning objectives" in text and "apply" not in text and "responsibilities" not in text:
        return True

    if "administrative checklist" in text and "apply" not in text and "responsibilities" not in text:
        return True

    return False


def _actual_role_text(position_name: str, description: str) -> str:
    title = lower_text(position_name)
    primary = lower_text(get_primary_text_window(description, max_chars=5000))
    return f"{title}\n{primary}"


def obvious_excluded_role(position_name: str, description: str) -> tuple[bool, str]:
    title = lower_text(position_name)
    text = _actual_role_text(position_name, description)

    title_exclusion_patterns = [
        (r"\bvolunteer\b|\bvoluntary\b|\bunpaid volunteer\b", "Volunteer role is outside allowed scope."),
        (r"\bteacher\b|\bteaching assistant\b", "Teaching role is outside allowed tech/business scope."),
        (r"\bnurse\b|\bregistered nurse\b|\bhealthcare assistant\b", "Medical role is outside allowed tech/business scope."),
        (r"\bwaiter\b|\bwaitress\b|\bchef\b|\bkitchen assistant\b|\bkitchen porter\b", "Hospitality role is outside allowed tech/business scope."),
        (r"\bcashier\b|\bretail assistant\b|\bsales associate\b|\bshop assistant\b|\bstore assistant\b", "Retail store role is outside allowed tech/business scope."),
        (r"\bcivil engineer(?:ing)?\b|\bgraduate civil engineer\b|\bstructural engineer\b|\bconstruction manager\b|\bsite manager\b|\bsite supervisor\b|\bquantity surveyor\b", "Construction / civil engineering role is outside allowed scope."),
        (r"\bmanufacturing technician\b|\bproduction operator\b|\bassembly technician\b|\bplant operator\b", "Manufacturing / shop-floor role is outside allowed scope."),
        (r"\brobotics technician\b|\belectro[- ]?mechanical\b|\bmechanical technician\b|\belectrical technician\b|\bmechatronics\b", "Hands-on robotics / mechanical / electrical technician role is outside allowed scope."),
        (r"\bpsychiatrist\b|\bphysician\b|\bsurgeon\b|\btherapist\b|\bpatient care\b", "Medical / clinical role is outside allowed scope."),
        (r"\brf test engineer\b|\bradio frequency test engineer\b", "RF test engineering role is outside allowed target scope."),
    ]

    for pattern, reason in title_exclusion_patterns:
        if re.search(pattern, title):
            return True, reason

    role_context_exclusion_patterns = [
        (
            r"\bresponsible for\b.{0,80}\b(shop floor|production line|assembly line|plant machinery|machine operation)\b",
            "Manufacturing / shop-floor role is outside allowed scope.",
        ),
        (
            r"\b(?:operate|repair|maintain|install|assemble|weld|fabricate)\b.{0,80}\b(machine|machinery|mechanical|electrical equipment|plant equipment|hardware)\b",
            "Hands-on mechanical / electrical role is outside allowed scope.",
        ),
        (
            r"\bprovide\b.{0,80}\b(patient care|clinical care|medical care)\b",
            "Medical / clinical role is outside allowed scope.",
        ),
        (
            r"\bworking on(?:-|\s)?site\b.{0,80}\b(construction site|building site)\b",
            "Construction / site-based role is outside allowed scope.",
        ),
        (
            r"\b(branch and yard duties|yard duties|trade counter|shop floor|store floor|retail store)\b",
            "Retail store / branch-floor role is outside allowed scope.",
        ),
    ]

    for pattern, reason in role_context_exclusion_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            return True, reason

    return False, ""


def detect_quick_tp_from_title(position_name: str) -> str:
    title = lower_text(position_name)

    tp_terms = [
        "engineer",
        "developer",
        "software",
        "data",
        "product",
        "ux",
        "ui",
        "devops",
        "site reliability",
        "security",
        "qa",
        "automation",
        "backend",
        "back end",
        "front end",
        "frontend",
        "full stack",
        "it support",
        "infrastructure",
        "architect",
        "technical",
        "support engineer",
        "systems engineer",
        "system engineer",
        "network engineer",
        "platform",
        "cloud",
        "sre",
        "machine learning",
        "ai engineer",
        "application engineer",
        "administrator",
        "designer",
        "brand designer",
        "backend engineer",
        "qa engineer",
        "quality assurance",
    ]

    if any(term in title for term in tp_terms):
        return "T&P job"

    return "Not T&P" if title else ""


def _has_retail_store_context(text: str) -> bool:
    negative_retail_terms = [
        "retail store",
        "store sales",
        "shop floor",
        "showroom",
        "cashier",
        "sales assistant",
        "sales associate",
        "customer floor",
        "in-store",
        "instore",
        "counter sales",
        "store manager",
        "branch showroom",
        "car showroom",
        "branch and yard",
        "yard duties",
        "trade counter",
        "branch sales",
        "branch retail",
        "store-floor",
        "store floor",
    ]
    return any(term in text for term in negative_retail_terms)


def detect_relevant_business_sales_role(position_name: str, description: str) -> bool:
    title = lower_text(position_name)
    text = lower_text(f"{position_name}\n{description}")

    positive_terms = [
        "business development",
        "business development representative",
        "business development consultant",
        "sales development representative",
        "bdr",
        "sdr",
        "account executive",
        "account manager",
        "account director",
        "commercial associate",
        "commercial manager",
        "sales consultant",
        "partnerships",
        "partner manager",
        "renewals",
        "sales operations",
        "revenue operations",
        "revops",
        "customer success",
        "client success",
        "client relationship",
        "customer operations",
        "customer support",
        "customer service representative",
        "customer service rep",
        "payment consultant",
        "payments consultant",
        "implementation manager",
        "lead generation",
    ]

    if any(term in title for term in positive_terms):
        return not _has_retail_store_context(text)

    if ("sales" in title or "commercial" in title or "client" in title or "customer" in title or "partnership" in title) and not _has_retail_store_context(text):
        return True

    return False


def detect_relevant_finance_accounting_role(position_name: str, description: str) -> bool:
    title = lower_text(position_name)
    text = lower_text(f"{position_name}\n{description}")

    positive_title_terms = [
        "accounting analyst",
        "accountant",
        "accounting",
        "finance analyst",
        "financial analyst",
        "finance business partner",
        "fp&a",
        "financial planning and analysis",
        "accounts payable",
        "accounts receivable",
        "ap analyst",
        "ar analyst",
        "payroll",
        "tax",
        "treasury",
        "audit",
        "auditor",
        "controller",
        "assistant controller",
        "bookkeeper",
        "finance manager",
        "accounting manager",
        "credit controller",
        "billing analyst",
        "revenue accountant",
        "investment operations",
        "fund accountant",
        "management accountant",
        "financial accountant",
        "transfer pricing",
        "pricing analyst",
        "commercial finance",
        "financial crime",
        "credit risk",
        "risk analyst",
        "risk manager",
        "compliance analyst",
        "compliance manager",
        "regulatory compliance",
    ]

    hard_negative_title_terms = [
        "sales account manager",
        "account executive",
        "account director",
        "customer account manager",
        "bank teller",
        "branch advisor",
        "cashier",
    ]

    finance_context_terms = [
        "general ledger",
        "month-end",
        "month end",
        "journal entries",
        "reconciliation",
        "accounts payable",
        "accounts receivable",
        "payroll",
        "tax",
        "vat",
        "treasury",
        "audit",
        "financial reporting",
        "management accounts",
        "statutory accounts",
        "balance sheet",
        "profit and loss",
        "p&l",
        "fp&a",
        "forecasting",
        "budgeting",
        "controller",
        "sap",
        "oracle finance",
        "invoice",
        "invoicing",
        "billing",
        "credit control",
        "credit risk",
        "financial crime",
        "aml",
        "kyc",
        "compliance",
        "regulatory",
        "risk management",
        "investment operations",
        "fund accounting",
        "transfer pricing",
    ]

    if any(term in title for term in hard_negative_title_terms):
        return False

    if any(term in title for term in positive_title_terms):
        return True

    if (
        "finance" in title
        or "financial" in title
        or "accounting" in title
        or "accountant" in title
        or "billing" in title
        or "credit controller" in title
        or "risk" in title
        or "compliance" in title
        or "financial crime" in title
    ) and any(term in text for term in finance_context_terms):
        return True

    return False


def detect_relevant_general_business_role(position_name: str, description: str) -> bool:
    title = lower_text(position_name)
    text = lower_text(f"{position_name}\n{description}")

    business_title_terms = [
        "business analyst",
        "business operations",
        "business support",
        "operations analyst",
        "operations associate",
        "operations manager",
        "operations coordinator",
        "data administrator",
        "administrator",
        "project manager",
        "project coordinator",
        "project lead",
        "program manager",
        "programme manager",
        "pmo",
        "change manager",
        "change analyst",
        "transformation",
        "implementation manager",
        "procurement",
        "buyer",
        "sourcing manager",
        "supply chain analyst",
        "legal counsel",
        "commercial counsel",
        "legal assistant",
        "legal specialist",
        "ai legal specialist",
        "legal operations",
        "paralegal",
        "company secretary",
        "contracts manager",
        "contract manager",
        "privacy",
        "privacy counsel",
        "privacy specialist",
        "data protection",
        "data protection officer",
        "governance risk compliance",
        "grc",
        "compliance specialist",
        "compliance officer",
        "regulatory specialist",
        "human resources",
        "hr generalist",
        "hr advisor",
        "hr business partner",
        "hr manager",
        "hr specialist",
        "hr coordinator",
        "people partner",
        "people advisor",
        "people specialist",
        "people ops",
        "people operations",
        "people experience",
        "employee relations",
        "learning and development",
        "l&d",
        "talent acquisition",
        "talent partner",
        "recruiter",
        "recruitment coordinator",
        "recruitment partner",
        "total rewards",
        "reward analyst",
        "compensation and benefits",
        "executive assistant",
        "personal assistant",
        "chief of staff",
        "office manager",
        "marketing manager",
        "marketing executive",
        "content marketing",
        "digital marketing",
        "product marketing",
        "brand marketing",
        "communications manager",
        "communications executive",
        "communications lead",
        "communications specialist",
        "communications officer",
        "communications coordinator",
        "corporate communications",
        "internal communications",
        "external communications",
        "pr manager",
        "pr lead",
        "public relations",
        "public relations lead",
        "public affairs",
        "media planner",
        "media planning",
        "growth marketing",
        "crm manager",
        "crm executive",
        "events manager",
        "events executive",
        "events coordinator",
        "community manager",
        "influencer marketing",
        "social media",
        "quality assurance",
        "qa analyst",
        "qa engineer",
        "quality engineer",
    ]

    if any(term in title for term in business_title_terms):
        return not _has_retail_store_context(text)

    return False


def detect_allowed_corporate_role(position_name: str, description: str) -> bool:
    return (
        detect_relevant_business_sales_role(position_name, description)
        or detect_relevant_finance_accounting_role(position_name, description)
        or detect_relevant_general_business_role(position_name, description)
    )


def title_has_leadership_signal(position_name: str) -> bool:
    title = lower_text(position_name)
    leadership_terms = [
        "head of",
        "director",
        "vice president",
        "vp ",
        "chief ",
        "engineering manager",
        "technical director",
        "cto",
        "cfo",
        "coo",
        "cio",
        "cmo",
        "cro",
        "cpo",
    ]
    return any(term in title for term in leadership_terms)


def _canonical_location_key(text: str) -> str:
    text = lower_text(text)
    text = text.replace("&", " and ")
    text = re.sub(r"\bgreat britain\b", " united kingdom ", text)
    text = re.sub(r"\bu\.?k\.?\b", " united kingdom ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _location_specific_tokens(text: str) -> list[str]:
    key = _canonical_location_key(text)
    if not key:
        return []

    tokens = []
    for token in key.split():
        if token in GENERIC_LOCATION_TOKENS:
            continue
        if len(token) <= 2:
            continue
        tokens.append(token)
    return tokens


def is_explicitly_foreign_location_text(value: str) -> bool:
    value_l = lower_text(value)
    return any(term in value_l for term in DISALLOWED_LOCATION_TERMS)


def has_disallowed_location_signal(text: str) -> bool:
    text_l = lower_text(get_primary_text_window(text))
    if not text_l:
        return False

    location_value_patterns = [
        r"(?im)^\s*location(?: city)?\s*[:\-]\s*(.+)$",
        r"(?im)^\s*job location\s*[:\-]\s*(.+)$",
        r"(?im)^\s*work location\s*[:\-]\s*(.+)$",
        r"(?im)^\s*city\s*[:\-]\s*(.+)$",
        r"(?im)^\s*based in\s+(.+)$",
        r"(?im)^\s*where you[’']ll work\s*[:\-]?\s*(.+)$",
    ]

    for pattern in location_value_patterns:
        for match in re.finditer(pattern, text_l):
            value = (match.group(1) or "").strip()
            if is_explicitly_foreign_location_text(value):
                return True

    disallowed_group = (
        r"philippines|metro manila|makati|bonifacio global city|taguig|"
        r"united states|usa|canada|australia|new south wales|nsw|north ryde|"
        r"germany|france|spain|italy|netherlands|belgium|sweden|norway|"
        r"denmark|finland|switzerland|austria|poland|portugal|india|"
        r"singapore|japan|china"
    )

    strong_actual_location_patterns = [
        rf"\b(?:location|job location|work location|city|based in|role is based in|position is based in)\b.{{0,100}}\b(?:{disallowed_group})\b",
        rf"\b(?:{disallowed_group})\b.{{0,80}}\b(?:office based|onsite|on-site|hybrid|work location|job location)\b",
        r"\bnorth ryde\b.{0,40}\b(?:nsw|australia)\b",
        r"\b(?:nsw|new south wales)\b.{0,40}\baustralia\b",
    ]

    for pattern in strong_actual_location_patterns:
        if re.search(pattern, text_l, flags=re.IGNORECASE | re.DOTALL):
            return True

    remote_block_patterns = [
        "remote apac",
        "apac only",
        "asia only",
        "remote asia",
        "latam only",
        "remote latam",
        "africa only",
        "remote africa",
        "usa only",
        "us only",
        "canada only",
        "australia only",
        "remote australia",
    ]
    return any(term in text_l for term in remote_block_patterns)


def normalize_location_match(value: str, allowed_locations: list[str]) -> str:
    value_key = _canonical_location_key(value)
    if not value_key:
        return ""

    exact_map = {_canonical_location_key(x): x for x in allowed_locations}
    if value_key in exact_map:
        return exact_map[value_key]

    if is_explicitly_foreign_location_text(value):
        return ""

    value_specific_tokens = set(_location_specific_tokens(value))
    if not value_specific_tokens:
        return ""

    best_value = ""
    best_score = -10**9

    for loc in allowed_locations:
        loc_key = _canonical_location_key(loc)
        loc_specific_tokens = set(_location_specific_tokens(loc))

        overlap = value_specific_tokens & loc_specific_tokens
        if not overlap:
            continue

        score = 0

        if value_key == loc_key:
            score += 5000

        if value_specific_tokens == loc_specific_tokens:
            score += 2000

        if value_specific_tokens.issubset(loc_specific_tokens):
            score += 700

        score += len(overlap) * 250

        extra_candidate_tokens = len(loc_specific_tokens - value_specific_tokens)
        score -= extra_candidate_tokens * 160

        if "," in loc:
            score += 40

        if len(loc_specific_tokens) == len(value_specific_tokens):
            score += 250

        if score > best_score:
            best_score = score
            best_value = loc

    if best_score >= 300:
        return best_value

    return ""


_SAFE_SKILL_ALIASES = {
    "PostgreSQL": [r"\bpostgresql\b", r"\bpostgres\b", r"\bsql/postgres\b"],
    "Machine Learning": [r"\bmachine learning\b", r"\bml\b", r"\bllms?\b", r"\blarge language models?\b"],
    "Artificial Intelligence": [r"\bartificial intelligence\b", r"\bai\b", r"\bllms?\b", r"\blarge language models?\b"],
    "Data Visualisation": [r"\bdata visuali[sz]ation\b", r"\bvisuali[sz]ation\b"],
    "Data Visualization": [r"\bdata visuali[sz]ation\b", r"\bvisuali[sz]ation\b"],
    "Data Driven": [r"\bdata[- ]driven\b"],
    "Performance Reporting": [r"\bperformance reporting\b", r"\bperformance data\b"],
    "VAT": [r"\bvat\b"],
    "BACS": [r"\bbacs\b"],
    "Accounts Payable": [r"\baccounts payable\b", r"\bap\b"],
    "Accounts Receivable": [r"\baccounts receivable\b", r"\bar\b"],
    "PyTorch": [r"\bpytorch\b"],
    "Tensorflow": [r"\btensorflow\b"],
    "SQL": [r"\bsql\b"],
    "Excel": [r"\bexcel\b"],
    "GitHub": [r"\bgithub\b"],
    "LinkedIn": [r"\blinkedin\b"],
    "R": [r"\br programming\b", r"\busing r\b", r"\bexperience with r\b", r"\br language\b"],
    "Flutter": [r"\bflutter framework\b", r"\bflutter development\b", r"\bflutter sdk\b", r"\bdart/flutter\b"],
    "Project Management": [r"\bproject management\b", r"\bmanage projects\b", r"\bdelivery of critical projects\b"],
    "Project Management Tools": [r"\bproject management tools\b", r"\bproject tools\b", r"\broadmap\b", r"\bplanned delivery\b"],
    "Business Analysis": [r"\bbusiness analyst\b", r"\bbusiness analysis\b", r"\brequirements\b", r"\brequirements elicitation\b"],
    "Graphic Design": [r"\bgraphic design\b", r"\bbrand designer\b", r"\bvisual design\b"],
    "Brand Marketing": [r"\bbrand marketing\b", r"\bbrand identity\b", r"\bbrand designer\b"],
    "Business Development": [r"\bbusiness development\b"],
    "Lead Generation": [r"\blead generation\b", r"\boutbound lead generation\b"],
    "Sales": [r"\bsales\b", r"\bsales pipeline\b", r"\bsales team\b"],
}


def skill_is_supported(skill: str, text: str) -> bool:
    if not skill or not text:
        return False

    text_l = lower_text(text)
    skill_n = normalize_text(skill)

    alias_patterns = _SAFE_SKILL_ALIASES.get(skill_n, [])
    if alias_patterns:
        return any(re.search(pattern, text_l, flags=re.IGNORECASE) for pattern in alias_patterns)

    skill_key = re.escape(lower_text(skill_n))
    if re.search(rf"(?<![a-z0-9]){skill_key}(?![a-z0-9])", text_l, flags=re.IGNORECASE):
        return True

    return False


def extract_deterministic_skills(position_name: str, description: str, allowed_skills: list[str], max_items: int = 10) -> list[str]:
    full_text = f"{position_name}\n{description}"

    found = []
    full_text_l = lower_text(full_text)

    for skill in allowed_skills:
        if skill_is_supported(skill, full_text):
            idx = full_text_l.find(lower_text(skill))
            if idx < 0:
                idx = 999999
            found.append((skill, idx))

    found.sort(key=lambda x: x[1])
    return dedupe_keep_order([skill for skill, _ in found])[:max_items]


def infer_skills_from_position_context(position_name: str, description: str, allowed_skills: list[str], max_items: int = 4) -> list[str]:
    title = lower_text(position_name)
    text = lower_text(f"{position_name}\n{description}")

    candidate_map = [
        (["project manager", "project coordinator", "project lead"], ["Project Management", "Project Management Tools"]),
        (["business analyst"], ["Business Analysis", "Project Management"]),
        (["brand designer", "designer"], ["Graphic Design", "Brand Marketing"]),
        (["accounts payable", "accounts payable assistant"], ["Accounts Payable", "VAT", "Excel", "BACS"]),
        (["analytics manager", "data analyst", "insight analyst"], ["SQL", "Data Visualisation", "Performance Reporting", "Data Driven"]),
        (["business development", "bdr", "sdr", "account executive"], ["Business Development", "Lead Generation", "Sales"]),
        (["finance analyst", "financial analyst", "accountant", "accounting analyst"], ["Excel", "Performance Reporting"]),
    ]

    allowed_lookup = {lower_text(x): x for x in allowed_skills}
    out = []

    for triggers, skills in candidate_map:
        if any(trigger in title for trigger in triggers):
            for skill in skills:
                key = lower_text(skill)
                if key in allowed_lookup and allowed_lookup[key] not in out:
                    out.append(allowed_lookup[key])

    if not out and "llm" in text:
        for skill in ["Machine Learning", "Artificial Intelligence"]:
            key = lower_text(skill)
            if key in allowed_lookup and allowed_lookup[key] not in out:
                out.append(allowed_lookup[key])

    return out[:max_items]


def infer_job_titles_from_position_name(position_name: str, allowed_job_titles: list[str]) -> list[str]:
    title = lower_text(position_name)
    allowed_lookup = {lower_text(x): x for x in allowed_job_titles}

    def pick(candidates: list[str]) -> list[str]:
        out = []
        for candidate in candidates:
            key = lower_text(candidate)
            if key in allowed_lookup and allowed_lookup[key] not in out:
                out.append(allowed_lookup[key])
        return out

    if title in allowed_lookup:
        return [allowed_lookup[title]]

    rules = [
        (["1st line", "first line", "2nd line", "second line", "3rd line", "third line"], ["Support Engineer", "System Administrator", "System Engineer"]),
        (["it engineer", "it support engineer", "support engineer", "technical support engineer"], ["Support Engineer", "System Engineer", "System Administrator"]),
        (["application engineer"], ["System Engineer", "Support Engineer", "Solutions Engineer"]),
        (["systems engineer", "system engineer"], ["System Engineer", "Support Engineer"]),
        (["network engineer"], ["Network Engineer", "System Engineer"]),
        (["administrator"], ["System Administrator", "Operations"]),
        (["analytics manager"], ["Business Analyst", "Data/Insight Analyst"]),
        (["accounts payable", "accounts payable assistant"], ["Finance/Accounting", "Operations"]),
        (["finance analyst", "financial analyst", "accountant", "accounting analyst"], ["Finance/Accounting", "Operations"]),
        (["brand designer", "brand design", "brand design lead", "design lead"], ["Graphic Designer", "Brand Marketing"]),
        (["business development", "bdr", "sdr"], ["SDR/BDR", "Business Development Manager", "Account Executive"]),
    ]

    for triggers, candidates in rules:
        if any(trigger in title for trigger in triggers):
            picked = pick(candidates)
            if picked:
                return picked[:3]

    contains_matches = []
    for allowed in allowed_job_titles:
        if lower_text(allowed) in title:
            contains_matches.append(allowed)

    deduped = dedupe_keep_order(contains_matches)
    if deduped:
        return deduped[:3]

    return []


def closest_salary_value(value: int, allowed_salaries: list[int]) -> str:
    if not allowed_salaries:
        return ""
    return str(min(allowed_salaries, key=lambda x: abs(x - value)))


def salary_context_exists(text: str) -> bool:
    t = lower_text(text)
    patterns = [
        r"\bsalary\b",
        r"\bcompensation\b",
        r"\bpay\b",
        r"\bpackage\b",
        r"£\s*\d",
        r"\$\s*\d",
        r"€\s*\d",
        r"\bgbp\b",
        r"\busd\b",
        r"\beur\b",
        r"\bcad\b",
        r"\bper annum\b",
        r"\bper year\b",
        r"\bannually\b",
    ]
    return any(re.search(p, t) for p in patterns)


def extract_remote_days(text: str) -> str:
    text_l = lower_text(get_primary_text_window(text))
    if not text_l:
        return "not specified"

    if re.search(r"\bfully remote\b|\b100% remote\b|\bremote-only\b|\bremote only\b|\bmostly async from anywhere\b", text_l):
        return "not specified"

    patterns = [
        (r"\b(?:works?|working)\s+on\s+site\s+four\s+days\s+a\s+week.*?\bone\s+flexible\s+work\s+from\s+home\s+day\b", "1"),
        (r"\bteam usually works on site four days a week.*?\bone\s+flexible\s+work\s+from\s+home\s+day\b", "1"),
        (r"\b4\s+days?\s+(?:a\s+week\s+)?(?:on[- ]site|in\s+the\s+office|on\s+site)\b.*?\b1\s+(?:day\s+)?(?:wfh|work\s+from\s+home|from\s+home)\b", "1"),
        (r"\b1\s+day\s+(?:a\s+week\s+)?(?:wfh|work\s+from\s+home|from\s+home)\b", "1"),
        (r"\b2\s+days?\s+(?:a\s+week\s+)?(?:wfh|work\s+from\s+home|from\s+home)\b", "2"),
        (r"\b3\s+days?\s+(?:a\s+week\s+)?(?:wfh|work\s+from\s+home|from\s+home)\b", "3"),
        (r"\b1\s*-\s*2\s+days?\s+in\s+the\s+office\b", "3"),
        (r"\b2\s*-\s*3\s+days?\s+in\s+the\s+office\b", "2"),
        (r"\b1\s+day\s+in\s+the\s+office\b", "4"),
        (r"\b2\s+days?\s+in\s+the\s+office\b", "3"),
        (r"\b3\s+days?\s+in\s+the\s+office\b", "2"),
        (r"\b4\s+days?\s+in\s+the\s+office\b", "1"),
    ]

    for pattern, value in patterns:
        if re.search(pattern, text_l, flags=re.IGNORECASE | re.DOTALL):
            return value

    return "not specified"


def extract_remote_preferences(text: str) -> list[str]:
    text_l = lower_text(get_primary_text_window(text))
    found = []

    strong_remote = [
        r"(?im)^\s*remote\s*$",
        r"\bfully remote\b",
        r"\bremote-only\b",
        r"\bremote only\b",
        r"\bwe work remotely\b",
        r"\bmostly async from anywhere\b",
        r"\bwork remotely and mostly async from anywhere\b",
    ]
    strong_hybrid = [
        r"\bhybrid\b",
        r"\bflexibility for occasional home working\b",
        r"\bwork from home day\b",
        r"\bhome working by agreement\b",
        r"\boccasional home working\b",
    ]
    strong_onsite = [
        r"\bon-site\b",
        r"\bonsite\b",
        r"\bon site\b",
        r"\bbased at our .* office\b",
        r"\bf/t site\b",
    ]

    if any(re.search(p, text_l, flags=re.IGNORECASE | re.DOTALL) for p in strong_remote):
        found.append("remote")

    if any(re.search(p, text_l, flags=re.IGNORECASE | re.DOTALL) for p in strong_hybrid):
        found.append("hybrid")

    if any(re.search(p, text_l, flags=re.IGNORECASE | re.DOTALL) for p in strong_onsite):
        found.append("onsite")

    if "remote" in found:
        if "hybrid" in found and not re.search(r"\bhybrid\b|\boccasional home working\b|\bwork from home day\b", text_l):
            found = [x for x in found if x != "hybrid"]
        if "onsite" in found and not re.search(r"\bon[- ]site\b|\bf/t site\b", text_l):
            found = [x for x in found if x != "onsite"]

    ordered = []
    for item in ["onsite", "hybrid", "remote"]:
        if item in found and item not in ordered:
            ordered.append(item)
    return ordered


def text_requires_non_english_language(text: str) -> bool:
    t = lower_text(text)

    language_names = (
        r"french|german|spanish|italian|dutch|portuguese|arabic|polish|swedish|norwegian|danish|finnish|czech|hungarian|romanian|bulgarian|croatian|serbian|slovak|slovenian|turkish|japanese|mandarin|cantonese|korean|thai|vietnamese|indonesian|malay"
    )

    negative_patterns = [
        rf"\bfluency in (?:{language_names})\b",
        rf"\b(?:{language_names}) and english\b",
        rf"\bbilingual\b.{{0,40}}\b(?:{language_names})\b",
        rf"\brequired language\b.{{0,40}}\b(?:{language_names})\b",
        rf"\bmust speak\b.{{0,40}}\b(?:{language_names})\b",
        rf"\b(?:{language_names})\s+required\b",
        rf"\bnative\s+(?:{language_names})\b",
        rf"\bbusiness[- ]level\s+(?:{language_names})\b",
        rf"\bprofessional[- ]level\s+(?:{language_names})\b",
        rf"\bproficiency in (?:{language_names})\b",
        rf"\bwritten and spoken\s+(?:{language_names})\b",
        rf"\benglish and (?:{language_names})\b",
        rf"\b(?:{language_names})\s+preferred\b",
    ]
    return any(re.search(p, t, flags=re.IGNORECASE) for p in negative_patterns)


def text_is_predominantly_non_english(text: str) -> bool:
    t = normalize_text(text)
    if not t:
        return False

    lower = t.lower()

    common_english_markers = [
        "responsibilities",
        "requirements",
        "location",
        "salary",
        "benefits",
        "about the role",
        "job description",
        "experience",
        "apply",
        "you will",
        "we are looking for",
        "job type",
    ]
    if sum(1 for x in common_english_markers if x in lower) >= 2:
        return False

    non_english_markers = [
        "responsabilidades",
        "requisitos",
        "ubicación",
        "salario",
        "beneficios",
        "puesto",
        "empleo",
        "experiencia requerida",
        "responsabilités",
        "exigences",
        "lieu",
        "salaire",
        "avantages",
        "stellenbeschreibung",
        "anforderungen",
        "standort",
        "gehalt",
        "descrizione",
        "requisiti",
        "posizione",
        "stipendio",
    ]
    if sum(1 for x in non_english_markers if x in lower) >= 2:
        return True

    accented = len(re.findall(r"[à-ÿÀ-Ÿ]", t))
    letters = len(re.findall(r"[A-Za-zÀ-Ÿ]", t))
    if letters >= 200 and accented / max(letters, 1) > 0.03:
        return True

    return False


def is_location_allowed(job_location: str, remote_preferences: list[str], source_text: str) -> bool:
    text = lower_text(source_text)
    if has_disallowed_location_signal(text):
        return False

    if not job_location or lower_text(job_location) == "unknown":
        return True

    loc = lower_text(job_location)
    remote_set = set(remote_preferences)

    uk_markers = ["uk", "united kingdom", "england", "scotland", "wales", "northern ireland"]
    if any(marker in loc for marker in uk_markers):
        return True

    if "ireland" in loc:
        return "remote" in remote_set

    if "europe" in loc or "emea" in loc:
        return "remote" in remote_set

    if "global" in loc or "worldwide" in loc:
        if "remote" not in remote_set:
            return False
        blocked_regions = [
            "remote apac",
            "apac only",
            "asia only",
            "remote asia",
            "latam only",
            "remote latam",
            "africa only",
            "remote africa",
            "usa only",
            "us only",
            "canada only",
            "australia only",
            "remote australia",
        ]
        return not any(term in text for term in blocked_regions)

    hard_disallowed = DISALLOWED_LOCATION_TERMS
    if any(term in loc for term in hard_disallowed):
        return False

    return True


def reason_strongly_says_not_relevant(reason: str) -> bool:
    r = lower_text(reason)
    if not r:
        return False

    hard_negative_signals = [
        "outside allowed",
        "outside allowed scope",
        "outside target scope",
        "not a real job posting",
        "educational",
        "informational",
        "checklist",
        "medical role",
        "clinical role",
        "patient care",
        "retail store role",
        "shop-floor role",
        "manufacturing / shop-floor role",
        "construction / civil engineering role",
        "rf test engineering role",
        "robotics technician",
        "hands-on robotics",
        "mechanical / electrical technician",
        "language other than english",
        "job description is primarily in another language",
        "location is not allowed",
        "outside allowed regions",
        "volunteer role",
        "usa only",
        "canada only",
        "philippines",
        "australia",
        "north ryde",
        "nsw",
        "apac only",
        "latam only",
        "africa only",
    ]

    return any(s in r for s in hard_negative_signals)
