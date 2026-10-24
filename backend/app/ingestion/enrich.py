"""Insurance-aware enrichment: definitions glossary, section category tags, table summaries."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from app.ingestion.tables import NormalisedTable

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- definitions glossary

DEFINITION_RE = re.compile(
    r"^\s*(?:[-•▪*]\s*)?(?:\d+(?:\.\d+)*[.)]?\s+|\(?[a-zA-Z]{1,4}[.)]\s+)?"  # optional numbering: "2.14", "a)", "(iv)"
    r"[\"“']?(?P<term>[A-Z][A-Za-z0-9'’()/&\- ]{1,60}?)[\"”']?\s*[:\-–]?\s+"
    r"(?P<verb>means|shall mean|refers to|is defined as|shall include|includes)\b\s*(?P<definition>.+)",
    re.S,
)
DEFINITIONS_HEADING_RE = re.compile(r"\bdefinitions?\b|\binterpretation\b|\bglossary\b", re.I)


@dataclass
class Definition:
    term: str
    definition: str
    page: int | None
    section: str


def extract_definition(text: str) -> tuple[str, str] | None:
    m = DEFINITION_RE.match(text)
    if not m:
        return None
    term = m.group("term").strip(" .:-")
    if len(term.split()) > 8:
        return None
    return term, f"{term} {m.group('verb')} {m.group('definition').strip()}"


def find_terms(text: str, glossary: dict[str, Definition], limit: int = 5) -> list[Definition]:
    """Defined terms mentioned in `text`, longest first (so "Pre-existing Disease" beats "Disease")."""
    found: list[Definition] = []
    lowered = text.lower()
    for key in sorted(glossary, key=len, reverse=True):
        if re.search(rf"(?<![\w-]){re.escape(key)}(?![\w-])", lowered):
            if not any(key in d.term.lower() for d in found):
                found.append(glossary[key])
                if len(found) >= limit:
                    break
    return found


# --------------------------------------------------------------------------- category tags

CATEGORIES: dict[str, list[str]] = {
    "coverage": [r"\bcover(ed|age|s)?\b", r"\bbenefits?\b", r"\bin-?patient\b", r"\bhospitali[sz]ation\b", r"\bday ?care\b",
                 r"\bdomiciliary\b", r"\bmaternity\b", r"\bambulance\b", r"\bayush\b", r"\borgan donor\b", r"\bpre-? ?hospitali", r"\bpost-? ?hospitali"],
    "exclusion": [r"\bexclu(ded|sions?|des)\b", r"\bnot (be )?(covered|payable|admissible)\b", r"\bshall not\b", r"\bpermanent exclusion"],
    "waiting_period": [r"\bwaiting periods?\b", r"\bpre-?existing\b", r"\bped\b", r"\bcooling period\b", r"\bmoratorium\b", r"\b(30|thirty) days\b", r"\bspecific diseases?\b"],
    "sub_limit": [r"\bsub-? ?limits?\b", r"\broom rent\b", r"\bicu\b", r"\bcapped\b", r"\bmaximum of\b", r"\bup to\b", r"\b% of (the )?sum insured\b", r"\blimit(s|ed)?\b"],
    "co_pay": [r"\bco-? ?pay(ment)?s?\b", r"\bdeductibles?\b", r"\bcost.?shar"],
    "claims": [r"\bclaims?\b", r"\bcashless\b", r"\breimbursement\b", r"\btpa\b", r"\bnetwork hospital", r"\bpre-?authori[sz]ation\b", r"\bintimation\b"],
    "renewal": [r"\brenew(al|able)?\b", r"\bgrace period\b", r"\bportability\b", r"\bmigration\b", r"\bfree look\b", r"\bcancell?ation\b", r"\bno claim bonus\b", r"\bcumulative bonus\b"],
    "premium": [r"\bpremiums?\b", r"\bgst\b", r"\binstal?ments?\b", r"\bloading\b", r"\bdiscounts?\b"],
    "definitions": [r"\bdefinitions?\b", r"\bmeans\b", r"\bshall mean\b"],
}
_COMPILED = {cat: [re.compile(p, re.I) for p in pats] for cat, pats in CATEGORIES.items()}


def tag_categories(text: str, section_path: list[str]) -> list[str]:
    """Keyword tagging. Heading matches count double because headings are the strongest signal."""
    heading = " ".join(section_path)
    scores: dict[str, int] = {}
    for cat, patterns in _COMPILED.items():
        score = sum(2 for p in patterns if p.search(heading)) + sum(1 for p in patterns if p.search(text))
        if score:
            scores[cat] = score
    if not scores:
        return ["general"]
    top = max(scores.values())
    return sorted(c for c, s in scores.items() if s >= max(2, top // 2)) or [max(scores, key=scores.get)]


def query_categories(query: str) -> set[str]:
    cats = set(tag_categories(query, []))
    cats.discard("general")
    cats.discard("definitions")
    if "coverage" in cats or re.search(r"\b(is|are|does|do)\b.*\b(covered|cover|included|payable)\b", query, re.I):
        cats.update({"coverage", "exclusion", "waiting_period", "sub_limit"})  # coverage answers need the caveats
    return cats


LLM_TAG_PROMPT = """Classify each numbered passage from a health insurance policy into one or more of these
categories: {cats}. Reply with one line per passage: "<number>: cat1, cat2". Use "general" if none apply.

{passages}"""


def llm_tag(passages: list[str], llm: BaseChatModel) -> list[list[str]]:
    cats = ", ".join(list(CATEGORIES) + ["general"])
    body = "\n\n".join(f"{i + 1}. {p[:800]}" for i, p in enumerate(passages))
    reply = llm.invoke([HumanMessage(content=LLM_TAG_PROMPT.format(cats=cats, passages=body))])
    result = [["general"] for _ in passages]
    for line in str(reply.content).splitlines():
        m = re.match(r"\s*(\d+)\s*[:.)-]\s*(.+)", line)
        if m and 0 < int(m.group(1)) <= len(passages):
            tags = [t.strip().lower() for t in m.group(2).split(",") if t.strip().lower() in CATEGORIES]
            if tags:
                result[int(m.group(1)) - 1] = tags
    return result


# --------------------------------------------------------------------------- table summaries

TABLE_SUMMARY_PROMPT = """Below is a table extracted from a health insurance policy document (section: {section}).
Write a 2-4 sentence summary that a search engine can match against user questions: say what the table
is about, which plans/variants or columns it compares, which benefits/rows it lists, and the most
important limits, waiting periods or percentages. Only use facts present in the table.

<table>
{markdown}
</table>"""


def summarise_tables(tables: list[NormalisedTable], llm: BaseChatModel | None, max_concurrency: int) -> int:
    """Fill NormalisedTable.summary in place; returns how many came from the LLM."""
    done = 0
    if llm is not None and tables:
        prompts = [
            [HumanMessage(content=TABLE_SUMMARY_PROMPT.format(section=" › ".join(t.section_path) or "unknown", markdown=t.to_markdown(0, 60)[:12000]))]
            for t in tables
        ]
        results = llm.batch(prompts, config={"max_concurrency": max_concurrency}, return_exceptions=True)
        for t, res in zip(tables, results):
            if isinstance(res, Exception):
                log.warning("Table summary failed: %s", type(res).__name__)
                continue
            text = res.content if isinstance(res.content, str) else str(res.text)
            if text.strip():
                t.summary = text.strip()
                done += 1
    for t in tables:
        if not t.summary:
            t.summary = t.fallback_summary()
    return done
