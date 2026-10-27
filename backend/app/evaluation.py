"""Live answer evaluation: groundedness, context relevance, answer relevance and a composite
confidence, computed after each answer has streamed and recorded as Langfuse scores.

One LLM-as-judge call returns every LLM-graded metric (the "RAG triad": context relevance,
groundedness, answer relevance). Cheap heuristics (citation validity/coverage, retrieval
strength, "not in document" detection) need no LLM and are always computed.

LLMs do not expose a calibrated confidence, so `confidence` is a weighted composite of the
measurable signals. See docs/OBSERVABILITY.md for the formulas and how to read them.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from app.store import Hit

log = logging.getLogger(__name__)

NOT_IN_DOC_RE = re.compile(r"does not specify", re.I)


def is_refusal(answer: str) -> bool:
    """The answer declines because the document lacks the information. The prompt makes the model open
    with "The policy document does not specify this.", so only the first line counts: a complete answer
    that mentions "does not specify" in a side caveat is not a refusal."""
    first_line = answer.strip().split("\n", 1)[0][:200]
    return bool(NOT_IN_DOC_RE.search(first_line))
CITATION_RE = re.compile(r"\[Source\s*([\d\s,]+)\]", re.I)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

CONFIDENCE_WEIGHTS = {"groundedness": 0.5, "context_relevance": 0.25, "answer_relevance": 0.15, "citation_validity": 0.1}
HIGH, MEDIUM = 0.8, 0.5

JUDGE_PROMPT = """You are grading an answer written by an assistant that must answer ONLY from excerpts of a
health insurance policy document. Be strict and literal.

<question>
{question}
</question>

<sources>
{sources}
</sources>

<answer>
{answer}
</answer>

Tasks:
1. Split the answer into atomic factual claims about the policy (amounts, limits, periods, conditions,
   coverage yes/no, procedures). Ignore citations, formatting, and the sentence "The policy document does
   not specify this." itself. For each claim decide if the sources explicitly support it (supported=true)
   or not (supported=false: contradicted, missing, or a guess). List the source numbers that support it.
2. Rate EVERY source for how relevant it is to the question: 0 = unrelated, 1 = partially related,
   2 = directly answers or constrains the answer.
3. Rate the answer's relevance to the question: 0 = does not address it, 1 = partially, 2 = fully
   (a correct "the document does not specify this" counts as 2 when the sources indeed lack it).

Reply with ONLY this JSON, no prose:
{{"claims": [{{"claim": "...", "supported": true, "sources": [1]}}],
 "source_relevance": [{{"source": 1, "score": 2}}],
 "answer_relevance": 2,
 "reason": "one short sentence"}}"""


@dataclass
class EvalResult:
    groundedness: float | None = None
    context_relevance: float | None = None
    answer_relevance: float | None = None
    citation_validity: float = 0.0
    citation_coverage: float = 0.0
    retrieval_strength: float = 0.0
    not_in_document: bool = False
    confidence: float | None = None
    confidence_label: str | None = None
    unsupported_claims: list[str] = field(default_factory=list)
    irrelevant_sources: list[int] = field(default_factory=list)
    source_scores: dict[int, int] = field(default_factory=dict)
    n_claims: int = 0
    reason: str = ""
    judge_error: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- heuristics (no LLM)

def cited_sources(answer: str) -> list[int]:
    nums: list[int] = []
    for group in CITATION_RE.findall(answer):
        nums.extend(int(n) for n in re.findall(r"\d+", group))
    return nums


def citation_validity(answer: str, n_sources: int, not_in_doc: bool) -> float:
    nums = cited_sources(answer)
    if not nums:
        return 1.0 if not_in_doc else 0.0  # a refusal need not cite; a factual answer must
    return sum(1 for n in nums if 1 <= n <= n_sources) / len(nums)


def citation_coverage(answer: str) -> float:
    """Share of substantive sentences/lines that carry at least one [Source N] tag."""
    parts = [p.strip() for p in SENTENCE_SPLIT_RE.split(answer)]
    parts = [p for p in parts if len(p) >= 25 and not p.startswith("#") and not NOT_IN_DOC_RE.search(p)]
    if not parts:
        return 1.0
    return sum(1 for p in parts if CITATION_RE.search(p)) / len(parts)


def retrieval_strength(hits: list[Hit]) -> float:
    """0.6 x best dense similarity (clamped to 0..1) + 0.4 x share of the top 3 parents that both the
    dense and the BM25 retriever found. High when both retrievers agree on strong matches."""
    if not hits:
        return 0.0
    sims = [h.dense_sim for h in hits if h.dense_sim is not None]
    best_sim = max(0.0, min(1.0, max(sims))) if sims else 0.0
    top = hits[:3]
    agree = sum(1 for h in top if h.dense_rank is not None and h.bm25_rank is not None) / len(top)
    return round(0.6 * best_sim + 0.4 * agree, 4)


# --------------------------------------------------------------------------- judge

def parse_json_object(text: str) -> dict:
    """Extract the JSON object from an LLM reply (tolerates code fences and stray prose)."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("reply contained no JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("reply JSON is not an object")
    return value


parse_judgement = parse_json_object  # the judge's reply is one JSON object


def apply_judgement(result: EvalResult, judgement: dict, n_sources: int) -> None:
    claims = judgement.get("claims") or []
    result.n_claims = len(claims)
    if claims:
        supported = [c for c in claims if c.get("supported") is True]
        result.groundedness = round(len(supported) / len(claims), 4)
        result.unsupported_claims = [str(c.get("claim", ""))[:300] for c in claims if c.get("supported") is not True]
    elif result.not_in_document:
        result.groundedness = 1.0  # a pure "not specified" answer makes no claims to be wrong about
    scores: dict[int, int] = {}
    for item in judgement.get("source_relevance") or []:
        try:
            n, s = int(item.get("source")), int(item.get("score"))
        except (TypeError, ValueError):
            continue
        if 1 <= n <= n_sources:
            scores[n] = max(0, min(2, s))
    if n_sources:
        # Sources the judge forgot to rate count as unrelated.
        result.context_relevance = round(sum(scores.get(n, 0) for n in range(1, n_sources + 1)) / (2 * n_sources), 4)
        result.irrelevant_sources = [n for n in range(1, n_sources + 1) if scores.get(n, 0) == 0]
    result.source_scores = scores
    try:
        result.answer_relevance = round(max(0, min(2, int(judgement.get("answer_relevance")))) / 2, 4)
    except (TypeError, ValueError):
        result.answer_relevance = None
    result.reason = str(judgement.get("reason", ""))[:300]


def compute_confidence(result: EvalResult) -> None:
    parts = {
        "groundedness": result.groundedness,
        "context_relevance": result.context_relevance,
        "answer_relevance": result.answer_relevance,
        "citation_validity": result.citation_validity,
    }
    if any(v is None for v in parts.values()):
        return  # no judge verdict: leave confidence unset rather than guess
    score = sum(CONFIDENCE_WEIGHTS[k] * v for k, v in parts.items())
    result.confidence = round(score, 4)
    result.confidence_label = "High" if score >= HIGH else "Medium" if score >= MEDIUM else "Low"


async def evaluate_answer(question: str, sources_block: str, answer: str, hits: list[Hit],
                          judge: BaseChatModel | None, callbacks: list | None = None) -> EvalResult:
    n_sources = len(hits)
    result = EvalResult(not_in_document=is_refusal(answer))
    result.citation_validity = round(citation_validity(answer, n_sources, result.not_in_document), 4)
    result.citation_coverage = round(citation_coverage(answer), 4)
    result.retrieval_strength = retrieval_strength(hits)
    if judge is None:
        return result
    prompt = JUDGE_PROMPT.format(question=question, sources=sources_block or "(no sources)", answer=answer)
    last_error: Exception | None = None
    for attempt in range(2):  # one retry when the reply isn't valid JSON
        try:
            reply = await judge.ainvoke([HumanMessage(content=prompt)], config={"callbacks": callbacks or [], "run_name": "judge"})
            apply_judgement(result, parse_judgement(reply.text), n_sources)
            last_error = None
            break
        except Exception as exc:  # noqa: BLE001 - evaluation must never affect the user
            last_error = exc
    if last_error is not None:
        result.judge_error = f"{type(last_error).__name__}: {last_error}"[:300]
    compute_confidence(result)
    return result


# --------------------------------------------------------------------------- recording

NUMERIC_SCORES = ("groundedness", "context_relevance", "answer_relevance", "citation_validity",
                  "citation_coverage", "retrieval_strength", "confidence")


def record_scores(trace, result: EvalResult) -> None:
    """Push the result to the trace as Langfuse scores. Text in comments is for developers only."""
    comments = {
        "groundedness": ("Unsupported: " + " | ".join(result.unsupported_claims)) if result.unsupported_claims else None,
        "context_relevance": f"Irrelevant sources: {result.irrelevant_sources}" if result.irrelevant_sources else None,
        "answer_relevance": result.reason or None,
    }
    for name in NUMERIC_SCORES:
        value = getattr(result, name)
        if value is not None:
            trace.score(name, float(value), comment=comments.get(name), data_type="NUMERIC")
    trace.score("not_in_document", result.not_in_document)
    if result.confidence_label:
        trace.score("confidence_label", result.confidence_label, data_type="CATEGORICAL")
    if result.judge_error:
        trace.score("eval_error", True, comment=result.judge_error)
