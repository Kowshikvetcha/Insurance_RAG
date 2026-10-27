"""Policy fit score: how well the uploaded policy's terms fit one person's health profile, 1-10.

A fixed rubric keeps the score stable and explainable. For each dimension that applies to the
person, we retrieve the relevant parents, the LLM rates the fit 1-10 with cited evidence (or null
when the document is silent), and `combine()` turns the ratings into one weighted score in code.
A final streamed LLM call explains the score; it never changes it.

The profile is sensitive health data: it lives only for the request. It is never stored on the
session or written to the Python logger, and trace inputs get `redacted_profile()` only.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass, field
from typing import AsyncIterator, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app import observability as obs
from app.config import Settings
from app.evaluation import parse_json_object
from app.prompts import FIT_DIMENSION_TEMPLATE, FIT_SUMMARY_TEMPLATE, FIT_SYSTEM_PROMPT
from app.providers import Reranker
from app.rag import citation, glossary_block, retrieve, source_block
from app.schemas import HealthProfile
from app.store import Hit, Session

log = logging.getLogger(__name__)

MAX_SOURCES_PER_DIMENSION = 6
MAX_DIMENSION_CONTEXT_CHARS = 12_000
LLM_CONCURRENCY = 4
DEAL_BREAKER_CAP = 4.0
BANDS = ((9.0, "Excellent fit"), (7.0, "Good fit"), (4.0, "Partial fit"), (0.0, "Poor fit"))


@dataclass(frozen=True)
class Dimension:
    key: str
    label: str
    instructions: str
    queries: Callable[[HealthProfile], list[str]]
    weight: Callable[[HealthProfile], float]
    applies: Callable[[HealthProfile], bool] = lambda p: True


def _members(p: HealthProfile) -> int:
    return len(set(p.covered_members))


DIMENSIONS: tuple[Dimension, ...] = (
    Dimension(
        "pre_existing", "Pre-existing conditions",
        "Whether the person's declared conditions are covered, after how long a pre-existing disease "
        "waiting period, and any condition-specific limits or loadings. With no declared conditions, "
        "assess how the policy treats pre-existing diseases in general.",
        lambda p: ["pre-existing disease waiting period coverage"]
        + [f"{c} coverage waiting period" for c in p.conditions[:4]],
        lambda p: 3.0 if p.conditions else 0.5,
    ),
    Dimension(
        "waiting_periods", "Waiting periods",
        "Initial waiting period and specific-disease or procedure waiting periods, especially for any "
        "procedures the person plans.",
        lambda p: ["initial waiting period specific disease waiting period list"]
        + [f"{x} waiting period" for x in p.planned_procedures[:3]],
        lambda p: 2.5 if p.planned_procedures else 1.5,
    ),
    Dimension(
        "sum_insured", "Sum insured adequacy",
        "Whether the available sum insured suits the person's city, family size, age and desired sum "
        "insured, including restoration, recharge and cumulative bonus.",
        lambda p: ["sum insured options restoration recharge cumulative bonus"],
        lambda p: 2.0 if _members(p) >= 3 or p.city_tier == "metro" else 1.5,
    ),
    Dimension(
        "cost_sharing", "Co-payment and deductibles",
        "Co-payment (including age-linked or zone-linked co-pay) and deductibles that would apply to "
        "this person's claims.",
        lambda p: ["co-payment percentage deductible", "co-payment age zone"],
        lambda p: 2.5 if p.avoid_copay or p.age >= 60 or "parents" in p.covered_members else 1.5,
    ),
    Dimension(
        "sub_limits", "Room rent and sub-limits",
        "Room rent and ICU limits, and disease- or procedure-wise sub-limits on what this person is "
        "likely to claim for.",
        lambda p: ["room rent limit ICU charges", "sub-limit capping procedure"]
        + [f"{x} sub-limit" for x in p.planned_procedures[:2]],
        lambda p: 2.5 if p.avoid_room_rent_limit or p.planned_procedures else 1.5,
    ),
    Dimension(
        "exclusions", "Exclusions that affect you",
        "Exclusions that would hit this person: their conditions, smoking or alcohol, hazardous "
        "activities, and anything else in their profile.",
        lambda p: ["permanent exclusions general exclusions"]
        + (["tobacco smoking alcohol substance abuse exclusion"] if p.smoker or p.alcohol == "regular" else [])
        + (["hazardous adventure sports exclusion"] if p.hazardous_activity.strip() else []),
        lambda p: 2.5 if p.conditions or p.smoker or p.hazardous_activity.strip() else 2.0,
    ),
    Dimension(
        "maternity", "Maternity",
        "Maternity and newborn cover, its waiting period and any limits, for a person planning a pregnancy.",
        lambda p: ["maternity benefit waiting period", "newborn baby cover delivery caesarean limit"],
        lambda p: 2.5,
        applies=lambda p: p.planning_pregnancy,
    ),
    Dimension(
        "extras", "OPD, dental and AYUSH",
        "Cover for the extras the person asked for: outpatient (OPD) consultations, dental and AYUSH "
        "treatment.",
        lambda p: [q for want, q in ((p.wants_opd, "OPD outpatient consultation cover"),
                                     (p.wants_dental, "dental treatment cover"),
                                     (p.wants_ayush, "AYUSH treatment cover")) if want],
        lambda p: 1.0,
        applies=lambda p: p.wants_opd or p.wants_dental or p.wants_ayush,
    ),
    Dimension(
        "renewal_age", "Age limits and renewal",
        "Entry and exit age limits for everyone to be covered, lifelong renewability, and age-based "
        "premium loading or benefit changes.",
        lambda p: ["entry age exit age renewal", "lifelong renewability age loading"],
        lambda p: 2.0 if p.age >= 55 or "parents" in p.covered_members else 1.0,
    ),
)


@dataclass
class DimensionResult:
    key: str
    label: str
    weight: float
    status: str = "rated"  # rated | not_specified | failed
    rating: int | None = None
    deal_breaker: bool = False
    verdict: str = ""
    evidence: list[dict] = field(default_factory=list)
    sources: list[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class FitScore:
    score: float | None
    band: str
    confidence: str  # High | Medium | Low
    rated_weight_share: float
    capped: bool
    deal_breakers: list[str]
    not_specified: list[str]
    breakdown: list[dict]

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- pure helpers

def applicable_dimensions(profile: HealthProfile) -> list[tuple[Dimension, float]]:
    return [(d, d.weight(profile)) for d in DIMENSIONS if d.applies(profile)]


def bmi(profile: HealthProfile) -> float | None:
    if not profile.height_cm or not profile.weight_kg:
        return None
    return round(profile.weight_kg / (profile.height_cm / 100) ** 2, 1)


def rupees(amount: int) -> str:
    """Policies quote amounts in lakh/crore, so give the LLM the same units to compare against."""
    if amount >= 1_00_00_000:
        return f"₹{amount / 1_00_00_000:g} crore"
    if amount >= 1_00_000:
        return f"₹{amount / 1_00_000:g} lakh"
    return f"₹{amount:,}"


def band(score: float | None) -> str:
    if score is None:
        return "Not enough information"
    return next(label for floor, label in BANDS if score >= floor)


def combine(results: list[DimensionResult]) -> FitScore:
    """Weighted mean of the rated dimensions, capped by deal-breakers. Silent dimensions are left
    out of the mean (silence is not evidence either way) but lower the confidence."""
    rated = [r for r in results if r.status == "rated" and r.rating is not None]
    total_w = sum(r.weight for r in results) or 1.0
    rated_w = sum(r.weight for r in rated)
    score: float | None = None
    capped = False
    if rated_w > 0:
        score = sum(r.weight * r.rating for r in rated) / rated_w
        if any(r.deal_breaker for r in rated) and score > DEAL_BREAKER_CAP:
            score, capped = DEAL_BREAKER_CAP, True
        score = round(min(10.0, max(1.0, score)), 1)
    share = rated_w / total_w
    confidence = "High" if share >= 0.85 else "Medium" if share >= 0.6 else "Low"
    return FitScore(
        score=score,
        band=band(score),
        confidence=confidence if score is not None else "Low",
        rated_weight_share=round(share, 3),
        capped=capped,
        deal_breakers=[r.label for r in rated if r.deal_breaker],
        not_specified=[r.label for r in results if r.status != "rated"],
        breakdown=[{"key": r.key, "label": r.label, "weight": r.weight, "rating": r.rating,
                    "share": round(r.weight / rated_w, 3) if r.status == "rated" and r.rating is not None else 0.0}
                   for r in results],
    )


def person_block(p: HealthProfile) -> str:
    """The profile as the LLM reads it. Only facts the person gave; blanks are left out."""
    lines = [f"Age: {p.age}"]
    if p.gender != "unspecified":
        lines.append(f"Gender: {p.gender}")
    if p.city_tier != "unspecified":
        lines.append(f"City: {p.city_tier}")
    lines.append("To be covered: " + ", ".join(dict.fromkeys(p.covered_members)))
    if p.desired_sum_insured:
        lines.append(f"Desired sum insured: {rupees(p.desired_sum_insured)}")
    if p.annual_budget:
        lines.append(f"Annual premium budget: {rupees(p.annual_budget)}")
    lines.append("Pre-existing conditions: " + (", ".join(p.conditions) if p.conditions else "none declared"))
    for label, value in (("Current medications", p.medications), ("Past surgeries", p.past_surgeries),
                         ("Hazardous activities", p.hazardous_activity)):
        if value.strip():
            lines.append(f"{label}: {value.strip()}")
    if p.hospitalised_last_2y:
        lines.append("Hospitalised in the last 2 years: yes")
    lines.append(f"Smoker/tobacco: {'yes' if p.smoker else 'no'}; alcohol: {p.alcohol}")
    if (b := bmi(p)) is not None:
        lines.append(f"BMI: {b}")
    if p.planning_pregnancy:
        lines.append("Planning a pregnancy: yes")
    if p.planned_procedures:
        lines.append("Planned procedures: " + ", ".join(p.planned_procedures))
    wants = [w for flag, w in ((p.wants_opd, "OPD"), (p.wants_dental, "dental"), (p.wants_ayush, "AYUSH")) if flag]
    if wants:
        lines.append("Wants cover for: " + ", ".join(wants))
    priorities = [w for flag, w in ((p.avoid_copay, "no co-payment"), (p.avoid_room_rent_limit, "no room rent limit")) if flag]
    if priorities:
        lines.append("Priorities: " + ", ".join(priorities))
    return "\n".join(lines)


def redacted_profile(p: HealthProfile) -> dict:
    """What a trace may record about the person: coarse bands and counts, no condition names."""
    decade = (p.age // 10) * 10
    return {
        "age_band": f"{decade}-{decade + 9}",
        "members": _members(p),
        "n_conditions": len(p.conditions),
        "n_planned_procedures": len(p.planned_procedures),
        "flags": [k for k in ("smoker", "planning_pregnancy", "wants_opd", "wants_dental", "wants_ayush",
                              "avoid_copay", "avoid_room_rent_limit") if getattr(p, k)],
    }


def parse_dimension_reply(text: str, result: DimensionResult) -> None:
    """Fill `result` from the LLM's JSON. Sources outside this dimension's excerpts are dropped."""
    data = parse_json_object(text)
    allowed = set(result.sources)
    rating = data.get("rating")
    if rating is None:
        result.status, result.rating = "not_specified", None
    else:
        result.rating = max(1, min(10, int(round(float(rating)))))
        result.status = "rated"
    result.deal_breaker = bool(data.get("deal_breaker")) and result.rating is not None
    result.verdict = str(data.get("verdict") or "")[:600]
    evidence = []
    for item in data.get("evidence") or []:
        try:
            n = int(item.get("source"))
        except (TypeError, ValueError, AttributeError):
            continue
        if n in allowed:
            evidence.append({"source": n, "text": str(item.get("text") or "")[:400]})
    result.evidence = evidence[:5]


# --------------------------------------------------------------------------- the stream

def _retrieve_for(session: Session, profile: HealthProfile, dim: Dimension, reranker: Reranker,
                  settings: Settings, trace: obs.Span) -> list[Hit]:
    """Run each of the dimension's queries and keep the best-ranked distinct parents."""
    seen: dict[str, Hit] = {}
    for query in dim.queries(profile):
        for hit in retrieve(session, query, reranker, settings, trace):
            seen.setdefault(hit.parent.id, hit)
    return list(seen.values())[:MAX_SOURCES_PER_DIMENSION]


def _excerpts(numbered: list[tuple[int, Hit]]) -> str:
    blocks, used = [], 0
    for n, hit in numbered:
        text = hit.parent.text[: max(0, MAX_DIMENSION_CONTEXT_CHARS - used)]
        if not text:
            break
        used += len(text)
        blocks.append(source_block(n, hit, text))
    return "\n\n".join(blocks)


async def _assess(session: Session, person: str, dim: Dimension, weight: float,
                  numbered: list[tuple[int, Hit]], llm: BaseChatModel, sem: asyncio.Semaphore,
                  trace: obs.Span) -> DimensionResult:
    result = DimensionResult(dim.key, dim.label, weight, sources=[n for n, _ in numbered])
    if not numbered:
        result.status = "not_specified"
        return result
    excerpts = _excerpts(numbered)
    prompt = FIT_DIMENSION_TEMPLATE.format(
        label=dim.label, instructions=dim.instructions, person=person,
        definitions=glossary_block(session, excerpts), excerpts=excerpts)
    messages = [SystemMessage(content=FIT_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    span = trace.child(f"dimension:{dim.key}", input={"sources": result.sources, "weight": weight})
    last_error: Exception | None = None
    async with sem:
        for _ in range(2):  # one retry when the reply isn't valid JSON
            try:
                reply = await llm.ainvoke(messages, config={"callbacks": span.callbacks(), "run_name": f"fit:{dim.key}"})
                parse_dimension_reply(reply.text, result)
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001 - one bad dimension must not sink the whole score
                last_error = exc
    if last_error is not None:
        result.status, result.rating = "failed", None
        log.warning("fit dimension %s failed: %s", dim.key, type(last_error).__name__)
    span.end(output={"status": result.status, "rating": result.rating, "deal_breaker": result.deal_breaker},
             level="WARNING" if last_error else "DEFAULT")
    return result


def _assessments_text(results: list[DimensionResult]) -> str:
    lines = []
    for r in results:
        rating = f"{r.rating}/10" if r.rating is not None else "not specified"
        flag = " DEAL-BREAKER" if r.deal_breaker else ""
        lines.append(f"- {r.label} [weight {r.weight:g}]: {rating}{flag}. {r.verdict}")
        lines.extend(f"    - [Source {e['source']}] {e['text']}" for e in r.evidence)
    return "\n".join(lines)


async def fit_score_stream(session: Session, profile: HealthProfile, llm: BaseChatModel, reranker: Reranker,
                           settings: Settings) -> AsyncIterator[tuple[str, object]]:
    """Yields ("stage", str), ("citations", list), ("dimension", dict)*, ("score", dict), ("token", str)*."""
    trace = obs.start_trace(
        "fit_score",
        input=redacted_profile(profile),
        metadata={"filename": session.filename, "llm": f"{settings.llm_provider}:{settings.llm_model}"},
        session_id=obs.short_hash(session.id),
        tags=["fit_score"],
    )
    fit: FitScore | None = None
    try:
        dims = applicable_dimensions(profile)
        yield "stage", f"Searching the policy for {len(dims)} areas that matter to you"
        found = await asyncio.gather(*(asyncio.to_thread(_retrieve_for, session, profile, d, reranker, settings, trace)
                                       for d, _ in dims))

        # One source numbering across all dimensions, so every [Source N] means the same excerpt.
        numbers: dict[str, int] = {}
        citations: list[dict] = []
        numbered_per_dim: list[list[tuple[int, Hit]]] = []
        for hits in found:
            numbered = []
            for hit in hits:
                if hit.parent.id not in numbers:
                    numbers[hit.parent.id] = len(numbers) + 1
                    citations.append(citation(session, numbers[hit.parent.id], hit))
                numbered.append((numbers[hit.parent.id], hit))
            numbered_per_dim.append(numbered)
        yield "citations", citations

        yield "stage", "Assessing each area against your profile"
        person = person_block(profile)
        sem = asyncio.Semaphore(LLM_CONCURRENCY)
        tasks = [asyncio.create_task(_assess(session, person, d, w, numbered, llm, sem, trace))
                 for (d, w), numbered in zip(dims, numbered_per_dim)]
        try:
            for done in asyncio.as_completed(tasks):
                yield "dimension", (await done).as_dict()
        finally:
            for t in tasks:
                t.cancel()  # client went away: don't keep paying for LLM calls
        results = [t.result() for t in tasks]  # rubric order, regardless of completion order

        fit = combine(results)
        yield "score", fit.as_dict()
        trace.update(output=fit.as_dict())
        if fit.score is not None:
            trace.score("fit_score", fit.score, data_type="NUMERIC")

        yield "stage", "Writing the reasoning"
        score_text = f"{fit.score:g}/10 ({fit.band})" if fit.score is not None else "not enough information to score"
        if fit.capped:
            score_text += f", capped at {DEAL_BREAKER_CAP:g} because of a deal-breaker"
        prompt = FIT_SUMMARY_TEMPLATE.format(
            person=person, score_text=score_text, confidence=fit.confidence,
            assessments=_assessments_text(results), unknowns=", ".join(fit.not_specified) or "nothing")
        messages = [SystemMessage(content=FIT_SYSTEM_PROMPT), HumanMessage(content=prompt)]
        async for chunk in llm.astream(messages, config={"callbacks": trace.callbacks(), "run_name": "fit:summary"}):
            if chunk.text:
                yield "token", chunk.text
    except Exception as exc:
        trace.update(level="ERROR", status_message=f"{type(exc).__name__}: {exc}"[:300])
        raise
    finally:
        trace.end(output=fit.as_dict() if fit else None)
