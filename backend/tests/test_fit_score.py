"""Policy fit score: rubric rules, score combination, reply parsing, and the offline SSE flow."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from app.deps import AppState
from app.fit_score import (
    DimensionResult,
    applicable_dimensions,
    combine,
    fit_score_stream,
    parse_dimension_reply,
    person_block,
    redacted_profile,
)
from app.jobs import JobManager
from app.main import app
from app.providers import NoopReranker, get_llm
from app.schemas import HealthProfile
from app.store import SessionStore
from tests.conftest import offline_settings


def _weights(profile: HealthProfile) -> dict[str, float]:
    return {d.key: w for d, w in applicable_dimensions(profile)}


def _r(key: str, weight: float, rating: int | None, deal_breaker: bool = False) -> DimensionResult:
    return DimensionResult(key, key, weight, status="rated" if rating is not None else "not_specified",
                           rating=rating, deal_breaker=deal_breaker)


def test_optional_dimensions_apply_only_when_relevant():
    base = _weights(HealthProfile(age=30))
    assert "maternity" not in base and "extras" not in base
    assert "maternity" in _weights(HealthProfile(age=30, planning_pregnancy=True))
    assert "extras" in _weights(HealthProfile(age=30, wants_dental=True))


def test_declared_conditions_raise_pre_existing_weight():
    assert _weights(HealthProfile(age=45, conditions=["diabetes"]))["pre_existing"] > _weights(HealthProfile(age=45))["pre_existing"]


def test_combine_weighted_mean():
    fit = combine([_r("a", 3.0, 8), _r("b", 1.0, 4)])
    assert fit.score == 7.0 and fit.band == "Good fit" and fit.confidence == "High" and not fit.capped


def test_combine_deal_breaker_caps_score():
    fit = combine([_r("a", 1.0, 9), _r("b", 1.0, 9, deal_breaker=True)])
    assert fit.score == 4.0 and fit.capped and fit.deal_breakers == ["b"]


def test_combine_leaves_silent_dimensions_out_but_lowers_confidence():
    fit = combine([_r("a", 1.0, 8), _r("b", 2.0, None)])
    assert fit.score == 8.0 and fit.confidence == "Low" and fit.not_specified == ["b"]
    assert combine([_r("a", 1.0, None)]).score is None


def test_parse_dimension_reply_clamps_and_drops_foreign_sources():
    r = DimensionResult("x", "X", 1.0, sources=[3, 4])
    parse_dimension_reply('```json\n{"rating": 14, "deal_breaker": true, "verdict": "v",'
                          ' "evidence": [{"source": 3, "text": "ok"}, {"source": 9, "text": "bad"}]}\n```', r)
    assert r.rating == 10 and r.deal_breaker and r.evidence == [{"source": 3, "text": "ok"}]
    r = DimensionResult("x", "X", 1.0, sources=[1])
    parse_dimension_reply('{"rating": null, "deal_breaker": true, "verdict": "silent"}', r)
    assert r.status == "not_specified" and not r.deal_breaker
    with pytest.raises(ValueError):
        parse_dimension_reply("no json here", DimensionResult("x", "X", 1.0))


def test_redacted_profile_has_no_condition_names():
    p = HealthProfile(age=47, conditions=["diabetes", "asthma"], medications="metformin")
    red = redacted_profile(p)
    assert red["age_band"] == "40-49" and red["n_conditions"] == 2
    assert "diabetes" not in str(red) and "metformin" not in str(red)
    assert "diabetes" in person_block(p)


def test_fit_score_unknown_session_is_404():
    settings = offline_settings()
    store = SessionStore(settings.session_ttl_minutes)
    app.state.app_state = AppState(settings, store, JobManager(store, settings), get_llm(settings), NoopReranker())
    with TestClient(app) as c:
        r = c.post("/fit-score", json={"session_id": "nope", "profile": {"age": 40}})
        assert r.status_code == 404
        assert c.post("/fit-score", json={"session_id": "nope", "profile": {"age": -1}}).status_code == 422


@pytest.mark.slow
def test_fit_score_stream_offline(sample_session):
    settings = offline_settings()
    profile = HealthProfile(age=45, conditions=["diabetes"], planned_procedures=["knee replacement"],
                            planning_pregnancy=True)

    async def run():
        return [e async for e in fit_score_stream(sample_session, profile, get_llm(settings), NoopReranker(), settings)]

    events = asyncio.run(run())
    kinds = [k for k, _ in events]
    assert kinds.index("citations") < kinds.index("dimension") < kinds.index("score") < kinds.index("token")
    dims = [p for k, p in events if k == "dimension"]
    assert {d["key"] for d in dims} == {d.key for d, _ in applicable_dimensions(profile)}
    citations = next(p for k, p in events if k == "citations")
    cited = {e["source"] for d in dims for e in d["evidence"]}
    assert cited and cited <= {c["source"] for c in citations}
    score = next(p for k, p in events if k == "score")
    assert score["score"] == 6.0  # the fake rates every area 6


def test_sum_insured_is_rupees_and_shown_in_lakh_or_crore():
    assert "₹10 lakh" in person_block(HealthProfile(age=40, desired_sum_insured=1_000_000))
    assert "₹1.5 crore" in person_block(HealthProfile(age=40, desired_sum_insured=15_000_000))
    assert "₹25,000" in person_block(HealthProfile(age=40, annual_budget=25_000))
