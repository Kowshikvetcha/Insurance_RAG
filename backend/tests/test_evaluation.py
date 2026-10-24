"""Live evaluation maths and heuristics, with a canned judge (no keys, no Langfuse)."""

import asyncio
import json

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from app import observability as obs
from app.evaluation import (
    is_refusal,
    citation_coverage,
    citation_validity,
    cited_sources,
    evaluate_answer,
    parse_judgement,
    retrieval_strength,
)
from app.ingestion.chunking import Parent
from app.store import Hit


def hit(pid: str, dense_rank=None, dense_sim=None, bm25_rank=None) -> Hit:
    parent = Parent(id=pid, type="text", text="x", pages=[1], section_path=["S"])
    return Hit(parent, 0.1, Document(page_content="x"), dense_rank=dense_rank, dense_sim=dense_sim, bm25_rank=bm25_rank)


def judge_replying(payload: dict | str) -> GenericFakeChatModel:
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return GenericFakeChatModel(messages=iter([AIMessage(content=content), AIMessage(content=content)]))


ANSWER = ("For the Silver plan, room rent is limited to 1% of Sum Insured per day [Source 1]. "
          "For the Gold plan, a single private room is covered [Source 1]. "
          "A 2-year waiting period applies to cataract [Source 3].")
HITS = [hit("p1", 0, 0.82, 0), hit("p2", 3, 0.61, None), hit("p3", None, None, 1)]


def test_citation_heuristics():
    assert cited_sources("a [Source 1] b [Source 2, 3] c [source 1]") == [1, 2, 3, 1]
    assert citation_validity("x [Source 1] y [Source 9]", n_sources=3, not_in_doc=False) == 0.5
    assert citation_validity("no citations here", 3, not_in_doc=False) == 0.0
    assert citation_validity("The policy document does not specify this.", 3, not_in_doc=True) == 1.0
    long_cited = "This sentence is long enough and cited [Source 1]."
    long_plain = "This sentence is long enough but has no citation at all."
    assert citation_coverage(f"{long_cited} {long_plain}") == 0.5


def test_refusal_is_only_an_opening_refusal():
    assert is_refusal("The policy document does not specify this.\n\nClosest is X [Source 1].")
    assert not is_refusal("Gold covers Rs. 5,000 [Source 1].\n- The policy document does not specify the claim limit.")
    assert not is_refusal("The policy does not cover international travel insurance.")


def test_retrieval_strength_rewards_strong_agreeing_retrievers():
    strong = retrieval_strength([hit("a", 0, 0.9, 0), hit("b", 1, 0.8, 2), hit("c", 2, 0.7, 1)])
    weak = retrieval_strength([hit("a", 0, 0.3, None), hit("b", None, None, 0), hit("c", 5, 0.2, None)])
    assert strong == round(0.6 * 0.9 + 0.4 * 1.0, 4)
    assert weak < strong
    assert retrieval_strength([]) == 0.0


def test_parse_judgement_tolerates_fences_and_prose():
    assert parse_judgement('Sure!\n```json\n{"answer_relevance": 2}\n```') == {"answer_relevance": 2}
    assert parse_judgement('blah {"a": 1} trailing') == {"a": 1}


def test_full_evaluation_with_canned_judge():
    judgement = {
        "claims": [
            {"claim": "Silver room rent 1% of SI per day", "supported": True, "sources": [1]},
            {"claim": "Gold single private room", "supported": True, "sources": [1]},
            {"claim": "2-year waiting period for cataract", "supported": False, "sources": []},
        ],
        "source_relevance": [{"source": 1, "score": 2}, {"source": 2, "score": 0}],  # source 3 left unrated
        "answer_relevance": 2,
        "reason": "Answers the room-rent question for both plans.",
    }
    r = asyncio.run(evaluate_answer("room rent?", "<source id=1>...</source>", ANSWER, HITS, judge_replying(judgement)))
    assert r.groundedness == round(2 / 3, 4)
    assert r.unsupported_claims == ["2-year waiting period for cataract"]
    assert r.context_relevance == round((2 + 0 + 0) / 6, 4)  # unrated source counts as unrelated
    assert r.irrelevant_sources == [2, 3]
    assert r.answer_relevance == 1.0
    assert r.citation_validity == 1.0 and r.citation_coverage == 1.0
    expected = 0.5 * (2 / 3) + 0.25 * (2 / 6) + 0.15 * 1.0 + 0.1 * 1.0
    assert r.confidence == round(expected, 4)
    assert r.confidence_label == "Medium"
    assert r.not_in_document is False and r.judge_error is None


def test_not_in_document_answer_is_fully_grounded():
    answer = "The policy document does not specify this."
    judgement = {"claims": [], "source_relevance": [{"source": 1, "score": 1}], "answer_relevance": 2}
    r = asyncio.run(evaluate_answer("is dental implant covered?", "...", answer, HITS[:1], judge_replying(judgement)))
    assert r.not_in_document is True
    assert r.groundedness == 1.0 and r.citation_validity == 1.0
    assert r.confidence_label in {"High", "Medium"}


def test_bad_judge_reply_is_recorded_not_raised():
    r = asyncio.run(evaluate_answer("q", "...", ANSWER, HITS, judge_replying("I cannot comply.")))
    assert r.judge_error and "ValueError" in r.judge_error
    assert r.groundedness is None and r.confidence is None
    assert r.citation_validity == 1.0  # heuristics still computed


def test_no_judge_means_heuristics_only():
    r = asyncio.run(evaluate_answer("q", "...", ANSWER, HITS, judge=None))
    assert r.groundedness is None and r.confidence is None and r.judge_error is None
    assert r.retrieval_strength > 0


def test_switches_turn_off_tracing_and_the_judge():
    from app.rag import _judge_for_this_answer
    from tests.conftest import offline_settings

    keys = dict(langfuse_public_key="pk-lf-x", langfuse_secret_key="sk-lf-x")
    assert obs.configure(offline_settings(observability_enabled=False, **keys)) is False  # master switch wins over keys
    assert obs.enabled() is False

    judge = judge_replying({})
    assert _judge_for_this_answer(offline_settings(eval_enabled=False), judge) is None
    assert _judge_for_this_answer(offline_settings(observability_enabled=False), judge) is None
    assert _judge_for_this_answer(offline_settings(eval_enabled=True, eval_sample_rate=1.0), judge) is judge
    assert _judge_for_this_answer(offline_settings(eval_enabled=True, eval_sample_rate=0.0), judge) is None


def test_observability_is_a_noop_without_keys():
    assert obs.enabled() is False
    trace = obs.start_trace("chat", input={"q": "x"})
    child = trace.child("retrieve")
    assert trace.trace_id is None and child.callbacks() == []
    child.end(output={"ok": True})
    trace.score("groundedness", 0.5)  # must not raise
    trace.end()
