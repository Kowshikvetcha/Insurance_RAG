"""End-to-end ingestion on the synthetic policy PDF using the real Docling models.
Slow (~1 min on CPU the first time); run with `pytest -m slow` or just `pytest`."""

import asyncio

import pytest

from app.providers import NoopReranker, get_llm
from app.rag import Turn, answer_stream, retrieve
from tests.conftest import offline_settings

pytestmark = pytest.mark.slow


def _table(session):
    tables = session.report["tables"]
    assert len(tables) == 1, tables
    return tables[0]


def test_cross_page_benefit_table_is_merged(sample_session):
    t = _table(sample_session)
    assert t["merged_from"] == 2
    assert t["pages"] in ("pp. 1–2", "pp. 1-2")
    assert t["rows"] >= 36
    assert t["cols"] == 3
    assert t["footnotes"] == 2


def test_cell_values_land_under_correct_plan_headers(sample_session):
    rows = [u.page_content for u in sample_session.index.units if u.metadata["type"] == "table_row"]
    room = next(r for r in rows if "Room Rent" in r)
    assert "Silver: 1% of Sum Insured*" in room
    assert "Gold: Single private room" in room
    assert "In-patient Benefits › Room Rent" in room  # row-group label propagated
    maternity = next(r for r in rows if "Maternity Benefit" in r)
    assert "Silver: Not covered" in maternity
    assert "Other Benefits › Maternity" in maternity
    assert "24 months" in maternity  # '#' footnote linked to the row
    # a row from the second page of the table
    assert any("Compassionate Visit" in r and "Gold: Economy airfare" in r for r in rows)


def test_definitions_ocr_and_figures(sample_session):
    report = sample_session.report
    assert {"Hospital", "Pre-existing Disease", "Room Rent", "Co-payment"} <= set(report["definition_terms"])
    assert report["ocr_pages"] == [4]
    claims = [p for p in sample_session.index.parents.values() if "Claims" in " ".join(p.section_path)]
    assert claims and "48 hours" in " ".join(p.text for p in claims)  # text recovered from the scanned page by OCR
    figs = [f for f in report["figures"] if f["status"] != "skipped"]
    assert len(figs) == 1 and figs[0]["page"] == 3
    fig_parent = next(p for p in sample_session.index.parents.values() if p.type == "figure")
    assert "21,800" in fig_parent.text.replace(" ", "").replace("Rs.", "") or "21,800" in fig_parent.text


GOLDEN = [
    ("What is the room rent limit on the Silver plan?", "table"),
    ("Is maternity covered?", "table"),
    ("What is the waiting period for pre-existing diseases?", "4. Waiting Periods"),
    ("What are the permanent exclusions?", "5. Permanent Exclusions"),
    ("How do I make a cashless claim?", "7. Claims Procedure"),
    ("What is the premium for a 50 year old?", "figure"),
    ("Is cosmetic surgery covered?", "5. Permanent Exclusions"),
    ("How much is covered for cataract surgery?", "table"),
]


def test_golden_retrieval_hit_rate(sample_session):
    settings = offline_settings()
    hits = 0
    for question, expected in GOLDEN:
        results = retrieve(sample_session, question, NoopReranker(), settings)
        found = any(h.parent.type == expected or expected in h.parent.section_path for h in results)
        hits += found
    # Fake embeddings are random, so this measures the BM25 + structure leg alone; real embeddings do better.
    assert hits / len(GOLDEN) >= 0.75, hits


def test_answer_stream_yields_tokens_then_citations(sample_session):
    async def run():
        events = []
        async for kind, payload in answer_stream(sample_session, "Room rent limit?", [Turn("user", "hi")],
                                                  get_llm(offline_settings()), NoopReranker(), offline_settings()):
            events.append((kind, payload))
        return events

    events = asyncio.run(run())
    kinds = [k for k, _ in events]
    assert kinds[0] == "query" and kinds[-1] == "citations" and "token" in kinds
    citations = events[-1][1]
    assert citations and all(c["page_label"] for c in citations)
