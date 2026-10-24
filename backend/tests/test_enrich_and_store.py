import time

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from app.ingestion.chunking import Parent, build_structure, to_chunked
from app.ingestion.enrich import extract_definition, find_terms, query_categories, tag_categories
from app.ingestion.models import Block, ParsedDocument, TableGrid
from app.store import HybridIndex, Session, SessionStore, tokenize


def test_extract_definition_handles_numbering_and_markers():
    assert extract_definition("2.2 Pre-existing Disease means any condition diagnosed within 36 months")[0] == "Pre-existing Disease"
    assert extract_definition("- 2.1 Hospital means any institution")[0] == "Hospital"
    assert extract_definition("(a) Accident means a sudden, unforeseen event")[0] == "Accident"
    assert extract_definition("The policy covers hospitalisation expenses.") is None


def test_find_terms_prefers_longest_match():
    glossary = {}
    for text in ["Pre-existing Disease means X", "Disease means Y", "Hospital means Z"]:
        term, d = extract_definition(text)
        from app.ingestion.enrich import Definition

        glossary[term.lower()] = Definition(term, d, 1, "Definitions")
    found = [d.term for d in find_terms("Is a pre-existing disease covered in hospital?", glossary)]
    assert found[0] == "Pre-existing Disease"
    assert "Hospital" in found
    assert "Disease" not in found


def test_category_tagging():
    assert "exclusion" in tag_categories("Cosmetic surgery is not covered.", ["5. Permanent Exclusions"])
    assert "waiting_period" in tag_categories("Pre-existing diseases are excluded for 36 months.", ["4. Waiting Periods"])
    assert tag_categories("Lorem ipsum dolor.", []) == ["general"]
    cats = query_categories("Is maternity covered?")
    assert {"coverage", "exclusion", "waiting_period"} <= cats


def test_tokenize_handles_hyphenated_insurance_terms():
    toks = tokenize("Co-payment of 20% for AYUSH")
    assert {"co-payment", "copayment", "payment", "20%", "ayush"} <= set(toks)


def _doc():
    table = TableGrid(rows=[["Benefit", "Silver", "Gold"], ["Room Rent", "1% of SI", "No limit"], ["Maternity", "Not covered", "Rs. 50,000"]],
                      header_rows=1, page_start=2, page_end=2)
    blocks = [
        Block("heading", "2. Definitions", page=1, level=1),
        Block("list_item", "- 2.1 Hospital means an institution with at least 10 beds.", page=1),
        Block("heading", "3. Schedule of Benefits", page=2, level=1),
        Block("table", page=2, table=table),
        Block("heading", "5. Permanent Exclusions", page=3, level=1),
        Block("text", "Cosmetic surgery and obesity treatment are not covered.", page=3),
        Block("heading", "7. Claims Procedure", page=4, level=1),
        Block("text", "For cashless claims intimate the TPA 48 hours before planned hospitalisation.", page=4),
    ]
    return ParsedDocument("policy.pdf", 4, blocks)


def test_structure_builds_parents_units_and_glossary():
    chunked = to_chunked(build_structure(_doc(), table_group_rows=40))
    assert "hospital" in chunked.glossary
    types = {u.metadata["type"] for u in chunked.units}
    assert {"text", "table_row"} <= types
    table_parent = next(p for p in chunked.parents.values() if p.type == "table")
    assert "| Benefit | Silver | Gold |" in table_parent.text
    assert table_parent.pages == [2]
    row_unit = next(u for u in chunked.units if u.metadata["type"] == "table_row" and "Maternity" in u.page_content)
    assert row_unit.metadata["parent_id"] == table_parent.id
    assert row_unit.page_content.startswith("[policy.pdf › 3. Schedule of Benefits]")
    excl = next(p for p in chunked.parents.values() if p.section_path == ["5. Permanent Exclusions"])
    assert "exclusion" in excl.categories


def _index():
    chunked = to_chunked(build_structure(_doc(), table_group_rows=40))
    idx = HybridIndex(chunked.units, chunked.parents, DeterministicFakeEmbedding(size=64))
    idx.embed(batch_size=2)
    return idx


def test_hybrid_search_returns_parents_with_bm25_signal():
    idx = _index()
    # Fake embeddings are random, so check the lexical leg directly and the fused result at parent level.
    assert "Maternity" in idx.units[idx._sparse("maternity", k=10)[0]].page_content
    assert idx.search("maternity Gold", k=10)[0].parent.type == "table"
    top = idx.search("cashless TPA", k=10)[0]
    assert top.parent.section_path == ["7. Claims Procedure"]


def test_session_store_isolation_and_ttl():
    store = SessionStore(ttl_minutes=1)
    a = Session("a", "a.pdf", _index(), {}, {})
    b = Session("b", "b.pdf", _index(), {}, {})
    store.add(a)
    store.add(b)
    assert store.get("a").filename == "a.pdf"
    assert store.get("missing") is None
    b.last_access = time.time() - 120
    assert store.evict_expired() == 1
    assert store.get("b") is None and store.get("a") is not None
    assert store.delete("a") and store.get("a") is None


def test_parent_docstring_types():
    p = Parent(id="p1", type="text", text="x", pages=[1], section_path=[])
    assert p.categories == ["general"]
    assert isinstance(Document(page_content="x"), Document)
