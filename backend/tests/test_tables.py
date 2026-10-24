from app.ingestion.models import Block, TableGrid
from app.ingestion.tables import (
    absorb_trailing_footnotes,
    merge_cross_page_tables,
    normalise_table,
    split_groups,
)

HEADER = [["Benefit", "Plan Variant", "Plan Variant"], ["Benefit", "Silver", "Gold"]]


def grid(rows, page, header_rows=2, top=0.1, bottom=0.9, footnotes=None):
    return TableGrid(rows=[list(r) for r in rows], header_rows=header_rows, page_start=page, page_end=page,
                     top_ratio=top, bottom_ratio=bottom, footnotes=list(footnotes or []))


def test_merges_table_continued_on_next_page_with_repeated_header():
    a = grid(HEADER + [["Room Rent", "1% of SI*", "No limit"]], page=1)
    b = grid(HEADER + [["Maternity", "Not covered", "Rs. 50,000"]], page=2)
    blocks = merge_cross_page_tables([Block("table", table=a), Block("table", table=b)])
    assert len(blocks) == 1
    merged = blocks[0].table
    assert merged.merged_from == 2
    assert merged.page_start == 1 and merged.page_end == 2
    assert [r[0] for r in merged.body] == ["Room Rent", "Maternity"]


def test_continuation_without_header_is_merged():
    a = grid(HEADER + [["Room Rent", "1%", "No limit"]], page=3)
    b = grid([["ICU", "2%", "No limit"]], page=4, header_rows=0)
    assert len(merge_cross_page_tables([Block("table", table=a), Block("table", table=b)])) == 1


def test_does_not_merge_across_headings_or_different_shapes():
    a = grid(HEADER + [["Room Rent", "1%", "No limit"]], page=1)
    b = grid(HEADER + [["ICU", "2%", "No limit"]], page=2)
    with_heading = [Block("table", table=a), Block("heading", text="4. Waiting Periods"), Block("table", table=b)]
    assert len([x for x in merge_cross_page_tables(with_heading) if x.kind == "table"]) == 2

    c = grid([["A", "B"], ["x", "y"]], page=2, header_rows=1)
    a2 = grid(HEADER + [["Room Rent", "1%", "No limit"]], page=1)
    assert len(merge_cross_page_tables([Block("table", table=a2), Block("table", table=c)])) == 2


def test_does_not_merge_when_first_table_ends_mid_page():
    a = grid(HEADER + [["Room Rent", "1%", "No limit"]], page=1, bottom=0.4)
    b = grid(HEADER + [["ICU", "2%", "No limit"]], page=2)
    assert len(merge_cross_page_tables([Block("table", table=a), Block("table", table=b)])) == 2


def test_trailing_footnotes_are_absorbed_into_table():
    t = grid(HEADER + [["Cataract", "Rs. 40,000#", "Rs. 75,000#"]], page=1)
    blocks = absorb_trailing_footnotes([
        Block("table", table=t),
        Block("text", text="# Subject to a waiting period of 24 months."),
        Block("text", text="Regular paragraph that follows."),
    ])
    assert t.footnotes == ["# Subject to a waiting period of 24 months."]
    assert [b.kind for b in blocks] == ["table", "text"]


def test_normalise_flattens_headers_group_rows_and_links_footnotes():
    rows = HEADER + [
        ["In-patient Benefits"] * 3,
        ["Room Rent", "1% of Sum Insured*", "Single private room"],
        ["", "max Rs. 5,000", ""],
        ["Cataract", "Rs. 40,000#", "Rs. 75,000#"],
    ]
    t = normalise_table(grid(rows, 1, footnotes=["* Proportionate deduction applies.", "# 24 month waiting period."]),
                        "t1", ["3. Schedule of Benefits"])
    assert t.columns == ["Benefit", "Plan Variant › Silver", "Plan Variant › Gold"]
    assert [r[0] for r in t.rows] == ["In-patient Benefits › Room Rent", "In-patient Benefits › Room Rent", "In-patient Benefits › Cataract"]
    assert t.row_notes[0] == ["* Proportionate deduction applies."]
    assert t.row_notes[2] == ["# 24 month waiting period."]
    sentence = t.row_sentence(0)
    assert "Plan Variant › Silver: 1% of Sum Insured*" in sentence
    assert "Note: * Proportionate deduction applies." in sentence
    assert t.title == "3. Schedule of Benefits"
    md = t.to_markdown()
    assert md.splitlines()[0] == "| Benefit | Plan Variant › Silver | Plan Variant › Gold |"
    assert "Note: # 24 month waiting period." in md


def test_header_inferred_when_parser_found_none():
    t = normalise_table(grid([["Age band", "Premium"], ["18-35", "6,200"], ["36-45", "8,900"]], 1, header_rows=0), "t1", [])
    assert t.columns == ["Age band", "Premium"]
    assert len(t.rows) == 2


def test_numeric_first_row_is_not_taken_as_header():
    t = normalise_table(grid([["18-35", "6,200"], ["36-45", "8,900"]], 1, header_rows=0), "t1", [])
    assert t.columns == ["Column 1", "Column 2"]
    assert len(t.rows) == 2


def test_split_groups():
    t = normalise_table(grid([["A", "B"]] + [[str(i), "x"] for i in range(95)], 1, header_rows=1), "t1", [])
    assert split_groups(t, 40) == [(0, 40), (40, 80), (80, 95)]
    assert split_groups(t, 200) == [(0, 95)]
