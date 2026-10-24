"""Table normalisation: cross-page merging, header flattening, row-group labels, footnote
linking, and the three representations used downstream (markdown grid, row sentences, summary)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.ingestion.models import Block, TableGrid

# Footnote markers commonly used in policy schedules, at the end of a cell: "₹5,000*", "Covered#", "Yes¹"
MARKER_RE = re.compile(r"(\*{1,3}|#{1,2}|†|‡|§|[¹²³⁴⁵⁶⁷⁸⁹]|\^\d{1,2}|\(\d{1,2}\))$")
FOOTNOTE_START_RE = re.compile(r"^(\*{1,3}|#{1,2}|†|‡|§|[¹²³⁴⁵⁶⁷⁸⁹]|\^?\d{1,2}[.)]?|\(\d{1,2}\))\s*\S")
NUMERIC_RE = re.compile(r"^[\s₹$€£%.,\d\-–/()x×+]*$")

# A table that ends below this point of its page may continue on the next one...
CONTINUES_BOTTOM = 0.70
# ...if the next page's table starts above this point.
CONTINUES_TOP = 0.35


# --------------------------------------------------------------------------- cross-page merging

def _same_header(a: TableGrid, b: TableGrid) -> bool:
    if not a.header_rows or not b.header_rows:
        return False
    norm = lambda rows: [[c.lower() for c in r] for r in rows]  # noqa: E731
    return norm(a.header) == norm(b.header)


def _continues(a: TableGrid, b: TableGrid) -> bool:
    if a.page_end is None or b.page_start is None or b.page_start != a.page_end + 1:
        return False
    if a.num_cols != b.num_cols:
        return False
    if b.header_rows and not _same_header(a, b):
        return False  # a new header means a new table
    near_bottom = a.bottom_ratio is None or a.bottom_ratio >= CONTINUES_BOTTOM
    near_top = b.top_ratio is None or b.top_ratio <= CONTINUES_TOP
    return near_bottom and near_top


def merge_cross_page_tables(blocks: list[Block]) -> list[Block]:
    """Merge a table with the table that continues it on the next page. Only footnotes may sit
    between the two halves; any heading or body text means they're separate tables."""
    out: list[Block] = []
    last_table_idx: int | None = None
    for block in blocks:
        if block.kind == "table" and block.table and last_table_idx is not None:
            prev = out[last_table_idx].table
            between = out[last_table_idx + 1 :]
            if all(b.kind == "footnote" for b in between) and _continues(prev, block.table):
                cur = block.table
                prev.rows.extend(cur.rows[cur.header_rows :])  # drop the repeated header
                prev.footnotes.extend(cur.footnotes)
                prev.page_end = cur.page_end
                prev.bottom_ratio = cur.bottom_ratio
                prev.merged_from += cur.merged_from
                continue
        out.append(block)
        if block.kind == "table":
            last_table_idx = len(out) - 1
        elif block.kind not in {"footnote"}:
            last_table_idx = None
    return out


def absorb_trailing_footnotes(blocks: list[Block]) -> list[Block]:
    """Footnote-looking lines directly after a table ("* subject to 24 months waiting period")
    belong to that table."""
    out: list[Block] = []
    current: TableGrid | None = None
    for block in blocks:
        if block.kind == "table":
            current = block.table
            out.append(block)
            continue
        if current is not None and block.kind in {"footnote", "text"} and FOOTNOTE_START_RE.match(block.text) and len(block.text) < 600:
            current.footnotes.append(block.text)
            continue
        current = None
        out.append(block)
    return out


# --------------------------------------------------------------------------- normalised table

@dataclass
class NormalisedTable:
    id: str
    title: str
    section_path: list[str]
    columns: list[str]
    rows: list[list[str]]
    row_notes: list[list[str]]  # footnote texts that apply to each row
    footnotes: list[str]
    page_start: int | None
    page_end: int | None
    merged_from: int = 1
    summary: str = ""
    groups: list[tuple[int, int]] = field(default_factory=list)  # [start, end) row ranges

    @property
    def pages(self) -> list[int]:
        if self.page_start is None:
            return []
        return list(range(self.page_start, (self.page_end or self.page_start) + 1))

    def page_label(self) -> str:
        if self.page_start is None:
            return ""
        if self.page_end and self.page_end != self.page_start:
            return f"pp. {self.page_start}–{self.page_end}"
        return f"p. {self.page_start}"

    def to_markdown(self, start: int = 0, end: int | None = None, with_footnotes: bool = True) -> str:
        esc = lambda s: s.replace("|", "\\|").replace("\n", " ")  # noqa: E731
        lines = ["| " + " | ".join(esc(c) for c in self.columns) + " |", "|" + "---|" * len(self.columns)]
        for row in self.rows[start:end]:
            lines.append("| " + " | ".join(esc(c) for c in row) + " |")
        if with_footnotes and self.footnotes:
            lines.append("")
            lines.extend(f"Note: {f}" for f in self.footnotes)
        return "\n".join(lines)

    def row_sentence(self, i: int) -> str:
        parts = [f"{col}: {val}" for col, val in zip(self.columns, self.rows[i]) if val]
        text = f"{self.title} — " + " | ".join(parts)
        if self.row_notes[i]:
            text += " | Note: " + " ".join(self.row_notes[i])
        return text

    def fallback_summary(self) -> str:
        labels = [r[0] for r in self.rows if r and r[0]]
        uniq = list(dict.fromkeys(labels))
        shown = ", ".join(uniq[:12]) + (" …" if len(uniq) > 12 else "")
        return f"Table '{self.title}' ({self.page_label()}) with columns {', '.join(self.columns)}. Rows cover: {shown}."


def _flatten_header(header: list[list[str]], ncols: int) -> list[str]:
    cols: list[str] = []
    for j in range(ncols):
        parts: list[str] = []
        for row in header:
            cell = row[j] if j < len(row) else ""
            if cell and (not parts or parts[-1] != cell):
                parts.append(cell)
        cols.append(" › ".join(parts) if parts else f"Column {j + 1}")
    seen: dict[str, int] = {}
    for j, c in enumerate(cols):  # make duplicate names unique so row sentences stay unambiguous
        if c in seen:
            seen[c] += 1
            cols[j] = f"{c} ({seen[c]})"
        else:
            seen[c] = 1
    return cols


def _looks_like_header(row: list[str]) -> bool:
    filled = [c for c in row if c]
    return len(filled) == len(row) and not any(NUMERIC_RE.match(c) for c in filled) and len(set(filled)) == len(filled)


def _footnote_map(footnotes: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for note in footnotes:
        m = FOOTNOTE_START_RE.match(note)
        if m:
            marker = m.group(1).rstrip(".)").lstrip("^").strip("()")
            mapping.setdefault(marker, note)
    return mapping


def normalise_table(grid: TableGrid, table_id: str, section_path: list[str]) -> NormalisedTable:
    ncols = grid.num_cols
    rows = [r + [""] * (ncols - len(r)) for r in grid.rows]
    header_rows = grid.header_rows
    if header_rows == 0 and len(rows) > 1 and _looks_like_header(rows[0]):
        header_rows = 1
    columns = _flatten_header(rows[:header_rows], ncols)
    body = rows[header_rows:]

    # Full-width rows ("In-patient Benefits" spanning every column) are group labels, not data.
    out_rows: list[list[str]] = []
    group = ""
    prev_label = ""
    for row in body:
        filled = {c for c in row if c}
        if ncols > 1 and len(filled) == 1 and all(c for c in row):
            group = next(iter(filled))
            continue
        if not any(row):
            continue
        row = list(row)
        # Forward-fill the row label when a merged label cell was only kept on its first row.
        if not row[0] and prev_label and any(row[1:]):
            row[0] = prev_label
        prev_label = row[0]
        if group and row[0] and not row[0].startswith(group):
            row[0] = f"{group} › {row[0]}"
        out_rows.append(row)

    marker_notes = _footnote_map(grid.footnotes)
    row_notes: list[list[str]] = []
    for row in out_rows:
        notes: list[str] = []
        for cell in row:
            m = MARKER_RE.search(cell)
            if m:
                marker = m.group(1).lstrip("^").strip("()")
                note = marker_notes.get(marker)
                if note and note not in notes:
                    notes.append(note)
        row_notes.append(notes)

    title = grid.caption or (section_path[-1] if section_path else "") or (f"Table on page {grid.page_start}" if grid.page_start else "Table")
    return NormalisedTable(
        id=table_id,
        title=title,
        section_path=section_path,
        columns=columns,
        rows=out_rows,
        row_notes=row_notes,
        footnotes=grid.footnotes,
        page_start=grid.page_start,
        page_end=grid.page_end,
        merged_from=grid.merged_from,
    )


def split_groups(table: NormalisedTable, max_rows: int) -> list[tuple[int, int]]:
    n = len(table.rows)
    if n <= max_rows:
        return [(0, n)]
    return [(s, min(s + max_rows, n)) for s in range(0, n, max_rows)]


def is_real_table(grid: TableGrid) -> bool:
    """Layout models sometimes box a single text column as a table; treat those as text."""
    filled_rows = [r for r in grid.rows if any(c for c in r)]
    return grid.num_cols >= 2 and len(filled_rows) >= 2


def grid_as_text(grid: TableGrid) -> str:
    return "\n".join(" ".join(c for c in r if c) for r in grid.rows if any(r))
