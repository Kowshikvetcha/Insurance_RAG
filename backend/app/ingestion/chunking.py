"""Structure-aware chunking into a two-level index:

* parents  – what the LLM reads: a section window of prose, a full table (or row group of a big one
             with its header repeated), or a figure description.
* units    – what gets embedded/searched: small prose chunks, one sentence per table row, table
             summaries, figure descriptions. Each unit points at its parent.

Every unit carries a contextual header (document › section path › table title) so a fragment like
"₹5,000 per day" still says which plan and section it came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.enrich import DEFINITIONS_HEADING_RE, Definition, extract_definition, tag_categories
from app.ingestion.models import Block, Figure, ParsedDocument
from app.ingestion.tables import NormalisedTable, grid_as_text, is_real_table, normalise_table, split_groups

CHILD_CHARS = 900
PARENT_CHARS = 3200
_long_splitter = RecursiveCharacterTextSplitter(chunk_size=CHILD_CHARS, chunk_overlap=120, separators=["\n\n", "\n", ". ", "; ", ", ", " "])


@dataclass
class Parent:
    id: str
    type: str  # text | table | figure
    text: str
    pages: list[int]
    section_path: list[str]
    title: str = ""
    categories: list[str] = field(default_factory=lambda: ["general"])
    ref: str = ""  # table id / figure id


@dataclass
class FigureRecord:
    id: str
    figure: Figure
    page: int | None
    section_path: list[str]


@dataclass
class ChunkedDocument:
    parents: dict[str, Parent]
    units: list[Document]
    tables: list[NormalisedTable]
    table_parent_ids: dict[str, list[str]]
    figures: list[FigureRecord]
    glossary: dict[str, Definition]


def _header(filename: str, section_path: list[str], extra: str = "") -> str:
    path = " › ".join([filename, *section_path])
    return f"[{path}]" + (f" ({extra})" if extra else "")


class _Builder:
    def __init__(self, filename: str, table_group_rows: int):
        self.filename = filename
        self.table_group_rows = table_group_rows
        self.parents: dict[str, Parent] = {}
        self.units: list[Document] = []
        self.tables: list[NormalisedTable] = []
        self.table_parent_ids: dict[str, list[str]] = {}
        self.figures: list[FigureRecord] = []
        self.glossary: dict[str, Definition] = {}
        self.stack: list[tuple[int, str]] = []
        self.buffer: list[tuple[str, int | None]] = []
        self.counter = 0

    @property
    def section_path(self) -> list[str]:
        return [t for _, t in self.stack]

    def _next(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}{self.counter}"

    def _unit(self, text: str, parent: Parent, kind: str, pages: list[int], header_extra: str = "") -> None:
        content = _header(self.filename, parent.section_path, header_extra) + "\n" + text
        self.units.append(
            Document(
                page_content=content,
                metadata={
                    "unit_id": self._next("u"),
                    "parent_id": parent.id,
                    "type": kind,
                    "pages": pages,
                    "section_path": parent.section_path,
                    "categories": parent.categories,
                },
            )
        )

    # ---------------------------------------------------------------- prose
    def add_heading(self, block: Block) -> None:
        self.flush()
        while self.stack and self.stack[-1][0] >= block.level:
            self.stack.pop()
        self.stack.append((block.level, block.text))

    def add_text(self, text: str, page: int | None) -> None:
        self.buffer.append((text, page))
        in_definitions = any(DEFINITIONS_HEADING_RE.search(h) for h in self.section_path)
        if in_definitions or text.lstrip()[:1] in {'"', "“"}:
            found = extract_definition(text)
            if found:
                term, definition = found
                self.glossary.setdefault(term.lower(), Definition(term, definition, page, " › ".join(self.section_path)))

    def flush(self) -> None:
        if not self.buffer:
            return
        # 1) pack paragraphs into child-sized pieces (keeping their pages)
        pieces: list[tuple[str, list[int]]] = []
        cur, cur_pages = "", []
        for text, page in self.buffer:
            parts = _long_splitter.split_text(text) if len(text) > CHILD_CHARS * 1.3 else [text]
            for part in parts:
                if cur and len(cur) + len(part) + 1 > CHILD_CHARS:
                    pieces.append((cur, cur_pages))
                    cur, cur_pages = "", []
                cur = f"{cur}\n{part}" if cur else part
                if page is not None and page not in cur_pages:
                    cur_pages.append(page)
        if cur:
            pieces.append((cur, cur_pages))
        # 2) group consecutive pieces into parents
        groups: list[list[tuple[str, list[int]]]] = [[]]
        size = 0
        for piece in pieces:
            if groups[-1] and size + len(piece[0]) > PARENT_CHARS:
                groups.append([])
                size = 0
            groups[-1].append(piece)
            size += len(piece[0])
        path = self.section_path
        for group in groups:
            text = "\n".join(p for p, _ in group)
            pages = sorted({pg for _, pgs in group for pg in pgs})
            parent = Parent(id=self._next("p"), type="text", text=text, pages=pages, section_path=path,
                            title=path[-1] if path else "", categories=tag_categories(text, path))
            self.parents[parent.id] = parent
            for piece_text, piece_pages in group:
                self._unit(piece_text, parent, "text", piece_pages)
        self.buffer = []

    # ---------------------------------------------------------------- tables
    def add_table(self, block: Block) -> None:
        grid = block.table
        if not is_real_table(grid):
            self.add_text(grid_as_text(grid), block.page)
            return
        self.flush()
        table = normalise_table(grid, self._next("t"), self.section_path)
        if not table.rows:
            return
        table.groups = split_groups(table, self.table_group_rows)
        self.tables.append(table)
        ids: list[str] = []
        for start, end in table.groups:
            label = table.page_label()
            span = f", rows {start + 1}–{end} of {len(table.rows)}" if len(table.groups) > 1 else ""
            body = f"Table: {table.title} ({label}{span})\n{table.to_markdown(start, end)}"
            parent = Parent(id=self._next("p"), type="table", text=body, pages=table.pages, section_path=table.section_path,
                            title=table.title, categories=tag_categories(body, table.section_path), ref=table.id)
            self.parents[parent.id] = parent
            ids.append(parent.id)
            for i in range(start, end):
                self._unit(table.row_sentence(i), parent, "table_row", table.pages, f"Table: {table.title}")
        self.table_parent_ids[table.id] = ids

    def add_table_summaries(self) -> None:
        for table in self.tables:
            for pid in self.table_parent_ids.get(table.id, []):
                self._unit(table.summary, self.parents[pid], "table_summary", table.pages, f"Table: {table.title}")

    # ---------------------------------------------------------------- figures
    def add_figure(self, block: Block) -> None:
        fig = block.figure
        self.figures.append(FigureRecord(id=self._next("f"), figure=fig, page=block.page, section_path=self.section_path))

    def finish_figures(self) -> None:
        """Called after the vision step has filled Figure.description."""
        for rec in self.figures:
            desc = rec.figure.description
            if rec.figure.skipped_reason or not desc:
                continue
            title = rec.figure.caption or (f"Figure on page {rec.page}" if rec.page else "Figure")
            pages = [rec.page] if rec.page else []
            body = desc if desc.startswith(title) else f"Figure: {title}\n{desc}"
            parent = Parent(id=self._next("p"), type="figure", text=body, pages=pages, section_path=rec.section_path,
                            title=title, categories=tag_categories(body, rec.section_path), ref=rec.id)
            self.parents[parent.id] = parent
            for piece in _long_splitter.split_text(desc) if len(desc) > CHILD_CHARS * 1.3 else [desc]:
                self._unit(piece, parent, "figure", pages, f"Figure: {title}")


def build_structure(doc: ParsedDocument, table_group_rows: int) -> _Builder:
    """Walk blocks in reading order. Figures are only collected here; their units are created by
    `finish_figures` once descriptions exist, and table summary units by `add_table_summaries`."""
    b = _Builder(doc.filename, table_group_rows)
    for block in doc.blocks:
        if block.kind == "heading":
            b.add_heading(block)
        elif block.kind == "table":
            b.add_table(block)
        elif block.kind == "figure":
            b.add_figure(block)
        else:
            b.add_text(block.text, block.page)
    b.flush()
    return b


def to_chunked(b: _Builder) -> ChunkedDocument:
    return ChunkedDocument(b.parents, b.units, b.tables, b.table_parent_ids, b.figures, b.glossary)
