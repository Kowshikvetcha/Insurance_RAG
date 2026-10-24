"""Parser-neutral document model. Every parser backend produces a ParsedDocument; everything
downstream (table normalisation, enrichment, chunking) works only on these types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

BlockKind = Literal["heading", "text", "list_item", "footnote", "table", "figure"]


@dataclass
class TableGrid:
    rows: list[list[str]]  # full grid, merged/spanning cells already copied into every position
    header_rows: int = 0  # number of leading rows that are column headers
    caption: str = ""
    footnotes: list[str] = field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    # where the table sits vertically on its last page, 0 (top) .. 1 (bottom); used for cross-page merging
    bottom_ratio: float | None = None
    top_ratio: float | None = None
    merged_from: int = 1  # how many physical tables were merged into this one

    @property
    def num_cols(self) -> int:
        return max((len(r) for r in self.rows), default=0)

    @property
    def header(self) -> list[list[str]]:
        return self.rows[: self.header_rows]

    @property
    def body(self) -> list[list[str]]:
        return self.rows[self.header_rows :]


@dataclass
class Figure:
    image_png: bytes | None
    caption: str = ""
    inner_text: str = ""  # text the parser/OCR found inside the figure
    width: float = 0.0
    height: float = 0.0
    description: str = ""  # filled by the vision LLM, or caption + inner text as a fallback
    described_by_vision: bool = False
    skipped_reason: str = ""  # e.g. "decorative" when filtered out


@dataclass
class Block:
    kind: BlockKind
    text: str = ""
    page: int | None = None
    level: int = 1  # heading level
    table: TableGrid | None = None
    figure: Figure | None = None
    id: str = ""


@dataclass
class ParsedDocument:
    filename: str
    num_pages: int
    blocks: list[Block]
    ocr_pages: list[int] = field(default_factory=list)  # pages with no usable text layer (OCR applied)
    warnings: list[str] = field(default_factory=list)
    parser: str = ""
