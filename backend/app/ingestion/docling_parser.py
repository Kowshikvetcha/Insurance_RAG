"""Docling backend: layout analysis, TableFormer table structure, OCR for image-only pages,
figure crops. Converts a DoclingDocument into our parser-neutral ParsedDocument."""

from __future__ import annotations

import io
import re
import threading

import pymupdf  # used only for a fast text-layer triage before Docling runs

from app.config import Settings
from app.ingestion.models import Block, Figure, ParsedDocument, TableGrid
from app.ingestion.parser_base import ProgressFn

# Pages whose text layer has fewer characters than this are treated as scanned / image-only.
MIN_TEXT_CHARS = 40
PAGES_PER_BATCH = 2  # smaller = finer progress; per-batch overhead is small once models are loaded
HEADING_NUMBER = re.compile(r"^\s*((?:\d+\.)*\d+)\.?\s+\S")


def heading_level(text: str, fallback: int) -> int:
    """Insurance wordings number their sections ("4.2.1 Room rent"); the numbering is a more
    reliable hierarchy signal than the layout model's heading level."""
    m = HEADING_NUMBER.match(text)
    if m:
        return m.group(1).count(".") + 1
    return max(1, fallback)


Word = tuple[float, float, float, float, str]  # x0, y0, x1, y1 (top-left origin, points), text


def triage_pdf(data: bytes) -> tuple[int, list[int], dict[int, list[Word]]]:
    """Return (page_count, pages without a usable text layer, text-layer words per page)."""
    low: list[int] = []
    words: dict[int, list[Word]] = {}
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        for i, page in enumerate(pdf, start=1):
            page_words = [tuple(w[:5]) for w in page.get_text("words")]
            if sum(len(w[4]) for w in page_words) < MIN_TEXT_CHARS:
                low.append(i)
            else:
                words[i] = page_words
        return pdf.page_count, low, words


class DoclingParser:
    name = "docling"

    def __init__(self, settings: Settings):
        self.settings = settings
        self._converter = None
        self._lock = threading.Lock()

    def _build_converter(self):
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
        from docling.document_converter import DocumentConverter, PdfFormatOption

        opts = PdfPipelineOptions()
        opts.do_ocr = self.settings.enable_ocr  # auto mode: OCR only bitmap regions / image-only pages
        opts.do_table_structure = True
        opts.table_structure_options.mode = TableFormerMode.ACCURATE
        opts.table_structure_options.do_cell_matching = True
        opts.generate_picture_images = True
        opts.images_scale = 2.0
        return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})

    def warmup(self) -> None:
        """Build the converter and load its models once, so the first upload isn't slow."""
        from docling.datamodel.base_models import InputFormat

        with self._lock:
            if self._converter is None:
                self._converter = self._build_converter()
                self._converter.initialize_pipeline(InputFormat.PDF)

    def parse(self, data: bytes, filename: str, progress: ProgressFn | None = None) -> ParsedDocument:
        from docling.datamodel.base_models import DocumentStream

        warnings: list[str] = []
        ocr_pages: list[int] = []
        words: dict[int, list[Word]] = {}
        is_pdf = filename.lower().endswith(".pdf")
        if is_pdf:
            num_pages, ocr_pages, words = triage_pdf(data)
            if num_pages > self.settings.max_pages:
                raise ValueError(f"Document has {num_pages} pages; the limit is {self.settings.max_pages}")
            if ocr_pages:
                if self.settings.enable_ocr:
                    warnings.append(f"{len(ocr_pages)} page(s) have no text layer and were read with OCR: {_fmt_pages(ocr_pages)}")
                else:
                    warnings.append(f"{len(ocr_pages)} page(s) have no text layer and OCR is disabled: {_fmt_pages(ocr_pages)}")
        else:
            num_pages = 1

        if self._converter is None and progress:
            progress(0, num_pages, "Loading layout, table and OCR models (first upload only)")
        self.warmup()

        def convert(page_range: tuple[int, int] | None = None):
            stream = DocumentStream(name=filename, stream=io.BytesIO(data))
            with self._lock:  # Docling's converter is not safe to share across concurrent conversions
                if page_range is None:
                    return self._converter.convert(stream).document
                return self._converter.convert(stream, page_range=page_range).document

        blocks: list[Block] = []
        if is_pdf:
            # Convert in small page batches so we can report real progress. Page numbers stay
            # absolute, and tables split across a batch boundary are rejoined by
            # merge_cross_page_tables exactly as tables split across pages are.
            for start in range(1, num_pages + 1, PAGES_PER_BATCH):
                end = min(start + PAGES_PER_BATCH - 1, num_pages)
                if progress:
                    progress(start - 1, num_pages, f"Reading page {start} of {num_pages}")
                blocks.extend(_to_blocks(convert((start, end)), words))
            if progress:
                progress(num_pages, num_pages, f"Read all {num_pages} pages")
        else:
            if progress:
                progress(0, 1, "Reading document")
            doc = convert()
            num_pages = max(len(doc.pages), 1)
            blocks = _to_blocks(doc, words)
            if progress:
                progress(1, 1, "Read document")
        for i, b in enumerate(blocks):
            b.id = f"b{i}"
        return ParsedDocument(
            filename=filename,
            num_pages=num_pages,
            blocks=blocks,
            ocr_pages=ocr_pages,
            warnings=warnings,
            parser=self.name,
        )


def _fmt_pages(pages: list[int]) -> str:
    shown = ", ".join(str(p) for p in pages[:15])
    return shown + (" …" if len(pages) > 15 else "")


def _page_and_ratios(item, doc) -> tuple[int | None, int | None, float | None, float | None]:
    """First page, last page, and vertical position (0=top, 1=bottom) of the item's top on its
    first page and bottom on its last page."""
    if not item.prov:
        return None, None, None, None
    first, last = item.prov[0], item.prov[-1]

    def ratios(prov):
        page = doc.pages.get(prov.page_no)
        if page is None or not page.size.height:
            return None, None
        h = page.size.height
        bb = prov.bbox
        if bb.coord_origin.name == "BOTTOMLEFT":
            top, bottom = h - bb.t, h - bb.b
        else:
            top, bottom = bb.t, bb.b
        return top / h, bottom / h

    top_ratio, _ = ratios(first)
    _, bottom_ratio = ratios(last)
    return first.page_no, last.page_no, top_ratio, bottom_ratio


def _top_left(bb, page_height: float) -> tuple[float, float, float, float]:
    if bb.coord_origin.name == "BOTTOMLEFT":
        return bb.l, page_height - bb.t, bb.r, page_height - bb.b
    return bb.l, bb.t, bb.r, bb.b


def recover_clipped_words(item, doc, words_by_page: dict[int, list[Word]]) -> dict[tuple[int, int], str]:
    """TableFormer's cell matching keeps only the text inside each predicted cell box, so text that
    overflows a cell border ("Rs. 75,000 (caesarean)#") is silently dropped. Find text-layer words
    inside the table's rows that fall outside every cell and append them to the cell they belong to
    (the right-most cell in that row starting left of the word). Returns {(row, col): extra text}."""
    extras: dict[tuple[int, int], list[str]] = {}
    if not item.prov or len(item.prov) != 1:
        return {}
    page_no = item.prov[0].page_no
    page = doc.pages.get(page_no)
    words = words_by_page.get(page_no)
    if not words or page is None:
        return {}
    h = page.size.height
    cells = [(c, _top_left(c.bbox, h)) for c in item.data.table_cells if c.bbox is not None]
    if not cells:
        return {}
    t_left, t_top, _, t_bottom = _top_left(item.prov[0].bbox, h)
    bands: dict[int, tuple[float, float]] = {}
    for c, (_, top, _, bottom) in cells:
        if c.row_span == 1:
            lo, hi = bands.get(c.start_row_offset_idx, (top, bottom))
            bands[c.start_row_offset_idx] = (min(lo, top), max(hi, bottom))
    pad = 1.5
    for x0, y0, x1, y1, text in words:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if not (t_top - pad <= cy <= t_bottom + pad) or x0 < t_left - pad:
            continue
        if any(l - pad <= cx <= r + pad and top - pad <= cy <= bottom + pad for _, (l, top, r, bottom) in cells):
            continue  # inside a cell: already handled by the table model
        row = next((r for r, (lo, hi) in bands.items() if lo - pad <= cy <= hi + pad), None)
        if row is None:
            continue
        in_row = [(c, bb) for c, bb in cells if c.start_row_offset_idx <= row < c.end_row_offset_idx and bb[0] <= cx]
        if not in_row:
            continue
        cell = max(in_row, key=lambda cb: cb[1][0])[0]
        if text in cell.text.split():
            continue
        extras.setdefault((cell.start_row_offset_idx, cell.start_col_offset_idx), []).append(text)
    return {k: " ".join(v) for k, v in extras.items()}


def _table_grid(item, doc, words_by_page: dict[int, list[Word]]) -> TableGrid:
    grid = item.data.grid  # spans are already expanded into every covered position
    extras = recover_clipped_words(item, doc, words_by_page)
    rows = [[
        re.sub(r"\s+", " ", f"{cell.text} {extras.get((cell.start_row_offset_idx, cell.start_col_offset_idx), '')}").strip()
        for cell in row
    ] for row in grid]
    header_rows = 0
    for row in grid:
        filled = [c for c in row if c.text.strip()]
        if filled and all(c.column_header for c in filled):
            header_rows += 1
        else:
            break
    footnotes = []
    for ref in item.footnotes:
        node = ref.resolve(doc)
        if getattr(node, "text", "").strip():
            footnotes.append(node.text.strip())
    first, last, top_ratio, bottom_ratio = _page_and_ratios(item, doc)
    return TableGrid(
        rows=rows,
        header_rows=header_rows,
        caption=item.caption_text(doc).strip(),
        footnotes=footnotes,
        page_start=first,
        page_end=last,
        top_ratio=top_ratio,
        bottom_ratio=bottom_ratio,
    )


def _figure(item, doc) -> Figure:
    png = None
    width = height = 0.0
    try:
        image = item.get_image(doc)
        if image is not None:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            png = buf.getvalue()
    except Exception:  # noqa: BLE001 - a missing crop just means no vision description
        png = None
    if item.prov:
        bb = item.prov[0].bbox
        width, height = abs(bb.r - bb.l), abs(bb.t - bb.b)
    inner = []
    for child, _ in doc.iterate_items(root=item, traverse_pictures=True):
        text = getattr(child, "text", "")
        if child is not item and text and text.strip():
            inner.append(text.strip())
    return Figure(image_png=png, caption=item.caption_text(doc).strip(), inner_text=" ".join(inner), width=width, height=height)


def _to_blocks(doc, words_by_page: dict[int, list[Word]]) -> list[Block]:
    from docling_core.types.doc import DocItemLabel, ListItem, PictureItem, SectionHeaderItem, TableItem, TextItem

    blocks: list[Block] = []
    attached: set[str] = set()  # captions/footnotes already attached to a table or figure
    for item, _depth in doc.iterate_items():
        page = item.prov[0].page_no if getattr(item, "prov", None) else None
        if isinstance(item, TableItem):
            attached.update(r.cref for r in item.captions + item.footnotes)
            blocks.append(Block(kind="table", page=page, table=_table_grid(item, doc, words_by_page)))
        elif isinstance(item, PictureItem):
            attached.update(r.cref for r in item.captions + item.footnotes)
            blocks.append(Block(kind="figure", page=page, figure=_figure(item, doc)))
        elif isinstance(item, SectionHeaderItem) or getattr(item, "label", None) == DocItemLabel.TITLE:
            text = item.text.strip()
            if text:
                fallback = getattr(item, "level", 1) or 1
                blocks.append(Block(kind="heading", text=text, page=page, level=heading_level(text, fallback)))
        elif isinstance(item, TextItem):
            if item.self_ref in attached:
                continue
            text = item.text.strip()
            if not text:
                continue
            if isinstance(item, ListItem):
                marker = (item.marker or "-").strip()
                blocks.append(Block(kind="list_item", text=f"{marker} {text}", page=page))
            elif item.label == DocItemLabel.FOOTNOTE:
                blocks.append(Block(kind="footnote", text=text, page=page))
            else:
                blocks.append(Block(kind="text", text=text, page=page))
    for i, b in enumerate(blocks):
        b.id = f"b{i}"
    return blocks
