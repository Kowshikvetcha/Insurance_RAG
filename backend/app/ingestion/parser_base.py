"""Parser interface + registry. Backends turn raw file bytes into a ParsedDocument."""

from __future__ import annotations

from typing import Callable, Protocol

from app.config import Settings
from app.ingestion.models import ParsedDocument

ProgressFn = Callable[[str], None]

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


class Parser(Protocol):
    name: str

    def parse(self, data: bytes, filename: str, progress: ProgressFn | None = None) -> ParsedDocument: ...

    def warmup(self) -> None: ...


class ParserNotAvailable(RuntimeError):
    pass


_PARSERS: dict[str, Parser] = {}


def get_parser(settings: Settings) -> Parser:
    name = settings.parser
    if name not in _PARSERS:
        if name == "docling":
            from app.ingestion.docling_parser import DoclingParser

            _PARSERS[name] = DoclingParser(settings)
        elif name in {"pymupdf4llm", "llamaparse", "azure_di"}:
            raise ParserNotAvailable(f"Parser '{name}' is reserved for a later release; use PARSER=docling")
        else:
            raise ParserNotAvailable(f"Unknown PARSER: {name}")
    return _PARSERS[name]
