from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Page:
    text: str
    source: str
    page: int | None  # номер страницы PDF, для txt — None


def load_document(path):
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(path)
        return [Page(p.extract_text() or "", path.name, n) for n, p in enumerate(reader.pages, 1)]
    return [Page(path.read_text(encoding="utf-8-sig"), path.name, None)]