import re
from dataclasses import dataclass

from rag import config


@dataclass
class Chunk:
    text: str
    source: str
    page: int | None


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def make_chunks(pages, max_chars=config.MAX_CHARS, overlap_sents=config.OVERLAP_SENTS):
    chunks = []
    for page in pages:
        cur = []
        for sent in split_sentences(page.text):
            if cur and sum(len(x) for x in cur) + len(sent) > max_chars:
                chunks.append(Chunk(" ".join(cur), page.source, page.page))
                cur = cur[-overlap_sents:] if overlap_sents else []
            cur.append(sent)
        if cur:
            chunks.append(Chunk(" ".join(cur), page.source, page.page))
    return chunks