from dataclasses import dataclass

from rag import config


@dataclass
class Answer:
    text: str
    hits: list
    answered: bool


def answer(question, retriever, generate, threshold=config.THRESHOLD, k=config.TOP_K):
    hits = retriever.search(question, k)
    if retriever.best_similarity(question) < threshold:
        return Answer(config.NO_ANSWER, hits, False)
    return Answer(generate(question, hits), hits, True)


def format_sources(hits):
    lines = []
    for n, h in enumerate(hits, 1):
        where = h.chunk.source + (f", стр. {h.chunk.page}" if h.chunk.page else "")
        lines.append(f"[{n}] {where} (сходство {h.similarity:.2f})\n{h.chunk.text}")
    return "\n\n".join(lines)