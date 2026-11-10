import zlib

import numpy as np
import pytest

from rag import config
from rag.chunking import Chunk, make_chunks
from rag.generator import build_messages
from rag.loaders import Page, load_document
from rag.pipeline import answer, format_sources
from rag.retriever import Retriever, tokenize


def fake_embed(texts):
    """Мешок слов на 64 измерения: одинаковые слова дают близкие векторы."""
    out = np.zeros((len(texts), 64))
    for i, text in enumerate(texts):
        for word in tokenize(text):
            out[i, zlib.crc32(word.encode()) % 64] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


@pytest.fixture
def chunks():
    texts = ["Дисперсия измеряет разброс значений", "Медиана делит выборку пополам", "Корреляция показывает линейную связь"]
    return [Chunk(t, "a.pdf", n) for n, t in enumerate(texts, 1)]


def test_chunks_keep_page_and_limit():
    pages = [Page("Первое предложение. Второе предложение. Третье предложение.", "a.pdf", 7)]
    chunks = make_chunks(pages, max_chars=30, overlap_sents=0)
    assert len(chunks) == 3
    assert all(c.page == 7 for c in chunks)


def test_chunks_overlap():
    pages = [Page("Раз раз раз. Два два два. Три три три.", "a.txt", None)]
    chunks = make_chunks(pages, max_chars=20, overlap_sents=1)
    assert chunks[1].text.startswith("Раз раз раз.")


def test_load_txt(tmp_path):
    path = tmp_path / "x.txt"
    path.write_text("Текст", encoding="utf-8")
    assert load_document(path)[0] == Page("Текст", "x.txt", None)


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_search_finds_relevant(chunks, mode):
    hits = Retriever(chunks, fake_embed, mode=mode).search("дисперсия и разброс", k=1)
    assert hits[0].chunk.page == 1


def test_stemming_matches_word_forms():
    assert tokenize("дисперсии") == tokenize("дисперсия")


def test_no_answer_below_threshold(chunks):
    def fail(question, hits):
        raise AssertionError("LLM не должна вызываться")

    res = answer("футбол чемпионат", Retriever(chunks, fake_embed), fail, threshold=0.5)
    assert res.text == config.NO_ANSWER and not res.answered


def test_answer_calls_generator(chunks):
    res = answer("дисперсия разброс", Retriever(chunks, fake_embed), lambda q, h: "ответ [1]", threshold=0.1)
    assert res.text == "ответ [1]" and res.answered


def test_sources_contain_page(chunks):
    hits = Retriever(chunks, fake_embed).search("дисперсия", k=1)
    assert "a.pdf, стр. 1" in format_sources(hits)


def test_messages_number_fragments(chunks):
    hits = Retriever(chunks, fake_embed).search("дисперсия", k=2)
    assert "[2]" in build_messages("вопрос", hits)[1]["content"]