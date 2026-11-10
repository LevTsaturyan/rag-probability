import re
from dataclasses import dataclass

import numpy as np
import snowballstemmer
from rank_bm25 import BM25Okapi

from rag import config
from rag.chunking import Chunk

_stemmer = snowballstemmer.stemmer("russian")


def tokenize(text):
    return _stemmer.stemWords(re.findall(r"\w+", text.lower()))


def load_embedder():
    """Возвращает функцию: список строк -> нормированные векторы."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config.EMBED_MODEL)
    model.max_seq_length = config.EMBED_MAX_LEN

    def embed(texts):
        return model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)

    return embed


@dataclass
class Hit:
    chunk: Chunk
    similarity: float  # косинусное сходство с вопросом


class Retriever:
    def __init__(self, chunks, embed, mode="hybrid", rrf_k=60):
        self.chunks, self.embed, self.mode, self.rrf_k = chunks, embed, mode, rrf_k
        self.vectors = embed([c.text for c in chunks])
        self.bm25 = BM25Okapi([tokenize(c.text) or [""] for c in chunks])

    def search(self, query, k=config.TOP_K):
        sims = self.vectors @ self.embed([query])[0]
        dense = np.argsort(-sims)
        sparse = np.argsort(-self.bm25.get_scores(tokenize(query)))
        if self.mode == "dense":
            order = dense
        elif self.mode == "bm25":
            order = sparse
        else:  # Reciprocal Rank Fusion: складываем 1 / (rrf_k + место) из двух рейтингов
            rrf = np.zeros(len(self.chunks))
            for ranking in (dense, sparse):
                rrf[ranking] += 1 / (self.rrf_k + np.arange(1, len(self.chunks) + 1))
            order = np.argsort(-rrf)
        return [Hit(self.chunks[i], float(sims[i])) for i in order[:k]]

    def best_similarity(self, query):
        return float((self.vectors @ self.embed([query])[0]).max())