"""Оценка поиска и подготовка ручной разметки ответов.

python evaluate.py search           таблица recall@5 и MRR по конфигурациям (split=test)
python evaluate.py threshold        подбирает порог на split=dev, проверяет на split=test
python evaluate.py answers          генерирует eval/answers.csv для ручной разметки
python evaluate.py score            считает итог по размеченному eval/answers.csv
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rag.chunking import make_chunks
from rag.loaders import load_document
from rag.retriever import Retriever

DOCS = Path("eval/docs")
QUESTIONS = Path("eval/questions.jsonl")
ANSWERS = Path("eval/answers.csv")
CHUNK_SIZES = [300, 600, 1000]
MODES = ["bm25", "dense", "hybrid"]


def load_questions(split=None):
    rows = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    return [r for r in rows if split is None or r["split"] == split]


def load_pages():
    return [p for f in sorted(DOCS.iterdir()) for p in load_document(f)]


def retrieval_metrics(retriever, questions, k=5):
    ranks = []
    for item in questions:
        rank = next((r for r, h in enumerate(retriever.search(item["q"], k), 1)
                     if item["key"].lower() in h.chunk.text.lower()), None)
        ranks.append(rank)
    return sum(r is not None for r in ranks) / len(ranks), float(np.mean([1 / r if r else 0 for r in ranks]))


def search_table(embed):
    questions = [q for q in load_questions("test") if q["answerable"]]
    pages = load_pages()
    print("| Режим | Размер фрагмента | recall@5 | MRR |\n|---|---|---|---|")
    for size in CHUNK_SIZES:
        retriever = Retriever(make_chunks(pages, max_chars=size), embed)
        for mode in MODES:
            retriever.mode = mode
            recall, mrr = retrieval_metrics(retriever, questions)
            print(f"| {mode} | {size} | {recall:.3f} | {mrr:.3f} |")


def rates(retriever, questions, threshold):
    predicted = np.array([retriever.best_similarity(q["q"]) >= threshold for q in questions])
    actual = np.array([q["answerable"] for q in questions])
    return float((predicted == actual).mean())


def tune_threshold(embed):
    retriever = Retriever(make_chunks(load_pages()), embed)
    dev, test = load_questions("dev"), load_questions("test")
    grid = np.arange(0.20, 0.81, 0.02)
    best = max(grid, key=lambda t: rates(retriever, dev, t))
    print(f"порог: {best:.2f}, точность на dev: {rates(retriever, dev, best):.3f}, на test: {rates(retriever, test, best):.3f}")


def write_answers(embed):
    from rag.generator import load_generator
    from rag.pipeline import answer, format_sources
    generate = load_generator()
    retriever = Retriever(make_chunks(load_pages()), embed)
    with ANSWERS.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["q", "answerable", "answered", "answer", "sources", "correct", "grounded"])
        for item in load_questions("test"):
            res = answer(item["q"], retriever, generate)
            w.writerow([item["q"], int(item["answerable"]), int(res.answered), res.text, format_sources(res.hits), "", ""])
    print(f"Заполните колонки correct и grounded (1 или 0) в {ANSWERS}")


def score_answers():
    with ANSWERS.open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f, delimiter=";") if r["answerable"] == "1"]
    print(f"вопросов: {len(rows)}")
    print(f"верные ответы: {np.mean([int(r['correct']) for r in rows]):.3f}")
    print(f"ответ опирается на источники: {np.mean([int(r['grounded']) for r in rows]):.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["search", "threshold", "answers", "score"])
    command = parser.parse_args().command
    if command == "score":
        return score_answers()
    from rag.retriever import load_embedder
    embed = load_embedder()
    {"search": search_table, "threshold": tune_threshold, "answers": write_answers}[command](embed)


if __name__ == "__main__":
    main()