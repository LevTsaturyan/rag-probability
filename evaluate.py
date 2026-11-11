"""Оценка поиска и подготовка ручной разметки ответов.

python evaluate.py search [--split test]   таблица recall@5 и MRR по конфигурациям (по умолчанию dev)
python evaluate.py threshold               подбирает порог на dev, проверяет на test
python evaluate.py answers                 генерирует eval/answers.csv для ручной разметки
python evaluate.py answers --model Qwen/Qwen2.5-7B-Instruct --4bit --out eval/answers_7b.csv
python evaluate.py score [--out eval/answers_7b.csv]   считает итог по размеченному файлу
"""
import argparse
import csv
import io
import json
import time
from pathlib import Path

import numpy as np

from rag import config
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


def search_table(embed, split):
    questions = [q for q in load_questions(split) if q["answerable"]]
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


def write_answers(embed, model, load_4bit, out):
    from rag.generator import load_generator
    from rag.pipeline import answer, format_sources
    generate = load_generator(model, load_4bit)
    retriever = Retriever(make_chunks(load_pages()), embed)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["q", "answerable", "answered", "answer", "sources", "seconds", "correct", "grounded"])
        for item in load_questions("test"):
            start = time.perf_counter()
            res = answer(item["q"], retriever, generate)
            seconds = time.perf_counter() - start
            w.writerow([item["q"], int(item["answerable"]), int(res.answered), res.text,
                        format_sources(res.hits), f"{seconds:.1f}", "", ""])
            print(f"{seconds:5.1f} с  {item['q'][:70]}")
    print(f"Заполните колонки correct и grounded (1 или 0) в {out}")


def read_answers(path):
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig", errors="replace")
    header = text.splitlines()[0]
    delimiter = "\t" if "\t" in header else ";" if ";" in header else ","
    return list(csv.DictReader(io.StringIO(text), delimiter=delimiter))


def score_answers(path):
    rows = read_answers(path)
    labeled = [r for r in rows if r["answerable"] == "1"]
    empty = [r["q"] for r in labeled if not r["correct"].strip() or not r["grounded"].strip()]
    if empty:
        raise SystemExit("Не размечены вопросы:\n" + "\n".join(empty))
    print(f"вопросов: {len(labeled)}")
    print(f"верные ответы: {np.mean([int(r['correct']) for r in labeled]):.3f}")
    print(f"ответ опирается на источники: {np.mean([int(r['grounded']) for r in labeled]):.3f}")
    unanswerable = [r for r in rows if r["answerable"] == "0"]
    print(f"отказ на вопросах без ответа: {sum(r['answered'] == '0' for r in unanswerable)} из {len(unanswerable)}")
    answered = [float(r["seconds"]) for r in rows if r.get("seconds") and r["answered"] == "1"]
    if answered:
        print(f"время ответа, медиана: {np.median(answered):.1f} с")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["search", "threshold", "answers", "score"])
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--model", default=config.LLM_MODEL)
    parser.add_argument("--4bit", dest="load_4bit", action="store_true", help="загрузить LLM в 4 битах")
    parser.add_argument("--out", type=Path, default=ANSWERS, help="файл с ответами")
    args = parser.parse_args()
    if args.command == "score":
        return score_answers(args.out)
    from rag.retriever import load_embedder
    embed = load_embedder("cpu" if args.load_4bit else None)
    if args.command == "search":
        return search_table(embed, args.split)
    if args.command == "threshold":
        return tune_threshold(embed)
    if args.out.exists():
        raise SystemExit(f"{args.out} уже существует. Укажите другой --out, чтобы не потерять разметку")
    write_answers(embed, args.model, args.load_4bit, args.out)


if __name__ == "__main__":
    main()