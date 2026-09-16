#!/usr/bin/env python3
"""
Comparación local de Recall@k: HyDE+BM25 vs TF-IDF.

Independiente de scripts/ragas.py. Mismo corpus y mismas queries
(dataset_qg_qa_sintetico.json) para comparar en igualdad de condiciones.

Nota: el TF-IDF de producción (pickle Neo4j) no aplica aquí porque el gold
del dataset sintético no coincide con ese índice. Se entrena un TF-IDF
sklearn sobre el mismo corpus sintético usado por HyDE+BM25.

Uso:
  .venv-ragas/bin/python scripts/compare_recall.py --limit 30 --top-k 3
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import ollama
import pandas as pd
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "dataset_qg_qa_sintetico.json"
DEFAULT_OUT_DIR = ROOT / "ragas_results" / "recall_compare"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:1.5b"
DEFAULT_LIMIT = 30


def load_corpus(dataset: list[dict]) -> dict[str, dict]:
    corpus: dict[str, dict] = {}
    title_to_id: dict[str, str] = {}
    counter = 1
    for item in dataset:
        for doc in item.get("documentos_fuente", []):
            title = (doc.get("titulo") or "").strip()
            if not title:
                continue
            if title not in title_to_id:
                gid = f"doc_global_{counter}"
                title_to_id[title] = gid
                corpus[gid] = {"title": title, "text": doc.get("abstract", "")}
                counter += 1
            doc["doc_id"] = title_to_id[title]
    return corpus


class HydeBM25Retriever:
    def __init__(self, corpus: dict[str, dict], llm_model: str, host: str):
        self.corpus = corpus
        self.corpus_ids = list(corpus.keys())
        self.llm_model = llm_model
        self.client = ollama.Client(host=host)
        texts = [
            f"{doc.get('title', '')} {doc.get('text', '')}".lower()
            for doc in corpus.values()
        ]
        self.bm25 = BM25Okapi([t.split() for t in texts])

    def _hyde(self, query: str) -> str:
        prompt = (
            f"Act as a scientific researcher. Write a 3-sentence abstract of a fictional "
            f"academic paper that answers the following query: '{query}'. "
            f"Use rigorous academic tone, go straight to the text with no introductions, "
            f"and ALWAYS write in English."
        )
        try:
            res = self.client.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.7, "num_predict": 150},
            )
            return res["message"]["content"].strip()
        except Exception:  # noqa: BLE001
            return ""

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        hypo = self._hyde(query)
        hyde_query = f"{query}\n{hypo}" if hypo else query
        scores = self.bm25.get_scores(hyde_query.lower().split())
        top_indices = np.argsort(scores)[::-1][:top_k]
        docs = []
        for idx in top_indices:
            doc_id = self.corpus_ids[int(idx)]
            docs.append(
                {
                    "doc_id": doc_id,
                    "score": float(scores[int(idx)]),
                    "title": self.corpus[doc_id].get("title", ""),
                }
            )
        return docs


class TfidfRetriever:
    """Baseline TF-IDF + cosine similarity sobre el mismo corpus sintético."""

    def __init__(self, corpus: dict[str, dict]):
        self.corpus = corpus
        self.corpus_ids = list(corpus.keys())
        texts = [
            f"{doc.get('title', '')} {doc.get('text', '')}" for doc in corpus.values()
        ]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            norm="l2",
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]
        docs = []
        for idx in top_indices:
            doc_id = self.corpus_ids[int(idx)]
            docs.append(
                {
                    "doc_id": doc_id,
                    "score": float(sims[int(idx)]),
                    "title": self.corpus[doc_id].get("title", ""),
                }
            )
        return docs


@dataclass
class CompareRow:
    query_id: str
    query_original: str
    expected_doc_ids: list[str]
    n_expected: int
    # HyDE+BM25
    hyde_retrieved: list[str]
    hyde_hits: int
    hyde_recall: float
    hyde_latency_ms: float
    # TF-IDF
    tfidf_retrieved: list[str]
    tfidf_hits: int
    tfidf_recall: float
    tfidf_latency_ms: float
    delta_recall: float  # hyde - tfidf
    winner: str


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def score_recall(expected: list[str], retrieved: list[str]) -> tuple[int, float]:
    hits = sum(1 for e in expected if e in retrieved)
    recall = hits / len(expected) if expected else 0.0
    return hits, recall


def save_outputs(out_dir: Path, rows: list[CompareRow], meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = meta["run_id"]
    payload = {"meta": meta, "results": [asdict(r) for r in rows]}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    (out_dir / f"compare_results_{stamp}.json").write_text(text, encoding="utf-8")
    (out_dir / "latest.json").write_text(text, encoding="utf-8")

    flat = [
        {
            "query_id": r.query_id,
            "query_original": r.query_original,
            "n_expected": r.n_expected,
            "hyde_recall": r.hyde_recall,
            "tfidf_recall": r.tfidf_recall,
            "delta_recall": r.delta_recall,
            "winner": r.winner,
            "hyde_hits": r.hyde_hits,
            "tfidf_hits": r.tfidf_hits,
            "hyde_latency_ms": r.hyde_latency_ms,
            "tfidf_latency_ms": r.tfidf_latency_ms,
            "hyde_retrieved": "|".join(r.hyde_retrieved),
            "tfidf_retrieved": "|".join(r.tfidf_retrieved),
            "expected_doc_ids": "|".join(r.expected_doc_ids),
        }
        for r in rows
    ]
    df = pd.DataFrame(flat)
    df.to_csv(out_dir / f"compare_results_{stamp}.csv", index=False, encoding="utf-8")
    df.to_csv(out_dir / "latest.csv", index=False, encoding="utf-8")

    hyde_r = [r.hyde_recall for r in rows]
    tfidf_r = [r.tfidf_recall for r in rows]
    hyde_hit = sum(1 for r in rows if r.hyde_hits > 0) / len(rows) if rows else 0
    tfidf_hit = sum(1 for r in rows if r.tfidf_hits > 0) / len(rows) if rows else 0
    hyde_perfect = sum(1 for r in rows if r.hyde_recall >= 1.0)
    tfidf_perfect = sum(1 for r in rows if r.tfidf_recall >= 1.0)
    wins = {
        "hyde_bm25": sum(1 for r in rows if r.winner == "hyde_bm25"),
        "tfidf": sum(1 for r in rows if r.winner == "tfidf"),
        "tie": sum(1 for r in rows if r.winner == "tie"),
    }

    detail_lines = [
        f"| {r.query_id} | {r.query_original[:48]} | {r.hyde_recall:.2f} | {r.tfidf_recall:.2f} | {r.delta_recall:+.2f} | {r.winner} |"
        for r in rows
    ]
    detail_table = "\n".join(detail_lines)

    summary = f"""# Comparación Recall@k: HyDE+BM25 vs TF-IDF

- Run ID: `{stamp}`
- Script: `scripts/compare_recall.py`
- Dataset: `{meta.get("dataset")}`
- Queries: **{len(rows)}** | Top-K: `{meta.get("top_k")}`
- HyDE model: `{meta.get("ollama_model")}`
- TF-IDF: sklearn `TfidfVectorizer` (1–2 grams, stop_words=english, sublinear_tf) sobre el **mismo** corpus sintético

## Promedios

| Métrica | HyDE+BM25 | TF-IDF | Δ (HyDE − TF-IDF) |
|---|---|---|---|
| Recall@{meta.get("top_k")} | {mean(hyde_r):.4f} | {mean(tfidf_r):.4f} | {mean(hyde_r) - mean(tfidf_r):+.4f} |
| Hit rate (≥1 gold) | {hyde_hit:.4f} | {tfidf_hit:.4f} | {hyde_hit - tfidf_hit:+.4f} |
| Queries Recall=1.0 | {hyde_perfect}/{len(rows)} | {tfidf_perfect}/{len(rows)} | — |
| Latencia media (ms) | {mean([r.hyde_latency_ms for r in rows]):.1f} | {mean([r.tfidf_latency_ms for r in rows]):.1f} | — |

## Ganadores por query

| Método | # queries |
|---|---|
| HyDE+BM25 mejor | {wins["hyde_bm25"]} |
| TF-IDF mejor | {wins["tfidf"]} |
| Empate | {wins["tie"]} |

## Detalle

| ID | Query | HyDE | TF-IDF | Δ | Winner |
|---|---|---|---|---|---|
{detail_table}
"""
    (out_dir / f"compare_summary_{stamp}.md").write_text(summary, encoding="utf-8")
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    if meta.get("print_summary"):
        print(summary)


def main() -> int:
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Compara Recall@k: HyDE+BM25 vs TF-IDF (local)"
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    ollama_host = os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
    if "ollama:" in ollama_host and not Path("/.dockerenv").exists():
        ollama_host = DEFAULT_OLLAMA_HOST
    ollama_model = os.getenv(
        "OLLAMA_HYDE_MODEL",
        os.getenv("OLLAMA_QA_MODEL", DEFAULT_OLLAMA_MODEL),
    )

    if not args.dataset.exists():
        print(f"ERROR: dataset no encontrado: {args.dataset}")
        return 1

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    corpus = load_corpus(dataset)
    items = dataset[args.offset :]
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    hyde = HydeBM25Retriever(corpus, llm_model=ollama_model, host=ollama_host)
    tfidf = TfidfRetriever(corpus)

    print(
        f"[compare] queries={len(items)} corpus={len(corpus)} "
        f"hyde={ollama_model}@{ollama_host} tfidf=sklearn top_k={args.top_k}"
    )

    rows: list[CompareRow] = []
    for local_idx, item in enumerate(tqdm(items, desc="Compare")):
        query = (item.get("query_original") or "").strip()
        if not query:
            continue
        query_id = f"q{args.offset + local_idx}"
        expected = [str(d.get("doc_id")) for d in item.get("documentos_fuente", [])]

        t0 = time.time()
        hyde_docs = hyde.retrieve(query, top_k=args.top_k)
        hyde_ms = (time.time() - t0) * 1000
        hyde_ids = [str(d["doc_id"]) for d in hyde_docs]
        hyde_hits, hyde_recall = score_recall(expected, hyde_ids)

        t1 = time.time()
        tfidf_docs = tfidf.retrieve(query, top_k=args.top_k)
        tfidf_ms = (time.time() - t1) * 1000
        tfidf_ids = [str(d["doc_id"]) for d in tfidf_docs]
        tfidf_hits, tfidf_recall = score_recall(expected, tfidf_ids)

        delta = hyde_recall - tfidf_recall
        if abs(delta) < 1e-9:
            winner = "tie"
        elif delta > 0:
            winner = "hyde_bm25"
        else:
            winner = "tfidf"

        rows.append(
            CompareRow(
                query_id=query_id,
                query_original=query,
                expected_doc_ids=expected,
                n_expected=len(expected),
                hyde_retrieved=hyde_ids,
                hyde_hits=hyde_hits,
                hyde_recall=round(hyde_recall, 4),
                hyde_latency_ms=round(hyde_ms, 2),
                tfidf_retrieved=tfidf_ids,
                tfidf_hits=tfidf_hits,
                tfidf_recall=round(tfidf_recall, 4),
                tfidf_latency_ms=round(tfidf_ms, 2),
                delta_recall=round(delta, 4),
                winner=winner,
            )
        )

        meta = {
            "run_id": run_id,
            "design": "compare HyDE+BM25 vs TF-IDF on same synthetic corpus",
            "dataset": str(args.dataset),
            "top_k": args.top_k,
            "ollama_model": ollama_model,
            "ollama_host": ollama_host,
            "n_queries": len(rows),
            "print_summary": False,
        }
        save_outputs(args.out_dir, rows, meta)

    meta = {
        "run_id": run_id,
        "design": "compare HyDE+BM25 vs TF-IDF on same synthetic corpus",
        "dataset": str(args.dataset),
        "top_k": args.top_k,
        "ollama_model": ollama_model,
        "ollama_host": ollama_host,
        "n_queries": len(rows),
        "print_summary": True,
    }
    save_outputs(args.out_dir, rows, meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
