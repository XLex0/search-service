#!/usr/bin/env python3
"""
Evaluación local del Retriever (Recall@k).

Independiente de scripts/ragas.py: no usa Gemini ni el generador Q&A.
Solo HyDE (Ollama) + BM25 sobre el corpus del dataset sintético.

Uso:
  .venv-ragas/bin/python scripts/evaluate_recall.py          # 30 queries
  .venv-ragas/bin/python scripts/evaluate_recall.py --limit 50 --top-k 3
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
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "dataset_qg_qa_sintetico.json"
DEFAULT_OUT_DIR = ROOT / "ragas_results" / "recall"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:1.5b"
DEFAULT_LIMIT = 30


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

    def retrieve(self, query: str, top_k: int = 3) -> tuple[list[dict], str]:
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
                    "text": self.corpus[doc_id].get("text", ""),
                }
            )
        return docs, hyde_query


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


@dataclass
class RecallRow:
    query_id: str
    query_original: str
    expected_doc_ids: list[str]
    retrieved_doc_ids: list[str]
    retrieved_titles: list[str]
    hits: int
    n_expected: int
    recall_at_k: float
    hyde_query: str
    status: str = "ok"
    error: str = ""
    latency_ms: float = 0.0


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def save_outputs(out_dir: Path, rows: list[RecallRow], meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = meta["run_id"]
    json_path = out_dir / f"recall_results_{stamp}.json"
    csv_path = out_dir / f"recall_results_{stamp}.csv"
    summary_path = out_dir / f"recall_summary_{stamp}.md"

    payload = {"meta": meta, "results": [asdict(r) for r in rows]}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    json_path.write_text(text, encoding="utf-8")
    (out_dir / "latest.json").write_text(text, encoding="utf-8")

    flat = [
        {
            "query_id": r.query_id,
            "query_original": r.query_original,
            "expected_doc_ids": "|".join(r.expected_doc_ids),
            "retrieved_doc_ids": "|".join(r.retrieved_doc_ids),
            "hits": r.hits,
            "n_expected": r.n_expected,
            "recall_at_k": r.recall_at_k,
            "status": r.status,
            "error": r.error,
            "latency_ms": r.latency_ms,
        }
        for r in rows
    ]
    df = pd.DataFrame(flat)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    df.to_csv(out_dir / "latest.csv", index=False, encoding="utf-8")

    ok = [r for r in rows if r.status == "ok"]
    recalls = [r.recall_at_k for r in ok]
    hit_rate = (
        sum(1 for r in ok if r.hits > 0) / len(ok) if ok else float("nan")
    )
    perfect = sum(1 for r in ok if r.recall_at_k >= 1.0)

    summary = f"""# Resumen evaluación Recall (local)

- Run ID: `{stamp}`
- Script: `scripts/evaluate_recall.py` (sin Gemini / sin generador Q&A)
- Retriever: HyDE + BM25 (`{meta.get("ollama_model")}` @ `{meta.get("ollama_host")}`)
- Dataset: `{meta.get("dataset")}`
- Queries: **{len(rows)}** (ok={len(ok)}, error={sum(1 for r in rows if r.status == "error")})
- Top-K: `{meta.get("top_k")}`

## Promedios

| Métrica | Score |
|---|---|
| Recall@{meta.get("top_k")} | {mean(recalls):.4f} |
| Hit rate (≥1 doc gold en top-k) | {hit_rate:.4f} |
| Queries con Recall=1.0 | {perfect}/{len(ok)} |

Archivos: `{json_path.name}`, `{csv_path.name}`
"""
    summary_path.write_text(summary, encoding="utf-8")
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    if meta.get("print_summary"):
        print(summary)


def main() -> int:
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Evaluación local Recall@k (HyDE+BM25, sin Gemini)"
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Queries a evaluar (default: {DEFAULT_LIMIT}; 0 = todas)",
    )
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
    retriever = HydeBM25Retriever(corpus, llm_model=ollama_model, host=ollama_host)

    print(
        f"[recall] queries={len(items)} corpus={len(corpus)} "
        f"ollama={ollama_model}@{ollama_host} top_k={args.top_k}"
    )

    rows: list[RecallRow] = []
    for local_idx, item in enumerate(tqdm(items, desc="Recall")):
        query = (item.get("query_original") or "").strip()
        if not query:
            continue
        query_id = f"q{args.offset + local_idx}"
        expected_ids = [str(d.get("doc_id")) for d in item.get("documentos_fuente", [])]
        t0 = time.time()
        try:
            docs, hyde_query = retriever.retrieve(query, top_k=args.top_k)
            retrieved_ids = [str(d["doc_id"]) for d in docs]
            hits = sum(1 for e in expected_ids if e in retrieved_ids)
            recall = hits / len(expected_ids) if expected_ids else 0.0
            rows.append(
                RecallRow(
                    query_id=query_id,
                    query_original=query,
                    expected_doc_ids=expected_ids,
                    retrieved_doc_ids=retrieved_ids,
                    retrieved_titles=[d.get("title", "") for d in docs],
                    hits=hits,
                    n_expected=len(expected_ids),
                    recall_at_k=round(recall, 4),
                    hyde_query=hyde_query,
                    latency_ms=round((time.time() - t0) * 1000, 2),
                )
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                RecallRow(
                    query_id=query_id,
                    query_original=query,
                    expected_doc_ids=expected_ids,
                    retrieved_doc_ids=[],
                    retrieved_titles=[],
                    hits=0,
                    n_expected=len(expected_ids),
                    recall_at_k=0.0,
                    hyde_query="",
                    status="error",
                    error=str(exc),
                    latency_ms=round((time.time() - t0) * 1000, 2),
                )
            )

        meta = {
            "run_id": run_id,
            "design": "local recall only: HyDE+BM25 on query_original",
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
        "design": "local recall only: HyDE+BM25 on query_original",
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
