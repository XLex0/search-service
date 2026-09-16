#!/usr/bin/env python3
"""
Pruebas de latencia: búsqueda → respuesta (pipeline segmentado local).

Mide, por query_original:
  1) Retriever HyDE+BM25 (ms)
  2) QAGenerationPipeline: compresión + generación Q&A (ms)
  3) End-to-end búsqueda→respuesta (ms)

Sin Gemini. Independiente de scripts/ragas.py.

Uso:
  .venv-ragas/bin/python scripts/evaluate_latency.py --limit 15 --top-k 3
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.search_engine.application.services.qa_generator_service import (  # noqa: E402
    QAGenerationPipeline,
    normalize_docs_for_qa,
)

DEFAULT_DATASET = ROOT / "dataset_qg_qa_sintetico.json"
DEFAULT_OUT_DIR = ROOT / "ragas_results" / "latency"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:1.5b"
DEFAULT_LIMIT = 15


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

    def retrieve(self, query: str, top_k: int = 3) -> tuple[list[dict], float, float]:
        t0 = time.perf_counter()
        hypo = self._hyde(query)
        t_hyde = (time.perf_counter() - t0) * 1000

        hyde_query = f"{query}\n{hypo}" if hypo else query
        t1 = time.perf_counter()
        scores = self.bm25.get_scores(hyde_query.lower().split())
        top_indices = np.argsort(scores)[::-1][:top_k]
        t_bm25 = (time.perf_counter() - t1) * 1000

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
        return docs, t_hyde, t_bm25


@dataclass
class LatencyRow:
    query_id: str
    query_original: str
    hyde_ms: float
    bm25_ms: float
    retrieve_ms: float
    compression_ms: float
    generation_ms: float
    generator_total_ms: float
    end_to_end_ms: float
    n_qa_pairs: int
    generator_status: str
    status: str = "ok"
    error: str = ""


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(values, p))


def mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else float("nan")


def save_outputs(out_dir: Path, rows: list[LatencyRow], meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = meta["run_id"]
    payload = {"meta": meta, "results": [asdict(r) for r in rows]}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    (out_dir / f"latency_results_{stamp}.json").write_text(text, encoding="utf-8")
    (out_dir / "latest.json").write_text(text, encoding="utf-8")

    flat = [asdict(r) for r in rows]
    df = pd.DataFrame(flat)
    df.to_csv(out_dir / f"latency_results_{stamp}.csv", index=False, encoding="utf-8")
    df.to_csv(out_dir / "latest.csv", index=False, encoding="utf-8")

    ok = [r for r in rows if r.status == "ok"]

    def col(name: str) -> list[float]:
        return [getattr(r, name) for r in ok]

    def block(title: str, key: str) -> str:
        vals = col(key)
        return (
            f"| {title} | {mean(vals):.1f} | {pct(vals, 50):.1f} | "
            f"{pct(vals, 95):.1f} | {min(vals) if vals else float('nan'):.1f} | "
            f"{max(vals) if vals else float('nan'):.1f} |"
        )

    summary = f"""# Latencia búsqueda → respuesta

- Run ID: `{stamp}`
- Script: `scripts/evaluate_latency.py`
- Flujo: `query_original` → HyDE+BM25 → QAGenerationPipeline
- Modelo Ollama: `{meta.get("ollama_model")}` @ `{meta.get("ollama_host")}`
- Queries: **{len(rows)}** (ok={len(ok)}) | Top-K: `{meta.get("top_k")}`
- Compresión / iteración: `{meta.get("use_compression")}` / `{meta.get("use_iterative")}`

## Resumen (ms)

| Etapa | Media | P50 | P95 | Min | Max |
|---|---|---|---|---|---|
{block("HyDE", "hyde_ms")}
{block("BM25", "bm25_ms")}
{block("Retriever total", "retrieve_ms")}
{block("Compresión", "compression_ms")}
{block("Generación Q&A", "generation_ms")}
{block("Generador total", "generator_total_ms")}
{block("End-to-end (búsqueda→respuesta)", "end_to_end_ms")}

## Detalle por query

| ID | Query | Retrieve | Generador | E2E | Pares |
|---|---|---|---|---|---|
"""
    for r in rows:
        summary += (
            f"| {r.query_id} | {r.query_original[:42]} | "
            f"{r.retrieve_ms:.0f} | {r.generator_total_ms:.0f} | "
            f"{r.end_to_end_ms:.0f} | {r.n_qa_pairs} |\n"
        )

    (out_dir / f"latency_summary_{stamp}.md").write_text(summary, encoding="utf-8")
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    if meta.get("print_summary"):
        print(summary)


def main() -> int:
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Latencia local: búsqueda → respuesta Q&A"
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--no-compression", action="store_true")
    parser.add_argument("--no-iterative", action="store_true")
    args = parser.parse_args()

    ollama_host = os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
    if "ollama:" in ollama_host and not Path("/.dockerenv").exists():
        ollama_host = DEFAULT_OLLAMA_HOST
    ollama_model = os.getenv(
        "OLLAMA_QA_MODEL", os.getenv("OLLAMA_HYDE_MODEL", DEFAULT_OLLAMA_MODEL)
    )
    os.environ["OLLAMA_HOST"] = ollama_host
    os.environ["OLLAMA_QA_MODEL"] = ollama_model

    if not args.dataset.exists():
        print(f"ERROR: dataset no encontrado: {args.dataset}")
        return 1

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    corpus = load_corpus(dataset)
    items = dataset[args.offset :]
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    use_compression = not args.no_compression
    use_iterative = not args.no_iterative
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    retriever = HydeBM25Retriever(corpus, llm_model=ollama_model, host=ollama_host)
    qa_pipeline = QAGenerationPipeline(model_name=ollama_model)

    print(
        f"[latency] queries={len(items)} corpus={len(corpus)} "
        f"ollama={ollama_model}@{ollama_host} top_k={args.top_k}"
    )

    rows: list[LatencyRow] = []
    for local_idx, item in enumerate(tqdm(items, desc="Latency")):
        query = (item.get("query_original") or "").strip()
        if not query:
            continue
        query_id = f"q{args.offset + local_idx}"
        t_e2e = time.perf_counter()
        try:
            docs, hyde_ms, bm25_ms = retriever.retrieve(query, top_k=args.top_k)
            retrieve_ms = hyde_ms + bm25_ms

            qa_docs = normalize_docs_for_qa(
                [
                    {
                        "doc_id": d["doc_id"],
                        "title": d["title"],
                        "abstract": d["text"],
                    }
                    for d in docs
                ],
                limit=args.top_k,
            )
            gen = qa_pipeline.process(
                query=query,
                retrieved_docs=qa_docs,
                use_compression=use_compression,
                use_iterative=use_iterative,
            )
            e2e_ms = (time.perf_counter() - t_e2e) * 1000

            rows.append(
                LatencyRow(
                    query_id=query_id,
                    query_original=query,
                    hyde_ms=round(hyde_ms, 2),
                    bm25_ms=round(bm25_ms, 2),
                    retrieve_ms=round(retrieve_ms, 2),
                    compression_ms=round(gen.latency.compression_ms, 2),
                    generation_ms=round(gen.latency.generation_ms, 2),
                    generator_total_ms=round(gen.latency.total_generator_ms, 2),
                    end_to_end_ms=round(e2e_ms, 2),
                    n_qa_pairs=len(gen.qa_pairs or []),
                    generator_status=gen.status,
                )
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                LatencyRow(
                    query_id=query_id,
                    query_original=query,
                    hyde_ms=0.0,
                    bm25_ms=0.0,
                    retrieve_ms=0.0,
                    compression_ms=0.0,
                    generation_ms=0.0,
                    generator_total_ms=0.0,
                    end_to_end_ms=round((time.perf_counter() - t_e2e) * 1000, 2),
                    n_qa_pairs=0,
                    generator_status="error",
                    status="error",
                    error=str(exc),
                )
            )

        meta = {
            "run_id": run_id,
            "design": "latency search->response: HyDE+BM25 + QAGenerationPipeline",
            "dataset": str(args.dataset),
            "top_k": args.top_k,
            "ollama_model": ollama_model,
            "ollama_host": ollama_host,
            "use_compression": use_compression,
            "use_iterative": use_iterative,
            "n_queries": len(rows),
            "print_summary": False,
        }
        save_outputs(args.out_dir, rows, meta)

    meta = {
        "run_id": run_id,
        "design": "latency search->response: HyDE+BM25 + QAGenerationPipeline",
        "dataset": str(args.dataset),
        "top_k": args.top_k,
        "ollama_model": ollama_model,
        "ollama_host": ollama_host,
        "use_compression": use_compression,
        "use_iterative": use_iterative,
        "n_queries": len(rows),
        "print_summary": True,
    }
    save_outputs(args.out_dir, rows, meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
