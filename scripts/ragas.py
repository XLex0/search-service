#!/usr/bin/env python3
"""
Evaluación segmentada del pipeline Centinela (alineada al producto real).

Flujo (igual que front: search → generate-qa):
  1) Retrieve con query_original → Recall@k
  2) QAGenerationPipeline(query, docs) → pares (q_i, a_i)
  3) RAGAS por par: Fidelidad + Answer Relevancy
     (sin Context Relevancy: no aplica bien a Q&A generado desde el contexto)

Uso:
  export GEMINI_API_KEY=...
  .venv-ragas/bin/python scripts/ragas.py          # 8 queries por defecto
  .venv-ragas/bin/python scripts/ragas.py --limit 5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import ollama
import pandas as pd
from dotenv import load_dotenv
from google import genai
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
DEFAULT_OUT_DIR = ROOT / "ragas_results"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:1.5b"
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
DEFAULT_EMBED_MODEL = "gemini-embedding-001"
DEFAULT_LIMIT_QUERIES = 8
FALLBACK_GEMINI_MODELS = (
    "gemini-3.6-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-flash-latest",
)
FALLBACK_EMBED_MODELS = (
    "gemini-embedding-001",
    "gemini-embedding-2",
    "gemini-embedding-2-preview",
)


# ---------------------------------------------------------------------------
# Gemini judge
# ---------------------------------------------------------------------------
class GeminiJudge:
    def __init__(
        self,
        api_key: str,
        model: str,
        embed_model: str,
        temperature: float = 0.0,
    ):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.embed_model = embed_model
        self.temperature = temperature
        self._resolved_model: str | None = None
        self._resolved_embed_model: str | None = None

    def _generate(self, prompt: str) -> str:
        models: list[str] = []
        preferred = self._resolved_model or self.model
        for m in (preferred, *FALLBACK_GEMINI_MODELS):
            if m not in models:
                models.append(m)

        last_err: Exception | None = None
        for model_name in models:
            try:
                res = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={
                        "temperature": self.temperature,
                        "response_mime_type": "application/json",
                        "automatic_function_calling": {"disable": True},
                    },
                )
                self._resolved_model = model_name
                return (res.text or "").strip()
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                continue
        raise RuntimeError(f"Gemini generate_content failed: {last_err}")

    def _parse_json(self, raw: str) -> Any:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", raw)
            if match:
                return json.loads(match.group(0))
            raise

    def embed(self, texts: list[str]) -> np.ndarray:
        models: list[str] = []
        preferred = self._resolved_embed_model or self.embed_model
        for m in (preferred, *FALLBACK_EMBED_MODELS):
            if m not in models:
                models.append(m)

        last_err: Exception | None = None
        for model_name in models:
            try:
                vectors: list[list[float]] = []
                for text in texts:
                    res = self.client.models.embed_content(
                        model=model_name,
                        contents=text,
                    )
                    if hasattr(res, "embeddings") and res.embeddings:
                        values = res.embeddings[0].values
                    elif hasattr(res, "embedding") and res.embedding is not None:
                        values = res.embedding.values
                    else:
                        raise RuntimeError(f"Unexpected embed response: {res!r}")
                    vectors.append(list(values))
                self._resolved_embed_model = model_name
                return np.asarray(vectors, dtype=np.float64)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                continue
        raise RuntimeError(f"Gemini embed_content failed: {last_err}")

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)


def score_faithfulness(judge: GeminiJudge, answer: str, contexts: list[str]) -> dict:
    """F = |V| / |S|."""
    context_block = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    extract_prompt = f"""Divide la siguiente respuesta en afirmaciones atómicas (hechos verificables).
Devuelve SOLO JSON: {{"statements": ["...", "..."]}}

Respuesta:
{answer}
"""
    payload = judge._parse_json(judge._generate(extract_prompt))
    statements = [s.strip() for s in payload.get("statements", []) if str(s).strip()]
    if not statements:
        return {
            "score": 0.0,
            "statements": [],
            "verified": [],
            "n_statements": 0,
            "n_verified": 0,
        }

    verified: list[bool] = []
    for stmt in statements:
        verify_prompt = f"""Dado el contexto recuperado, ¿la afirmación es estrictamente deducible del contexto?
No uses conocimiento externo. Responde SOLO JSON: {{"verdict": true}} o {{"verdict": false}}

Contexto:
{context_block}

Afirmación:
{stmt}
"""
        try:
            verdict = bool(
                judge._parse_json(judge._generate(verify_prompt)).get("verdict", False)
            )
        except Exception:  # noqa: BLE001
            verdict = False
        verified.append(verdict)

    n_v = sum(1 for v in verified if v)
    return {
        "score": round(n_v / len(statements), 4),
        "statements": statements,
        "verified": verified,
        "n_statements": len(statements),
        "n_verified": n_v,
    }


def score_answer_relevancy(
    judge: GeminiJudge, question: str, answer: str, n_questions: int = 3
) -> dict:
    """AR = (1/n) Σ sim(q, q_i)."""
    gen_prompt = f"""A partir de la respuesta, genera exactamente {n_questions} preguntas potenciales
que esa respuesta podría estar contestando. Devuelve SOLO JSON:
{{"questions": ["...", "..."]}}

Respuesta:
{answer}
"""
    payload = judge._parse_json(judge._generate(gen_prompt))
    synth = [q.strip() for q in payload.get("questions", []) if str(q).strip()][
        :n_questions
    ]
    if not synth:
        return {"score": 0.0, "generated_questions": [], "similarities": []}

    vectors = judge.embed([question, *synth])
    q_vec = vectors[0]
    sims = [judge._cosine(q_vec, vectors[i + 1]) for i in range(len(synth))]
    score = float(sum(sims) / len(sims)) if sims else 0.0
    return {
        "score": round(score, 4),
        "generated_questions": synth,
        "similarities": [round(s, 4) for s in sims],
    }


# ---------------------------------------------------------------------------
# Retriever (HyDE + BM25) sobre corpus del dataset
# ---------------------------------------------------------------------------
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
                    "text": self.corpus[doc_id].get("text", ""),
                }
            )
        return docs


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
class PairResult:
    query_id: str
    query_original: str
    pair_index: int
    question: str
    answer: str
    contexts: list[str]
    retrieved_doc_ids: list[str]
    expected_doc_ids: list[str]
    recall_at_k: float
    faithfulness: float
    answer_relevancy: float
    faithfulness_detail: dict = field(default_factory=dict)
    answer_relevancy_detail: dict = field(default_factory=dict)
    generator_status: str = ""
    status: str = "ok"
    error: str = ""
    latency_ms: float = 0.0


def mean_or_nan(values: list[float]) -> float:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def save_outputs(out_dir: Path, rows: list[PairResult], meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = meta.get("run_id", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    json_path = out_dir / f"ragas_results_{stamp}.json"
    csv_path = out_dir / f"ragas_results_{stamp}.csv"
    summary_path = out_dir / f"ragas_summary_{stamp}.md"
    latest_json = out_dir / "latest.json"
    latest_csv = out_dir / "latest.csv"
    latest_summary = out_dir / "summary.md"

    payload = {"meta": meta, "results": [asdict(r) for r in rows]}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    json_path.write_text(text, encoding="utf-8")
    latest_json.write_text(text, encoding="utf-8")

    flat = [
        {
            "query_id": r.query_id,
            "query_original": r.query_original,
            "pair_index": r.pair_index,
            "question": r.question,
            "answer": r.answer,
            "retrieved_doc_ids": "|".join(r.retrieved_doc_ids),
            "expected_doc_ids": "|".join(r.expected_doc_ids),
            "recall_at_k": r.recall_at_k,
            "faithfulness": r.faithfulness,
            "answer_relevancy": r.answer_relevancy,
            "generator_status": r.generator_status,
            "status": r.status,
            "error": r.error,
            "latency_ms": r.latency_ms,
        }
        for r in rows
    ]
    df = pd.DataFrame(flat)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    df.to_csv(latest_csv, index=False, encoding="utf-8")

    scored = [r for r in rows if r.status in {"ok", "partial"}]
    # Recall es por query: promediar una vez por query_id
    recall_by_query: dict[str, float] = {}
    for r in scored:
        recall_by_query[r.query_id] = r.recall_at_k

    summary = f"""# Resumen evaluación RAGAS (segmentada)

- Run ID: `{stamp}`
- Diseño: retrieve(`query_original`) → QAGenerationPipeline → RAGAS(F, AR)
- Modelo juez: `{meta.get("gemini_model")}`
- Embeddings: `{meta.get("embed_model")}`
- Ollama: `{meta.get("ollama_model")}`
- Queries: **{meta.get("n_queries")}** | Pares Q&A: **{len(rows)}**
  (ok={sum(1 for r in rows if r.status == "ok")}, partial={sum(1 for r in rows if r.status == "partial")}, error={sum(1 for r in rows if r.status == "error")})
- Top-K: `{meta.get("top_k")}`
- Compresión / iteración: `{meta.get("use_compression")}` / `{meta.get("use_iterative")}`

## Promedios

| Métrica | Score |
|---|---|
| Recall@{meta.get("top_k")} (por query) | {mean_or_nan(list(recall_by_query.values())):.4f} |
| Fidelidad (faithfulness) | {mean_or_nan([r.faithfulness for r in scored]):.4f} |
| Relevancia de la respuesta | {mean_or_nan([r.answer_relevancy for r in scored]):.4f} |

Archivos: `{json_path.name}`, `{csv_path.name}`
"""
    summary_path.write_text(summary, encoding="utf-8")
    latest_summary.write_text(summary, encoding="utf-8")
    if meta.get("print_summary"):
        print(summary)


def resolve_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if api_key:
        return api_key
    md_path = ROOT.parent / "ragasEvaluation.md"
    if md_path.exists():
        for line in md_path.read_text(encoding="utf-8").splitlines():
            if "GEMINI_API_KEY=" in line:
                return line.split("GEMINI_API_KEY=", 1)[1].strip()
    return ""


def main() -> int:
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Evaluación segmentada RAGAS (Recall + Fidelidad + Answer Relevancy)"
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT_QUERIES,
        help=f"Número de queries (query_original) a evaluar (default: {DEFAULT_LIMIT_QUERIES}; 0 = todas)",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--no-compression", action="store_true")
    parser.add_argument("--no-iterative", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Salta query_id ya presentes en latest.json",
    )
    args = parser.parse_args()

    api_key = resolve_api_key()
    if not api_key:
        print(
            "ERROR: define GEMINI_API_KEY en el entorno o en search-service/.env",
            file=sys.stderr,
        )
        return 1

    gemini_model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    embed_model = os.getenv("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBED_MODEL)
    ollama_host = os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
    if "ollama:" in ollama_host and not Path("/.dockerenv").exists():
        ollama_host = DEFAULT_OLLAMA_HOST
    ollama_model = os.getenv(
        "OLLAMA_QA_MODEL", os.getenv("OLLAMA_HYDE_MODEL", DEFAULT_OLLAMA_MODEL)
    )
    # QAGenerationPipeline lee OLLAMA_HOST / OLLAMA_QA_MODEL del entorno
    os.environ["OLLAMA_HOST"] = ollama_host
    os.environ["OLLAMA_QA_MODEL"] = ollama_model

    if not args.dataset.exists():
        print(f"ERROR: dataset no encontrado: {args.dataset}", file=sys.stderr)
        return 1

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    corpus = load_corpus(dataset)
    items = dataset[args.offset :]
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    existing_rows: list[PairResult] = []
    done_query_ids: set[str] = set()
    if args.resume and (args.out_dir / "latest.json").exists():
        prev = json.loads((args.out_dir / "latest.json").read_text(encoding="utf-8"))
        for item in prev.get("results", []):
            existing_rows.append(PairResult(**item))
            done_query_ids.add(item["query_id"])
        items = [
            it
            for i, it in enumerate(items)
            if f"q{args.offset + i}" not in done_query_ids
            and str(args.offset + dataset.index(it) if it in dataset else i)
            not in done_query_ids
        ]
        # Filtrar por query_original ya evaluada
        done_queries = {r.query_original for r in existing_rows}
        items = [it for it in items if it.get("query_original") not in done_queries]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    judge = GeminiJudge(api_key=api_key, model=gemini_model, embed_model=embed_model)
    retriever = HydeBM25Retriever(corpus, llm_model=ollama_model, host=ollama_host)
    qa_pipeline = QAGenerationPipeline(model_name=ollama_model)
    use_compression = not args.no_compression
    use_iterative = not args.no_iterative

    print(
        f"[ragas] queries={len(items)} corpus={len(corpus)} "
        f"gemini={gemini_model} ollama={ollama_model}@{ollama_host} "
        f"compression={use_compression} iterative={use_iterative}"
    )

    rows: list[PairResult] = list(existing_rows)
    n_queries_done = len({r.query_id for r in rows})

    for local_idx, item in enumerate(tqdm(items, desc="Queries")):
        query = (item.get("query_original") or "").strip()
        if not query:
            continue
        # id estable respecto al offset del dataset original
        absolute_idx = args.offset + local_idx
        query_id = f"q{absolute_idx}"
        t_query = time.time()

        expected_ids = [str(d.get("doc_id")) for d in item.get("documentos_fuente", [])]
        try:
            docs = retriever.retrieve(query, top_k=args.top_k)
            retrieved_ids = [str(d["doc_id"]) for d in docs]
            hits = sum(1 for e in expected_ids if e in retrieved_ids)
            recall = hits / len(expected_ids) if expected_ids else 0.0

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

            # Contextos = docs que realmente usó el generador (post-compresión si aplica)
            contexts = []
            for d in gen.processed_docs:
                title = d.get("title", "")
                text = d.get("text") or d.get("abstract") or ""
                contexts.append(f"{title}. {text}".strip())

            pairs = gen.qa_pairs or []
            if not pairs:
                rows.append(
                    PairResult(
                        query_id=query_id,
                        query_original=query,
                        pair_index=-1,
                        question="",
                        answer="",
                        contexts=contexts,
                        retrieved_doc_ids=retrieved_ids,
                        expected_doc_ids=expected_ids,
                        recall_at_k=round(recall, 4),
                        faithfulness=float("nan"),
                        answer_relevancy=float("nan"),
                        generator_status=gen.status,
                        status="error",
                        error=f"sin pares Q&A: {gen.status}",
                        latency_ms=round((time.time() - t_query) * 1000, 2),
                    )
                )
            else:
                for pair_idx, pair in enumerate(pairs):
                    t_pair = time.time()
                    question = (pair.get("question") or "").strip()
                    answer = (pair.get("answer") or "").strip()
                    metric_errors: list[str] = []
                    faith: dict = {"score": float("nan")}
                    ans_rel: dict = {"score": float("nan")}
                    try:
                        faith = score_faithfulness(judge, answer, contexts)
                    except Exception as exc:  # noqa: BLE001
                        metric_errors.append(f"faithfulness: {exc}")
                    try:
                        ans_rel = score_answer_relevancy(judge, question, answer)
                    except Exception as exc:  # noqa: BLE001
                        metric_errors.append(f"answer_relevancy: {exc}")

                    status = "ok" if not metric_errors else "partial"
                    rows.append(
                        PairResult(
                            query_id=query_id,
                            query_original=query,
                            pair_index=pair_idx,
                            question=question,
                            answer=answer,
                            contexts=contexts,
                            retrieved_doc_ids=retrieved_ids,
                            expected_doc_ids=expected_ids,
                            recall_at_k=round(recall, 4),
                            faithfulness=faith.get("score", float("nan")),
                            answer_relevancy=ans_rel.get("score", float("nan")),
                            faithfulness_detail=faith,
                            answer_relevancy_detail=ans_rel,
                            generator_status=gen.status,
                            status=status,
                            error="; ".join(metric_errors),
                            latency_ms=round((time.time() - t_pair) * 1000, 2),
                        )
                    )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                PairResult(
                    query_id=query_id,
                    query_original=query,
                    pair_index=-1,
                    question="",
                    answer="",
                    contexts=[],
                    retrieved_doc_ids=[],
                    expected_doc_ids=expected_ids,
                    recall_at_k=0.0,
                    faithfulness=float("nan"),
                    answer_relevancy=float("nan"),
                    status="error",
                    error=str(exc),
                    latency_ms=round((time.time() - t_query) * 1000, 2),
                )
            )

        n_queries_done = len({r.query_id for r in rows})
        meta = {
            "run_id": run_id,
            "design": "segmented: retrieve(query_original) -> QAGenerationPipeline -> RAGAS(F,AR)",
            "dataset": str(args.dataset),
            "top_k": args.top_k,
            "gemini_model": judge._resolved_model or gemini_model,
            "embed_model": judge._resolved_embed_model or embed_model,
            "ollama_model": ollama_model,
            "ollama_host": ollama_host,
            "use_compression": use_compression,
            "use_iterative": use_iterative,
            "n_queries": n_queries_done,
            "n_results": len(rows),
            "print_summary": False,
        }
        save_outputs(args.out_dir, rows, meta)

    meta = {
        "run_id": run_id,
        "design": "segmented: retrieve(query_original) -> QAGenerationPipeline -> RAGAS(F,AR)",
        "dataset": str(args.dataset),
        "top_k": args.top_k,
        "gemini_model": judge._resolved_model or gemini_model,
        "embed_model": judge._resolved_embed_model or embed_model,
        "ollama_model": ollama_model,
        "ollama_host": ollama_host,
        "use_compression": use_compression,
        "use_iterative": use_iterative,
        "n_queries": len({r.query_id for r in rows}),
        "n_results": len(rows),
        "print_summary": True,
    }
    save_outputs(args.out_dir, rows, meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
