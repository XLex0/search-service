# Latencia búsqueda → respuesta

- Run ID: `20260916T191220Z`
- Script: `scripts/evaluate_latency.py`
- Flujo: `query_original` → HyDE+BM25 → QAGenerationPipeline
- Modelo Ollama: `qwen2.5:1.5b` @ `http://127.0.0.1:11434`
- Queries: **15** (ok=15) | Top-K: `3`
- Compresión / iteración: `True` / `True`

## Resumen (ms)

| Etapa | Media | P50 | P95 | Min | Max |
|---|---|---|---|---|---|
| HyDE | 2676.1 | 2452.5 | 3705.0 | 1739.0 | 4511.1 |
| BM25 | 10.0 | 9.8 | 13.3 | 6.4 | 13.4 |
| Retriever total | 2686.1 | 2463.7 | 3715.0 | 1745.4 | 4524.6 |
| Compresión | 20407.3 | 20568.6 | 23815.5 | 16451.6 | 25184.6 |
| Generación Q&A | 15107.3 | 14739.6 | 21472.2 | 7648.2 | 23871.5 |
| Generador total | 35514.6 | 35120.0 | 45287.7 | 24102.5 | 49056.1 |
| End-to-end (búsqueda→respuesta) | 38200.9 | 39118.2 | 48634.4 | 26883.9 | 52424.3 |

## Detalle por query

| ID | Query | Retrieve | Generador | E2E | Pares |
|---|---|---|---|---|---|
| q0 | retrieval augmented generation | 4525 | 34594 | 39118 | 3 |
| q1 | dense passage retrieval | 2781 | 24102 | 26884 | 3 |
| q2 | hybrid search in information retrieval | 1899 | 27581 | 29480 | 3 |
| q3 | vector database indexing algorithms | 3262 | 37128 | 40390 | 3 |
| q4 | bm25 text retrieval optimization | 3368 | 49056 | 52424 | 3 |
| q5 | query expansion techniques in search engin | 2358 | 34984 | 37341 | 3 |
| q6 | hypothetical document embeddings hyde | 3337 | 43673 | 47010 | 3 |
| q7 | reranking models for information retrieval | 2733 | 37037 | 39770 | 3 |
| q8 | cross-encoder for semantic search | 2325 | 37330 | 39655 | 3 |
| q9 | evaluation metrics for RAG systems | 1745 | 35120 | 36865 | 3 |
| q10 | knowledge graph enhanced retrieval | 2432 | 33028 | 35460 | 3 |
| q11 | multi-vector retrieval models | 2464 | 37930 | 40393 | 1 |
| q12 | sparse and dense embeddings fusion | 2432 | 32368 | 34801 | 3 |
| q13 | document parsing techniques for rag | 2789 | 36733 | 39522 | 3 |
| q14 | semantic caching in language models | 1842 | 32057 | 33899 | 3 |
