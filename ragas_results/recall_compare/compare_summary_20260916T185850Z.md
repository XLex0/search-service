# Comparación Recall@k: HyDE+BM25 vs TF-IDF

- Run ID: `20260916T185850Z`
- Script: `scripts/compare_recall.py`
- Dataset: `/home/xelan/Documentos/Github/Tesis/CentinelaV3/search-service/dataset_qg_qa_sintetico.json`
- Queries: **30** | Top-K: `3`
- HyDE model: `qwen2.5:1.5b`
- TF-IDF: sklearn `TfidfVectorizer` (1–2 grams, stop_words=english, sublinear_tf) sobre el **mismo** corpus sintético

## Promedios

| Métrica | HyDE+BM25 | TF-IDF | Δ (HyDE − TF-IDF) |
|---|---|---|---|
| Recall@3 | 0.6278 | 0.8778 | -0.2500 |
| Hit rate (≥1 gold) | 0.9000 | 1.0000 | -0.1000 |
| Queries Recall=1.0 | 9/30 | 22/30 | — |
| Latencia media (ms) | 2178.3 | 2.7 | — |

## Ganadores por query

| Método | # queries |
|---|---|
| HyDE+BM25 mejor | 1 |
| TF-IDF mejor | 16 |
| Empate | 13 |

## Detalle

| ID | Query | HyDE | TF-IDF | Δ | Winner |
|---|---|---|---|---|---|
| q0 | retrieval augmented generation | 0.00 | 0.33 | -0.33 | tfidf |
| q1 | dense passage retrieval | 1.00 | 1.00 | +0.00 | tie |
| q2 | hybrid search in information retrieval | 0.33 | 1.00 | -0.67 | tfidf |
| q3 | vector database indexing algorithms | 1.00 | 1.00 | +0.00 | tie |
| q4 | bm25 text retrieval optimization | 0.67 | 0.67 | +0.00 | tie |
| q5 | query expansion techniques in search engines | 0.67 | 1.00 | -0.33 | tfidf |
| q6 | hypothetical document embeddings hyde | 0.67 | 1.00 | -0.33 | tfidf |
| q7 | reranking models for information retrieval | 0.33 | 0.33 | +0.00 | tie |
| q8 | cross-encoder for semantic search | 0.33 | 0.67 | -0.33 | tfidf |
| q9 | evaluation metrics for RAG systems | 1.00 | 1.00 | +0.00 | tie |
| q10 | knowledge graph enhanced retrieval | 0.67 | 1.00 | -0.33 | tfidf |
| q11 | multi-vector retrieval models | 1.00 | 1.00 | +0.00 | tie |
| q12 | sparse and dense embeddings fusion | 0.00 | 1.00 | -1.00 | tfidf |
| q13 | document parsing techniques for rag | 0.67 | 1.00 | -0.33 | tfidf |
| q14 | semantic caching in language models | 0.33 | 1.00 | -0.67 | tfidf |
| q15 | small language models efficiency | 0.67 | 0.67 | +0.00 | tie |
| q16 | prompt engineering strategies | 0.67 | 1.00 | -0.33 | tfidf |
| q17 | parameter efficient fine tuning peft | 1.00 | 1.00 | +0.00 | tie |
| q18 | quantization of large language models | 0.67 | 0.67 | +0.00 | tie |
| q19 | hallucination detection in llms | 1.00 | 1.00 | +0.00 | tie |
| q20 | question generation using neural networks | 0.50 | 1.00 | -0.50 | tfidf |
| q21 | question answering evaluation benchmarks | 0.33 | 0.33 | +0.00 | tie |
| q22 | synthetic dataset generation llm | 0.67 | 1.00 | -0.33 | tfidf |
| q23 | llm reasoning and chain of thought | 0.00 | 1.00 | -1.00 | tfidf |
| q24 | context length extrapolation in transformers | 0.67 | 1.00 | -0.33 | tfidf |
| q25 | llm as a judge evaluation methodology | 1.00 | 0.67 | +0.33 | hyde_bm25 |
| q26 | instruction tuning small language models | 0.33 | 1.00 | -0.67 | tfidf |
| q27 | agentic workflows and multi-agent systems | 0.67 | 1.00 | -0.33 | tfidf |
| q28 | multilingual large language models | 1.00 | 1.00 | +0.00 | tie |
| q29 | prompt injection vulnerability detection | 1.00 | 1.00 | +0.00 | tie |
