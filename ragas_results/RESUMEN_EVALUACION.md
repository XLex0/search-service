# Resumen de evaluación RAG (Centinela)

## Objetivo

Evaluar el pipeline RAG del sistema de forma **segmentada**, alineada al flujo real del producto (búsqueda y generación de Q&A son independientes), usando las métricas de la metodología tipo Ragas: **Recall**, **Fidelidad** y **Relevancia de la respuesta**. No se aplicó *Context Relevancy*, porque en este diseño las preguntas se generan a partir del contexto recuperado y esa métrica no mide bien el retriever ni el generador.

Las pruebas se separaron en scripts distintos para no mezclar lógica:
- `scripts/ragas.py` → Recall (piloto) + Fidelidad + Answer Relevancy (usa Gemini como juez).
- `scripts/evaluate_recall.py` → solo **Recall@k** ampliado (HyDE+BM25, 100% local).
- `scripts/compare_recall.py` → comparación **HyDE+BM25 vs TF-IDF** (mismo corpus/queries, local).
- `scripts/evaluate_latency.py` → **latencia** búsqueda → respuesta (retrieve + generate-qa).

## Proceso de la prueba RAGAS (segmentada)

1. **Dataset**  
   Se usó `search-service/dataset_qg_qa_sintetico.json`. No se evaluó el dataset completo: se tomaron las **primeras 8 queries** (`query_original`).

2. **Fase A — Retriever**  
   - Entrada: `query_original` (la búsqueda del usuario, no las `pregunta_qg` sintéticas).  
   - Método: HyDE + BM25 sobre el corpus construido a partir de los `documentos_fuente` del dataset.  
   - Salida: top-3 documentos.  
   - Métrica: **Recall@3** frente a los documentos esperados (`documentos_fuente`).

3. **Fase B — Generador**  
   - Entrada: la misma `query_original` + los docs recuperados.  
   - Componente real del sistema: `QAGenerationPipeline` (`qa_generator_service.py`), con compresión de contexto e iteración de corrección activadas.  
   - Salida: hasta 3 pares `(pregunta, respuesta)` en XML, como en producción / front (`generate-qa`).

4. **Fase C — Juez RAGAS (por cada par generado)**  
   Para cada par `(q_i, a_i)` se usaron como contexto los documentos que efectivamente recibió el generador (post-compresión):  
   - **Fidelidad**: se parte la respuesta en afirmaciones y se verifica si cada una es deducible del contexto → \(F = |V|/|S|\).  
   - **Answer Relevancy**: a partir de la respuesta se generan preguntas sintéticas, se embeben y se promedia la similitud coseno con la pregunta original del par → \(AR = \frac{1}{n}\sum sim(q, q_i)\).

5. **Script y artefactos**  
   - Script: `search-service/scripts/ragas.py` (rama `Evaluación`).  
   - Resultados: `search-service/ragas_results/` (`latest.json`, `latest.csv`, `summary.md`).  
   - Run ID: `20260916T181506Z`.

## Qué se usó (interno vs externo)

| Rol | Herramienta / modelo | Uso |
|---|---|---|
| Dataset local | `dataset_qg_qa_sintetico.json` | Queries y docs gold para Recall |
| Retriever + SLM local | **Ollama** + `qwen2.5:1.5b` | HyDE, compresión y generación de pares Q&A (pipeline Centinela) |
| Generador de producto | `QAGenerationPipeline` | Misma lógica que `generate-qa` / front |
| Juez LLM (externo) | **Google Gemini** `gemini-3.6-flash` | Extracción/verificación de afirmaciones y preguntas sintéticas para las métricas |
| Embeddings (externo) | **Gemini** `gemini-embedding-001` | Vectores para Answer Relevancy |
| Librerías de apoyo | `google-genai`, `rank-bm25`, `ollama`, etc. | Cliente API, BM25 y llamadas locales |

No se evaluó el flujo completo “de un solo golpe” ni se re-recuperó con las preguntas generadas: eso desalineaba la prueba respecto al sistema real.

## Configuración del run RAGAS (8 queries)

- Queries: **8**  
- Pares Q&A evaluados: **24** (3 por query; todos `ok`)  
- Top-K: **3**  
- Compresión: sí  
- Corrección iterativa del generador: sí  

## Resultados RAGAS (8 queries)

### Promedios globales

| Métrica | Score |
|---|---|
| Recall@3 (por query, piloto) | **0.6250** |
| Fidelidad (faithfulness) | **0.7139** |
| Relevancia de la respuesta (answer relevancy) | **0.7864** |

### Detalle por query

| ID | Query | Recall@3 | Fidelidad (media de 3 pares) | Answer Relevancy (media) |
|---|---|---|---|---|
| q0 | retrieval augmented generation | 0.00 | 0.75 | 0.83 |
| q1 | dense passage retrieval | 0.67 | 0.00 | 0.70 |
| q2 | hybrid search in information retrieval | 0.33 | 1.00 | 0.88 |
| q3 | vector database indexing algorithms | 1.00 | 1.00 | 0.79 |
| q4 | bm25 text retrieval optimization | 0.67 | 1.00 | 0.66 |
| q5 | query expansion techniques in search engines | 0.67 | 0.50 | 0.82 |
| q6 | hypothetical document embeddings hyde | 1.00 | 1.00 | 0.80 |
| q7 | reranking models for information retrieval | 0.67 | 0.46 | 0.81 |

### Lectura breve (RAGAS)

- El **generador** se sostiene bien en promedio (fidelidad ~0.71, relevancia de respuesta ~0.79).  
- El **retriever** es variable; por eso se amplió la prueba de Recall en un script aparte (abajo).  
- Casos como q1 (fidelidad 0) conviene revisarlos aparte.

---

## Prueba ampliada de Recall (local, 30 queries)

Como Recall **no usa Gemini** (solo Ollama HyDE + BM25), se corrió una evaluación más amplia en script separado.

- Script: `search-service/scripts/evaluate_recall.py`  
- Resultados: `search-service/ragas_results/recall/`  
- Run ID: `20260916T183926Z`  
- Queries: **30** (`query_original`, primeras del dataset)  
- Top-K: **3**  
- Externo: ninguno (solo local)

### Promedios (30 queries — HyDE+BM25)

| Métrica | Score |
|---|---|
| Recall@3 | **0.5833** |
| Hit rate (≥1 doc gold en top-3) | **0.8667** |
| Queries con Recall = 1.0 | **7 / 30** |
| Queries con Recall = 0 | **4 / 30** |

### Detalle por query (30)

| ID | Query | Hits | Recall@3 |
|---|---|---|---|
| q0 | retrieval augmented generation | 0/3 | 0.00 |
| q1 | dense passage retrieval | 3/3 | 1.00 |
| q2 | hybrid search in information retrieval | 0/3 | 0.00 |
| q3 | vector database indexing algorithms | 3/3 | 1.00 |
| q4 | bm25 text retrieval optimization | 3/3 | 1.00 |
| q5 | query expansion techniques in search engines | 2/3 | 0.67 |
| q6 | hypothetical document embeddings hyde | 3/3 | 1.00 |
| q7 | reranking models for information retrieval | 2/3 | 0.67 |
| q8 | cross-encoder for semantic search | 1/3 | 0.33 |
| q9 | evaluation metrics for RAG systems | 1/1 | 1.00 |
| q10 | knowledge graph enhanced retrieval | 2/3 | 0.67 |
| q11 | multi-vector retrieval models | 2/3 | 0.67 |
| q12 | sparse and dense embeddings fusion | 0/1 | 0.00 |
| q13 | document parsing techniques for rag | 2/3 | 0.67 |
| q14 | semantic caching in language models | 2/3 | 0.67 |
| q15 | small language models efficiency | 2/3 | 0.67 |
| q16 | prompt engineering strategies | 2/3 | 0.67 |
| q17 | parameter efficient fine tuning peft | 3/3 | 1.00 |
| q18 | quantization of large language models | 2/3 | 0.67 |
| q19 | hallucination detection in llms | 1/3 | 0.33 |
| q20 | question generation using neural networks | 1/2 | 0.50 |
| q21 | question answering evaluation benchmarks | 1/3 | 0.33 |
| q22 | synthetic dataset generation llm | 2/3 | 0.67 |
| q23 | llm reasoning and chain of thought | 1/3 | 0.33 |
| q24 | context length extrapolation in transformers | 2/3 | 0.67 |
| q25 | llm as a judge evaluation methodology | 2/3 | 0.67 |
| q26 | instruction tuning small language models | 1/3 | 0.33 |
| q27 | agentic workflows and multi-agent systems | 1/3 | 0.33 |
| q28 | multilingual large language models | 3/3 | 1.00 |
| q29 | prompt injection vulnerability detection | 0/2 | 0.00 |

### Lectura breve (Recall 30)

- En promedio Recall@3 ≈ **0.58**, pero el **hit rate es alto (~0.87)**: en la mayoría de queries entra al menos un documento gold en el top-3.  
- El cuello de botella suele ser recuperar **todos** los gold, no encontrar ninguno.  
- Fallos totales (recall 0): q0, q2, q12, q29 — candidatos a análisis cualitativo del retriever/HyDE.

---

## Comparación Recall: HyDE+BM25 vs TF-IDF (30 queries)

Script aparte (no mezcla lógica con `ragas.py` ni con el generador): `search-service/scripts/compare_recall.py`.  
Resultados: `search-service/ragas_results/recall_compare/` · Run ID: `20260916T185850Z`.

**Condiciones justas:** mismo dataset, mismas 30 `query_original`, mismo corpus sintético (501 docs), top-k=3.

**Nota metodológica:** no se usó el pickle TF-IDF de producción (Neo4j), porque el gold del dataset sintético no coincide con ese índice. El baseline es TF-IDF sklearn (`TfidfVectorizer`, unigramas+bigramas, `stop_words=english`, `sublinear_tf`) entrenado **sobre el mismo corpus** que HyDE+BM25.

### Promedios

| Métrica | HyDE+BM25 | TF-IDF | Δ (HyDE − TF-IDF) |
|---|---|---|---|
| Recall@3 | **0.6278** | **0.8778** | **-0.2500** |
| Hit rate (≥1 gold) | 0.9000 | **1.0000** | -0.1000 |
| Queries con Recall=1.0 | 9/30 | **22/30** | — |
| Latencia media | ~2178 ms | **~3 ms** | — |

### Ganadores por query

| Resultado | # queries |
|---|---|
| TF-IDF mejor | **16** |
| Empate | 13 |
| HyDE+BM25 mejor | 1 |

### Lectura breve (comparación)

- En este corpus/keywords del dataset sintético, **TF-IDF gana claramente** en Recall@3 (+0.25) y en cobertura perfecta (22 vs 9).  
- HyDE+BM25 es mucho más lento (~2 s/query por la llamada a Ollama) y, en queries cortas tipo keyword, la expansión HyDE a veces **diluye** la señal léxica que TF-IDF aprovecha bien.  
- El único caso donde HyDE mejora a TF-IDF en esta muestra es q25 (`llm as a judge evaluation methodology`).  
- Implicación para tesis: sobre este benchmark keyword-heavy, el baseline léxico sigue siendo fuerte; HyDE aporta más valor potencial en queries naturales/ambiguas o cuando el vocabulario query↔doc no coincide (habría que medir ese subconjunto aparte).

---

## Latencia búsqueda → respuesta (15 queries)

Script: `search-service/scripts/evaluate_latency.py`  
Resultados: `search-service/ragas_results/latency/` · Run ID: `20260916T191220Z`  
Flujo medido: `query_original` → HyDE+BM25 → `QAGenerationPipeline` (compresión + generación Q&A), modelo local `qwen2.5:1.5b`. Sin Gemini.

### Resumen (ms)

| Etapa | Media | P50 | P95 | Min | Max |
|---|---|---|---|---|---|
| HyDE | 2676 | 2453 | 3705 | 1739 | 4511 |
| BM25 | 10 | 10 | 13 | 6 | 13 |
| Retriever total | 2686 | 2464 | 3715 | 1745 | 4525 |
| Compresión | 20407 | 20569 | 23816 | 16452 | 25185 |
| Generación Q&A | 15107 | 14740 | 21472 | 7648 | 23872 |
| Generador total | 35515 | 35120 | 45288 | 24103 | 49056 |
| **End-to-end (búsqueda→respuesta)** | **38201** | **39118** | **48634** | **26884** | **52424** |

### Lectura breve (latencia)

- El E2E medio es ~**38 s** por query (P95 ~49 s) en este hardware/modelo.  
- El cuello de botella no es BM25 (~10 ms) sino el **LLM local**: compresión (~20 s) + generación (~15 s).  
- HyDE añade ~2.7 s al retrieve; el resto del E2E lo domina el generador.
