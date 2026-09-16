# Resumen evaluación Recall (local)

- Run ID: `20260916T183926Z`
- Script: `scripts/evaluate_recall.py` (sin Gemini / sin generador Q&A)
- Retriever: HyDE + BM25 (`qwen2.5:1.5b` @ `http://127.0.0.1:11434`)
- Dataset: `/home/xelan/Documentos/Github/Tesis/CentinelaV3/search-service/dataset_qg_qa_sintetico.json`
- Queries: **30** (ok=30, error=0)
- Top-K: `3`

## Promedios

| Métrica | Score |
|---|---|
| Recall@3 | 0.5833 |
| Hit rate (≥1 doc gold en top-k) | 0.8667 |
| Queries con Recall=1.0 | 7/30 |

Archivos: `recall_results_20260916T183926Z.json`, `recall_results_20260916T183926Z.csv`
