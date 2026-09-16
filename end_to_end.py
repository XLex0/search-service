import os
import json
import time
import copy
import re
import pandas as pd
import numpy as np
import ollama
from rank_bm25 import BM25Okapi
from pydantic import BaseModel

# =====================================================================
# 0. CONFIGURACIÓN DE EXPERIMENTOS (BANDERAS)
# =====================================================================
# Cambia estos valores a False para probar el impacto de cada componente
USE_COMPRESSION = True
USE_ITERATIVE_CORRECTION = True

# =====================================================================
# 1. ESQUEMAS Y PIPELINE RAG (qwen2.5:1.5b)
# =====================================================================
class LatencyBreakdown(BaseModel):
    hyde_prep_ms: float
    bm25_search_ms: float
    retriever_total_ms: float
    generator_ms: float
    end_to_end_total_ms: float

class RAGEvaluationOutput(BaseModel):
    query: str
    hyde_query: str
    retrieved_docs: list[dict]
    initial_raw_response: str
    generated_raw_response: str
    intentos_utilizados: int
    latency: LatencyBreakdown
    status: str

class HydeBM25Retriever:
    def __init__(self, corpus: dict, llm_model: str = "qwen2.5:1.5b"):
        self.corpus = corpus
        self.corpus_ids = list(corpus.keys())
        self.llm_model = llm_model
        self._build_bm25_index()

    def _build_bm25_index(self):
        corpus_texts = [
            f"{doc.get('title', '')} {doc.get('text', '')}".lower() 
            for doc in self.corpus.values()
        ]
        tokenized_corpus = [doc.split() for doc in corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _generate_hypothetical_document(self, raw_query: str) -> str:
        prompt = (
            f"Act as a scientific researcher. Write a 3-sentence abstract of a fictional "
            f"academic paper that answers the following query: '{raw_query}'. "
            f"Use rigorous academic tone, go straight to the text with no introductions, "
            f"and ALWAYS write in English."
        )
        try:
            res = ollama.chat(
                model=self.llm_model, 
                messages=[{'role': 'user', 'content': prompt}],
                options={"temperature": 0.7, "num_predict": 150} 
            )
            return res['message']['content'].strip()
        except Exception:
            return ""

    def retrieve(self, query: str, top_k: int = 5) -> dict:
        t0 = time.time()
        hypothetical_doc = self._generate_hypothetical_document(query)
        hyde_query = f"{query}\n{hypothetical_doc}" if hypothetical_doc else query
        t_hyde = (time.time() - t0) * 1000

        t1 = time.time()
        tokenized_query = hyde_query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        t_search = (time.time() - t1) * 1000

        docs = []
        for idx in top_indices:
            doc_id = self.corpus_ids[idx]
            docs.append({
                "doc_id": doc_id,
                "score": float(scores[idx]),
                "title": self.corpus[doc_id].get("title", ""),
                "text": self.corpus[doc_id].get("text", "")
            })

        return {"hyde_query": hyde_query, "docs": docs, "t_hyde_ms": t_hyde, "t_search_ms": t_search}

class ContextCompressor:
    def __init__(self, slm_model: str = "qwen2.5:1.5b", temperature: float = 0.0):
        self.slm_model = slm_model
        self.temperature = temperature

    def compress(self, query: str, docs: list[dict]) -> tuple[list[dict], float]:
        compressed_docs = []
        t_start = time.time()
        
        for doc in docs:
            prompt = f"""Actúa como un asistente de investigación científico. 
Tu tarea es leer el siguiente abstract en inglés y extraer SOLO la información útil para responder a la búsqueda del usuario.

Búsqueda del usuario: "{query}"
Abstract original: "{doc.get('text', '')}"

INSTRUCCIONES:
1. Escribe un resumen de máximo 3 oraciones con los datos clave.
2. Escribe el resumen en ESPAÑOL, pero MANTÉN TODOS LOS TÉRMINOS TÉCNICOS, algoritmos y métricas en su idioma ORIGINAL (INGLÉS).
3. Ve directo al grano, sin introducciones.

Resumen estructurado:"""
            try:
                res = ollama.chat(
                    model=self.slm_model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": self.temperature, "num_predict": 200}
                )
                compressed_text = res["message"]["content"].strip()
            except Exception:
                compressed_text = doc.get('text', '') 
            
            comp_doc = doc.copy()
            comp_doc["text"] = compressed_text
            compressed_docs.append(comp_doc)
            
        t_total = (time.time() - t_start) * 1000
        return compressed_docs, t_total

class AcademicQAGenerator:
    # EL PROMPT EXACTO PROBADO CON PROTEGI
    SYSTEM_PROMPT = (
        "Analiza y sintetiza todos los documentos proporcionados. "
        "Genera exactamente 3 pares de preguntas y respuestas técnicas, diversas y rigurosas en español, "
        "incluyendo datos cuantitativos y términos clave. No alucines. Responde únicamente con 3 bloques XML: "
        "<pregunta>Pregunta</pregunta><respuesta>Respuesta</respuesta>. "
        "Prohibido añadir texto plano, introducciones o formatos fuera de las etiquetas XML."
    )

    def __init__(self, slm_model: str = "qwen2.5:1.5b", temperature: float = 0.0):
        self.slm_model = slm_model
        self.temperature = temperature

    def _format_context(self, documents: list[dict]) -> str:
        return "\n\n".join([f"Doc {d.get('doc_id')}: {d.get('title')} | {d.get('text')}" for d in documents])

    def _reparar_etiquetas(self, texto: str) -> str:
        """Normaliza etiquetas XML dañadas por mayúsculas, espacios o mal cierre."""

        def _canon(slash: str, nombre: str) -> str:
            return f"<{slash}{nombre.lower()}>"

        # < Pregunta >, </RESPUESTA/>, <pregunta/> → forma canónica
        texto = re.sub(
            r"<\s*(/?)\s*(pregunta|respuesta)\s*/?\s*>",
            lambda m: _canon(m.group(1), m.group(2)),
            texto,
            flags=re.IGNORECASE,
        )
        # <pregunta  o </respuesta  sin '>' antes del siguiente '<' o del final
        texto = re.sub(
            r"<\s*(/?)\s*(pregunta|respuesta)\b(?![^<>]*>)",
            lambda m: _canon(m.group(1), m.group(2)),
            texto,
            flags=re.IGNORECASE,
        )
        # <pregunta>contenido<respuesta> sin </pregunta>
        texto = re.sub(
            r"(<pregunta>)(.*?)(?=<respuesta>)",
            lambda m: m.group(0) if "</pregunta>" in m.group(2) else f"{m.group(1)}{m.group(2)}</pregunta>",
            texto,
            flags=re.DOTALL,
        )
        # <respuesta>contenido<pregunta> (o EOF) sin </respuesta>
        texto = re.sub(
            r"(<respuesta>)(.*?)(?=<pregunta>|$)",
            lambda m: m.group(0) if "</respuesta>" in m.group(2) else f"{m.group(1)}{m.group(2)}</respuesta>",
            texto,
            flags=re.DOTALL,
        )
        return texto

    def _diagnosticar_fallo(self, texto: str) -> tuple[bool, str, str]:
        texto = self._reparar_etiquetas(texto)
        etiquetas_encontradas = re.findall(r'<[^>]+>', texto)
        if not etiquetas_encontradas:
            return False, "faltan las etiquetas XML obligatorias <pregunta> y <respuesta>", texto

        for tag in etiquetas_encontradas:
            tag_lower = tag.lower().replace(" ", "")
            if tag_lower not in ["<pregunta>", "</pregunta>", "<respuesta>", "</respuesta>"]:
                return False, f"tienes etiquetas mal escritas como '{tag}', usa exactamente <pregunta> y <respuesta> con sus respectivos cierres", texto

        if "<pregunta>" not in texto or "</pregunta>" not in texto or "<respuesta>" not in texto or "</respuesta>" not in texto:
            return False, "faltan las etiquetas XML de apertura o cierre", texto

        preguntas = re.findall(r'<pregunta>(.*?)</pregunta>', texto, flags=re.DOTALL | re.IGNORECASE)
        respuestas = re.findall(r'<respuesta>(.*?)</respuesta>', texto, flags=re.DOTALL | re.IGNORECASE)

        if len(preguntas) != 3 or len(respuestas) != 3:
            return False, f"cantidad incorrecta, hay {len(preguntas)} preguntas y {len(respuestas)} respuestas. Deben ser exactamente 3 de cada una", texto

        return True, "", texto

    def generate(self, user_query: str, retrieved_docs: list[dict], use_iterative: bool) -> tuple[str, str, int, float, str]:
        docs_context = self._format_context(retrieved_docs)
        
        user_content = (
            f"[BÚSQUEDA DEL USUARIO]\n{user_query}\n\n"
            f"[DOCUMENTOS RECUPERADOS]\n{docs_context}\n\n"
            f"RECUERDA: Genera EXACTAMENTE 3 pares. Empieza a generar el XML ahora:"
        )
        
        t0 = time.time()
        initial_raw, current_raw = "", ""
        status = "success"
        intentos = 1

        try:
            # PRIMERA GENERACIÓN (Usa System Prompt y User Prompt)
            res = ollama.chat(
                model=self.slm_model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                options={"temperature": self.temperature, "num_predict": 600}
            )
            current_raw = res["message"]["content"].strip()
            
            if current_raw.startswith("```xml"):
                current_raw = current_raw.replace("```xml", "").replace("```", "").strip()
            
            initial_raw = current_raw
            valido, tipo_fallo, current_raw = self._diagnosticar_fallo(current_raw)
            
            # ITERACIÓN DE CORRECCIÓN (Solo si está activa en la configuración)
            max_intentos = 3 if use_iterative else 1
            
            while not valido and intentos < max_intentos:
                intentos += 1
                prompt_correccion = f"toma en cuenta esto\n[{current_raw}]\ncorrige esto {tipo_fallo}"
                
                res_corr = ollama.chat(
                    model=self.slm_model,
                    # Para corregir, no le pasamos el contexto nuevo, solo el error.
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt_correccion}
                    ],
                    options={"temperature": self.temperature, "num_predict": 600}
                )
                current_raw = res_corr["message"]["content"].strip()
                if current_raw.startswith("```xml"):
                    current_raw = current_raw.replace("```xml", "").replace("```", "").strip()
                
                valido, tipo_fallo, current_raw = self._diagnosticar_fallo(current_raw)

        except Exception as e:
            current_raw = ""
            status = f"error: {str(e)}"

        return initial_raw, current_raw, intentos, (time.time() - t0) * 1000, status


class EndToEndAcademicRAG:
    def __init__(self, corpus: dict, model_name: str = "qwen2.5:1.5b"):
        self.retriever = HydeBM25Retriever(corpus=corpus, llm_model=model_name)
        self.compressor = ContextCompressor(slm_model=model_name) 
        self.generator = AcademicQAGenerator(slm_model=model_name)

    def run(self, query: str, top_k: int = 3, use_comp: bool = True, use_iter: bool = True) -> RAGEvaluationOutput:
        t_start = time.time()
        
        # 1. Retriever
        retrieval_res = self.retriever.retrieve(query=query, top_k=top_k)
        
        # 2. Compresión Opcional
        if use_comp:
            docs_to_generate, comp_lat_ms = self.compressor.compress(query, retrieval_res["docs"])
        else:
            docs_to_generate = retrieval_res["docs"]
            comp_lat_ms = 0.0
            
        # 3. Generación (con o sin iteración)
        initial_raw, final_output, intentos, gen_lat_ms, gen_status = self.generator.generate(query, docs_to_generate, use_iter)
        
        t_total_ms = (time.time() - t_start) * 1000

        latency = LatencyBreakdown(
            hyde_prep_ms=round(retrieval_res["t_hyde_ms"], 2),
            bm25_search_ms=round(retrieval_res["t_search_ms"], 2),
            retriever_total_ms=round(retrieval_res["t_hyde_ms"] + retrieval_res["t_search_ms"], 2),
            generator_ms=round(gen_lat_ms + comp_lat_ms, 2),
            end_to_end_total_ms=round(t_total_ms, 2)
        )

        return RAGEvaluationOutput(
            query=query, hyde_query=retrieval_res["hyde_query"],
            retrieved_docs=docs_to_generate,
            initial_raw_response=initial_raw,
            generated_raw_response=final_output,
            intentos_utilizados=intentos,
            latency=latency, status=gen_status
        )

# =====================================================================
# 2. FUNCIONES DE EVALUACIÓN Y CUADRO COMPARATIVO
# =====================================================================
def load_and_build_corpus(filepath: str) -> tuple[list, dict]:
    with open(filepath, 'r', encoding='utf-8') as f:
        dataset_original = json.load(f)
    dataset_modificado = copy.deepcopy(dataset_original)
    
    global_corpus, title_to_global_id, global_id_counter = {}, {}, 1
    for item in dataset_modificado:
        for doc in item.get("documentos_fuente", []):
            titulo = doc.get("titulo", "").strip()
            if titulo not in title_to_global_id:
                nuevo_id = f"doc_global_{global_id_counter}"
                title_to_global_id[titulo] = nuevo_id
                global_corpus[nuevo_id] = {"title": titulo, "text": doc.get("abstract", "")}
                global_id_counter += 1
            doc["doc_id"] = title_to_global_id[titulo]
    return dataset_modificado, global_corpus

def generate_comparative_table(csv_path: str):
    """Lee el CSV y genera un cuadro comparativo impreso en consola."""
    if not os.path.exists(csv_path):
        return
        
    df = pd.read_csv(csv_path)
    
    # Validar formato para las métricas
    def validar_xml(texto):
        if not isinstance(texto, str): return 0
        p = re.findall(r'<pregunta>(.*?)</pregunta>', texto, flags=re.DOTALL | re.IGNORECASE)
        r = re.findall(r'<respuesta>(.*?)</respuesta>', texto, flags=re.DOTALL | re.IGNORECASE)
        return 1 if len(p) == 3 and len(r) == 3 else 0

    df['Formato_Valido_Bool'] = df['Generated_Raw_Output'].apply(validar_xml)
    
    print("\n" + "="*70)
    print(" 📊 CUADRO COMPARATIVO DE CONFIGURACIONES (PROMEDIOS)")
    print("="*70)
    
    # Agrupar por la configuración usada
    if 'Configuracion' in df.columns:
        summary = df.groupby('Configuracion').agg({
            'Recall@3': 'mean',
            'Formato_Valido_Bool': lambda x: x.mean() * 100, # Porcentaje
            'Intentos_Utilizados': 'mean',
            'Latencia_Total_ms': 'mean'
        }).reset_index()
        
        # Formatear la tabla
        summary.rename(columns={
            'Recall@3': 'Recall Medio', 
            'Formato_Valido_Bool': '% Formato Válido',
            'Intentos_Utilizados': 'Intentos Medios',
            'Latencia_Total_ms': 'Latencia Media (ms)'
        }, inplace=True)
        
        print(summary.to_markdown(index=False, floatfmt=".2f"))
    else:
        print("Aún no hay suficientes datos con configuraciones variadas para agrupar.")
    print("="*70 + "\n")


def evaluate_dataset(filepath: str, output_csv: str = "evaluacion_rag_resultados.csv", top_k: int = 3):
    dataset, corpus = load_and_build_corpus(filepath)
    results = []
    processed_queries = set()
    
    # Etiqueta de la configuración actual
    config_label = f"Compresion:{USE_COMPRESSION} | Iteracion:{USE_ITERATIVE_CORRECTION}"
    
    if os.path.exists(output_csv):
        df_existing = pd.read_csv(output_csv)
        # Filtramos procesadas SOLO de esta misma configuración para no repetir la prueba
        if 'Configuracion' in df_existing.columns:
            df_current_config = df_existing[df_existing['Configuracion'] == config_label]
            processed_queries = set(df_current_config['Query'].tolist())
        results = df_existing.to_dict('records')

    print(f"\n[+] Configuración Actual: {config_label}")
    
    rag = EndToEndAcademicRAG(corpus=corpus, model_name="qwen2.5:1.5b")
    
    for idx, item in enumerate(dataset, 1):
        query = item["query_original"]
        if query in processed_queries:
            continue
            
        print(f"[{idx}/{len(dataset)}] Evaluando: {query}")
        expected_doc_ids = [str(d["doc_id"]) for d in item.get("documentos_fuente", [])]
        expected_qas = json.dumps(item.get("evaluaciones_sinteticas", []), ensure_ascii=False)
        
        res = rag.run(query=query, top_k=top_k, use_comp=USE_COMPRESSION, use_iter=USE_ITERATIVE_CORRECTION)
        
        retrieved_ids = [str(d["doc_id"]) for d in res.retrieved_docs]
        hits = sum(1 for expected_id in expected_doc_ids if expected_id in retrieved_ids)
        recall = hits / len(expected_doc_ids) if expected_doc_ids else 0.0
        
        contexto_recuperado = "\n\n".join([f"Doc {d.get('doc_id')}: {d.get('title')} | {d.get('text')}" for d in res.retrieved_docs])
        
        results.append({
            "Configuracion": config_label,
            "Query": query,
            "HyDE_Query": res.hyde_query,
            "Documentos_Esperados": ", ".join(expected_doc_ids),
            "Documentos_Recuperados": ", ".join(retrieved_ids),
            "Contexto_Recuperado": contexto_recuperado,
            f"Recall@{top_k}": round(recall, 2),
            "Expected_QA_Sintetico": expected_qas,
            "Initial_Raw_Output": res.initial_raw_response,
            "Generated_Raw_Output": res.generated_raw_response,
            "Intentos_Utilizados": res.intentos_utilizados,
            "Latencia_Retriever_ms": res.latency.retriever_total_ms,
            "Latencia_Generator_ms": res.latency.generator_ms,
            "Latencia_Total_ms": res.latency.end_to_end_total_ms,
            "Status": res.status
        })
        
        pd.DataFrame(results).to_csv(output_csv, index=False, encoding='utf-8')
        processed_queries.add(query)

    generate_comparative_table(output_csv)

if __name__ == "__main__":
    DATASET_FILE = "dataset_qg_qa_sintetico.json"
    if os.path.exists(DATASET_FILE):
        evaluate_dataset(DATASET_FILE, output_csv="evaluacion_rag_resultados.csv", top_k=3)
    else:
        print("Error: No se encontró el dataset.")