import os
import time
import re
import ollama
from pydantic import BaseModel
from typing import List, Dict, Tuple

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_QA_MODEL = "qwen2.5:1.5b"


def _ollama_client() -> ollama.Client:
    """Cliente HTTP al daemon Ollama de rag-service (mismo proceso, otro modelo)."""
    return ollama.Client(host=os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST))


def _qa_model_name() -> str:
    return os.environ.get("OLLAMA_QA_MODEL", DEFAULT_QA_MODEL)

# =====================================================================
# 1. ESQUEMAS DE RESPUESTA
# =====================================================================
EXPECTED_QA_PAIRS = 3


class GenerationLatency(BaseModel):
    compression_ms: float
    generation_ms: float
    total_generator_ms: float


class QAGenerationResult(BaseModel):
    query: str
    processed_docs: List[Dict]
    initial_raw_response: str
    final_raw_response: str
    intentos_utilizados: int
    status: str
    latency: GenerationLatency
    qa_pairs: List[Dict[str, str]] = []

# =====================================================================
# 2. COMPONENTES DEL GENERADOR
# =====================================================================
class ContextCompressor:
    def __init__(self, slm_model: str | None = None, temperature: float = 0.0):
        self.slm_model = slm_model or _qa_model_name()
        self.temperature = temperature
        self._client = _ollama_client()

    def compress(self, query: str, docs: List[Dict]) -> Tuple[List[Dict], float]:
        compressed_docs = []
        t_start = time.time()
        
        for doc in docs:
            # Soporta tanto el formato del script RAG antiguo ('text') como el del nuevo Retriever ('abstract' o 'content')
            original_text = doc.get('text', doc.get('abstract', doc.get('content', '')))
            
            prompt = f"""Actúa como un asistente de investigación científico. 
Tu tarea es leer el siguiente abstract en inglés y extraer SOLO la información útil para responder a la búsqueda del usuario.

Búsqueda del usuario: "{query}"
Abstract original: "{original_text}"

INSTRUCCIONES:
1. Escribe un resumen de máximo 3 oraciones con los datos clave.
2. Escribe el resumen en ESPAÑOL, pero MANTÉN TODOS LOS TÉRMINOS TÉCNICOS, algoritmos y métricas en su idioma ORIGINAL (INGLÉS).
3. Ve directo al grano, sin introducciones.

Resumen estructurado:"""
            try:
                res = self._client.chat(
                    model=self.slm_model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": self.temperature, "num_predict": 200}
                )
                compressed_text = res["message"]["content"].strip()
            except Exception:
                compressed_text = original_text 
            
            comp_doc = doc.copy()
            # Estandarizamos la salida para que el generador siempre lea 'text'
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

    def __init__(self, slm_model: str | None = None, temperature: float = 0.0):
        self.slm_model = slm_model or _qa_model_name()
        self.temperature = temperature
        self._client = _ollama_client()

    def _format_context(self, documents: List[Dict]) -> str:
        formatted = []
        for d in documents:
            doc_id = d.get('doc_id', d.get('id', d.get('article_id', 'N/A')))
            title = d.get('title', 'Sin título')
            text = d.get('text', d.get('abstract', d.get('content', '')))
            formatted.append(f"Doc {doc_id}: {title} | {text}")
        return "\n\n".join(formatted)

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

    @staticmethod
    def _strip_markdown_fence(texto: str) -> str:
        texto = texto.strip()
        if texto.startswith("```xml"):
            texto = texto.replace("```xml", "").replace("```", "").strip()
        elif texto.startswith("```"):
            texto = texto.replace("```", "").strip()
        return texto

    @staticmethod
    def _pair_to_xml(pair: Dict[str, str]) -> str:
        return (
            f"<pregunta>{pair['question']}</pregunta>"
            f"<respuesta>{pair['answer']}</respuesta>"
        )

    @classmethod
    def _pairs_to_xml(cls, pairs: List[Dict[str, str]]) -> str:
        return "".join(cls._pair_to_xml(p) for p in pairs)

    @staticmethod
    def _is_valid_pair(question: str, answer: str) -> bool:
        return bool(question.strip()) and bool(answer.strip())

    def _evaluar_respuesta(
        self, texto: str
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], str, str]:
        """
        Separa pares válidos de inválidos.

        Returns:
            valid_pairs, failed_pairs, diagnosis_message, repaired_text
        """
        texto = self._reparar_etiquetas(texto)
        etiquetas_encontradas = re.findall(r"<[^>]+>", texto)
        if not etiquetas_encontradas:
            return (
                [],
                [],
                "faltan las etiquetas XML obligatorias <pregunta> y <respuesta>",
                texto,
            )

        for tag in etiquetas_encontradas:
            tag_lower = tag.lower().replace(" ", "")
            if tag_lower not in [
                "<pregunta>",
                "</pregunta>",
                "<respuesta>",
                "</respuesta>",
            ]:
                return (
                    [],
                    [],
                    f"tienes etiquetas mal escritas como '{tag}', usa exactamente "
                    "<pregunta> y <respuesta> con sus respectivos cierres",
                    texto,
                )

        if (
            "<pregunta>" not in texto
            or "</pregunta>" not in texto
            or "<respuesta>" not in texto
            or "</respuesta>" not in texto
        ):
            return [], [], "faltan las etiquetas XML de apertura o cierre", texto

        preguntas = re.findall(
            r"<pregunta>(.*?)</pregunta>", texto, flags=re.DOTALL | re.IGNORECASE
        )
        respuestas = re.findall(
            r"<respuesta>(.*?)</respuesta>", texto, flags=re.DOTALL | re.IGNORECASE
        )

        valid_pairs: List[Dict[str, str]] = []
        failed_pairs: List[Dict[str, str]] = []
        paired = min(len(preguntas), len(respuestas))
        for i in range(paired):
            question = preguntas[i].strip()
            answer = respuestas[i].strip()
            pair = {"question": question, "answer": answer}
            if self._is_valid_pair(question, answer):
                valid_pairs.append(pair)
            else:
                failed_pairs.append(pair)

        # Preguntas/respuestas huérfanas (desbalance) → fallidas a reescribir
        for q in preguntas[paired:]:
            failed_pairs.append({"question": q.strip(), "answer": ""})
        for a in respuestas[paired:]:
            failed_pairs.append({"question": "", "answer": a.strip()})

        if not valid_pairs and not failed_pairs:
            diagnosis = "no se pudo extraer ningún par <pregunta>/<respuesta>"
        elif failed_pairs:
            diagnosis = (
                f"hay {len(failed_pairs)} pares inválidos o incompletos que debes "
                f"reescribir; genera exactamente {len(failed_pairs)} pares válidos"
            )
        else:
            diagnosis = ""

        return valid_pairs, failed_pairs, diagnosis, texto

    def _diagnosticar_fallo(self, texto: str) -> Tuple[bool, str, str]:
        """Compat: True solo si hay exactamente EXPECTED_QA_PAIRS pares válidos."""
        valid, failed, msg, repaired = self._evaluar_respuesta(texto)
        completo = len(valid) >= EXPECTED_QA_PAIRS and not failed
        if completo:
            return True, "", repaired
        if not msg and len(valid) < EXPECTED_QA_PAIRS:
            msg = (
                f"cantidad insuficiente, hay {len(valid)} pares válidos. "
                f"Deben ser exactamente {EXPECTED_QA_PAIRS}"
            )
        return False, msg, repaired

    def _system_prompt_for_count(self, n: int) -> str:
        if n == EXPECTED_QA_PAIRS:
            return self.SYSTEM_PROMPT
        return (
            "Analiza y sintetiza todos los documentos proporcionados. "
            f"Genera exactamente {n} pares de preguntas y respuestas técnicas, "
            "diversas y rigurosas en español, incluyendo datos cuantitativos y "
            "términos clave. No alucines. Responde únicamente con bloques XML: "
            "<pregunta>Pregunta</pregunta><respuesta>Respuesta</respuesta>. "
            "Prohibido añadir texto plano, introducciones o formatos fuera de las etiquetas XML."
        )

    def _build_correction_prompt(
        self,
        failed_pairs: List[Dict[str, str]],
        missing: int,
        diagnosis: str,
        last_raw: str,
    ) -> str:
        if failed_pairs:
            failed_xml = self._pairs_to_xml(failed_pairs)
            return (
                f"Necesito exactamente {missing} pares corregidos.\n"
                f"Reescribe SOLO estos pares fallidos en formato XML válido:\n"
                f"[{failed_xml}]\n"
                f"Problema: {diagnosis}\n"
                f"No regeneres pares ya aceptados. Empieza el XML ahora:"
            )
        return (
            f"toma en cuenta esto\n[{last_raw}]\n"
            f"corrige esto {diagnosis}. "
            f"Genera exactamente {missing} pares XML válidos."
        )

    def generate(
        self, user_query: str, retrieved_docs: List[Dict], use_iterative: bool
    ) -> Tuple[str, str, int, float, str, List[Dict[str, str]]]:
        docs_context = self._format_context(retrieved_docs)

        user_content = (
            f"[BÚSQUEDA DEL USUARIO]\n{user_query}\n\n"
            f"[DOCUMENTOS RECUPERADOS]\n{docs_context}\n\n"
            f"RECUERDA: Genera EXACTAMENTE {EXPECTED_QA_PAIRS} pares. "
            "Empieza a generar el XML ahora:"
        )

        t0 = time.time()
        initial_raw = ""
        current_raw = ""
        saved_pairs: List[Dict[str, str]] = []
        failed_pairs: List[Dict[str, str]] = []
        diagnosis = ""
        status = "success"
        intentos = 0
        max_intentos = 3 if use_iterative else 1

        try:
            while len(saved_pairs) < EXPECTED_QA_PAIRS and intentos < max_intentos:
                intentos += 1
                missing = EXPECTED_QA_PAIRS - len(saved_pairs)

                if intentos == 1:
                    messages = [
                        {
                            "role": "system",
                            "content": self._system_prompt_for_count(
                                EXPECTED_QA_PAIRS
                            ),
                        },
                        {"role": "user", "content": user_content},
                    ]
                else:
                    messages = [
                        {
                            "role": "system",
                            "content": self._system_prompt_for_count(missing),
                        },
                        {
                            "role": "user",
                            "content": self._build_correction_prompt(
                                failed_pairs, missing, diagnosis, current_raw
                            ),
                        },
                    ]

                res = self._client.chat(
                    model=self.slm_model,
                    messages=messages,
                    options={"temperature": self.temperature, "num_predict": 600},
                )
                current_raw = self._strip_markdown_fence(res["message"]["content"])
                if intentos == 1:
                    initial_raw = current_raw

                new_valid, failed_pairs, diagnosis, current_raw = self._evaluar_respuesta(
                    current_raw
                )

                # Conserva pares válidos hasta completar el cupo de 3 (sin duplicar pregunta)
                seen_questions = {p["question"].casefold() for p in saved_pairs}
                for pair in new_valid:
                    if len(saved_pairs) >= EXPECTED_QA_PAIRS:
                        break
                    key = pair["question"].casefold()
                    if key in seen_questions:
                        continue
                    seen_questions.add(key)
                    saved_pairs.append(pair)

                # Si aún faltan y no hay fallidos explícitos, pedir los faltantes
                missing = EXPECTED_QA_PAIRS - len(saved_pairs)
                if missing > 0 and not failed_pairs and not diagnosis:
                    diagnosis = (
                        f"faltan {missing} pares. Genera exactamente {missing} "
                        "pares nuevos en XML"
                    )
                    failed_pairs = []

            if len(saved_pairs) >= EXPECTED_QA_PAIRS:
                status = "success"
            elif saved_pairs:
                status = (
                    f"partial: se guardaron {len(saved_pairs)} de "
                    f"{EXPECTED_QA_PAIRS} pares válidos"
                )
            else:
                status = (
                    "error: no se pudo extraer ningún par pregunta/respuesta válido"
                )

        except Exception as e:
            current_raw = self._pairs_to_xml(saved_pairs) if saved_pairs else ""
            if saved_pairs:
                status = f"partial: error tras guardar {len(saved_pairs)} pares: {e}"
            else:
                status = f"error: {str(e)}"

        final_raw = self._pairs_to_xml(saved_pairs) if saved_pairs else current_raw
        return (
            initial_raw,
            final_raw,
            intentos,
            (time.time() - t0) * 1000,
            status,
            saved_pairs,
        )


# =====================================================================
# 3. PIPELINE ORQUESTADOR (El que llamarás desde afuera)
# =====================================================================
class QAGenerationPipeline:
    def __init__(self, model_name: str | None = None):
        resolved = model_name or _qa_model_name()
        self.compressor = ContextCompressor(slm_model=resolved)
        self.generator = AcademicQAGenerator(slm_model=resolved)

    def process(self, query: str, retrieved_docs: List[Dict], use_compression: bool = True, use_iterative: bool = True) -> QAGenerationResult:
        """
        Recibe la query original y la lista de documentos recuperados (top_k).
        Ejecuta la compresión (opcional) y la generación iterativa.
        """
        t_start = time.time()
        
        # 1. Compresión Opcional
        if use_compression:
            docs_to_generate, comp_lat_ms = self.compressor.compress(query, retrieved_docs)
        else:
            docs_to_generate = retrieved_docs
            comp_lat_ms = 0.0
            
        # 2. Generación (con o sin iteración)
        (
            initial_raw,
            final_output,
            intentos,
            gen_lat_ms,
            gen_status,
            qa_pairs,
        ) = self.generator.generate(
            user_query=query,
            retrieved_docs=docs_to_generate,
            use_iterative=use_iterative,
        )
        
        t_total_ms = (time.time() - t_start) * 1000

        latency = GenerationLatency(
            compression_ms=round(comp_lat_ms, 2),
            generation_ms=round(gen_lat_ms, 2),
            total_generator_ms=round(t_total_ms, 2)
        )

        return QAGenerationResult(
            query=query,
            processed_docs=docs_to_generate,
            initial_raw_response=initial_raw,
            final_raw_response=final_output,
            intentos_utilizados=intentos,
            status=gen_status,
            latency=latency,
            qa_pairs=qa_pairs,
        )


def parse_qa_pairs(raw_response: str) -> List[Dict[str, str]]:
    """Extract question/answer pairs from generator XML output."""
    if not raw_response:
        return []
    preguntas = re.findall(
        r"<pregunta>(.*?)</pregunta>", raw_response, flags=re.DOTALL | re.IGNORECASE
    )
    respuestas = re.findall(
        r"<respuesta>(.*?)</respuesta>", raw_response, flags=re.DOTALL | re.IGNORECASE
    )
    pairs = []
    for pregunta, respuesta in zip(preguntas, respuestas, strict=False):
        pairs.append(
            {
                "question": pregunta.strip(),
                "answer": respuesta.strip(),
            }
        )
    return pairs


def normalize_docs_for_qa(documents: List[Dict], limit: int = 3) -> List[Dict]:
    """Map search-hit fields into the shape expected by QAGenerationPipeline."""
    normalized: List[Dict] = []
    for doc in documents[:limit]:
        normalized.append(
            {
                "article_id": str(
                    doc.get("scopus_id")
                    or doc.get("article_id")
                    or doc.get("doc_id")
                    or doc.get("id")
                    or ""
                ),
                "title": doc.get("title") or "",
                "abstract": doc.get("abstract")
                or doc.get("text")
                or doc.get("content")
                or "",
            }
        )
    return normalized


# =====================================================================
# 4. EJEMPLO DE USO (CÓMO LLAMARLO DESDE TU OTRO SCRIPT)
# =====================================================================
if __name__ == "__main__":
    # Simulación de lo que te entregaría tu script de recuperación (LLMSearchService)
    dummy_query = "What are the applications of quantum computing in cryptography?"
    dummy_retrieved_docs = [
        {
            "article_id": "123",
            "title": "Quantum Cryptography Advancements",
            "abstract": "This paper discusses how Shor's algorithm impacts RSA encryption and introduces QKD protocols."
        },
        {
            "article_id": "124",
            "title": "Post-Quantum Security",
            "abstract": "An overview of lattice-based cryptography to defend against quantum attacks."
        }
    ]

    # Inicializas el pipeline una sola vez
    pipeline = QAGenerationPipeline()

    # Le pasas los resultados de la búsqueda
    print("Iniciando generación...")
    resultado = pipeline.process(
        query=dummy_query,
        retrieved_docs=dummy_retrieved_docs,
        use_compression=True,
        use_iterative=True
    )

    print("\n--- RESULTADO FINAL ---")
    print(resultado.final_raw_response)
    print(f"\nIntentos: {resultado.intentos_utilizados}")
    print(f"Latencia Total Generación: {resultado.latency.total_generator_ms} ms")