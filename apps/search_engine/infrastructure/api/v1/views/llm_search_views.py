import logging

from drf_spectacular.utils import extend_schema
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from apps.search_engine.application.services.llm_search_service import LLMSearchService
from apps.search_engine.application.services.qa_generator_service import (
    EXPECTED_QA_PAIRS,
    QAGenerationPipeline,
    normalize_docs_for_qa,
)

logger = logging.getLogger(__name__)

# Top-N search hits used as context for the independent QA generation call.
QA_CONTEXT_DOCS = 3


class LLMSearchViewSet(viewsets.ViewSet):
    _llm_search_service = None
    _qa_pipeline = None

    @property
    def llm_search_service(self):
        # Instanciacion perezosa: LLMSearchService carga modelos SciBERT/KeyBERT
        # pesados. Si esto se hace a nivel de clase (import time), cualquier
        # test que cargue el urlconf completo (via Django test client) crashea
        # aunque no toque este endpoint.
        if LLMSearchViewSet._llm_search_service is None:
            LLMSearchViewSet._llm_search_service = LLMSearchService()
        return LLMSearchViewSet._llm_search_service

    @property
    def qa_pipeline(self):
        if LLMSearchViewSet._qa_pipeline is None:
            LLMSearchViewSet._qa_pipeline = QAGenerationPipeline()
        return LLMSearchViewSet._qa_pipeline

    @extend_schema(
        summary="Search using LLM",
        description=(
            "Search articles using LLM-based semantic search. "
            "Returns ranked hits only. QA generation is a separate request "
            "(`generate-qa`) so search latency/failures stay independent."
        ),
        tags=["Search"],
    )
    @action(detail=False, methods=["post"], url_path="semantic-search")
    def semantic_search(self, request):
        try:
            query = request.data.get("query")
            if not query:
                return Response(
                    {"error": "Query is required"}, status=status.HTTP_400_BAD_REQUEST
                )
            top_k = request.data.get("top_k", 10)
            use_hyde = bool(request.data.get("use_hyde", False))

            results = self.llm_search_service.search(query, top_k, use_hyde=use_hyde)

            transformed_results = []
            for result in results:
                transformed_results.append(
                    {
                        "title": result["title"],
                        "abstract": result["abstract"],
                        "author_count": result["author_count"],
                        "affiliation_count": result["affiliation_count"],
                        "publication_date": result["publication_date"],
                        "scopus_id": result["article_id"],
                        "authors": result["authors"],
                        "affiliations": result["affiliations"],
                        "relevance": result["relevance_score"],
                    }
                )

            response_data = {
                "data": transformed_results,
                "years": list(
                    set(
                        r["publication_date"].split("-")[0]
                        for r in results
                        if r["publication_date"]
                    )
                ),
                "total": len(results),
                # Hint for clients: fire generate-qa with these top hits next.
                "qa_context_hint": {
                    "endpoint": "generate-qa",
                    "top_n": QA_CONTEXT_DOCS,
                    "documents": transformed_results[:QA_CONTEXT_DOCS],
                },
            }

            return Response(response_data, status=status.HTTP_200_OK)

        except Exception as e:
            return Response(
                {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @extend_schema(
        summary="Generate academic Q&A from search hits",
        description=(
            "Independent from semantic-search. Pass the original query and the "
            f"top {QA_CONTEXT_DOCS} search hits; uses Ollama (rag-service daemon) "
            "via QAGenerationPipeline. Failures here must not affect search results."
        ),
        tags=["Search"],
    )
    @action(detail=False, methods=["post"], url_path="generate-qa")
    def generate_qa(self, request):
        query = request.data.get("query")
        documents = request.data.get("documents")
        if documents is None:
            documents = request.data.get("data")

        if not query:
            return Response(
                {"error": "Query is required"}, status=status.HTTP_400_BAD_REQUEST
            )
        if not isinstance(documents, list) or not documents:
            return Response(
                {"error": "documents must be a non-empty list (top search hits)"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        use_compression = bool(request.data.get("use_compression", True))
        use_iterative = bool(request.data.get("use_iterative", True))
        top_docs = normalize_docs_for_qa(documents, limit=QA_CONTEXT_DOCS)
        source_docs = [
            {
                "article_id": d.get("article_id"),
                "title": d.get("title"),
            }
            for d in top_docs
        ]

        def _error_payload(
            error_status: str,
            *,
            raw_response: str = "",
            intentos: int = 0,
            latency: dict | None = None,
        ) -> dict:
            return {
                "query": query,
                "status": error_status,
                "error": error_status,
                "qa_pairs": [],
                "raw_response": raw_response,
                "intentos_utilizados": intentos,
                "latency": latency
                or {
                    "compression_ms": 0.0,
                    "generation_ms": 0.0,
                    "total_generator_ms": 0.0,
                },
                "source_docs": source_docs,
            }

        try:
            result = self.qa_pipeline.process(
                query=query,
                retrieved_docs=top_docs,
                use_compression=use_compression,
                use_iterative=use_iterative,
            )

            qa_pairs = [
                pair
                for pair in (result.qa_pairs or [])
                if pair.get("question", "").strip() and pair.get("answer", "").strip()
            ]

            # Soft-fail total solo si no quedó ningún par usable.
            if not qa_pairs:
                return Response(
                    _error_payload(
                        result.status
                        if str(result.status).startswith("error")
                        else (
                            "error: la respuesta de generación está vacía o no tiene "
                            "pares pregunta/respuesta válidos"
                        ),
                        raw_response=result.final_raw_response or "",
                        intentos=result.intentos_utilizados,
                        latency=result.latency.model_dump(),
                    ),
                    status=status.HTTP_200_OK,
                )

            return Response(
                {
                    "query": result.query,
                    "status": result.status,
                    "qa_pairs": qa_pairs,
                    "raw_response": result.final_raw_response,
                    "intentos_utilizados": result.intentos_utilizados,
                    "latency": result.latency.model_dump(),
                    "source_docs": source_docs,
                    "expected_pairs": EXPECTED_QA_PAIRS,
                },
                status=status.HTTP_200_OK,
            )
        except Exception as e:
            logger.exception("generate-qa failed independently of search")
            return Response(
                _error_payload(f"error: {e}"),
                status=status.HTTP_200_OK,
            )
