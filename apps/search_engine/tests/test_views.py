from unittest.mock import patch

from rest_framework import status
from rest_framework.test import APIRequestFactory, APITestCase

from apps.search_engine.infrastructure.api.v1.views.affiliation_views import (
    AffiliationViewSet,
)
from apps.search_engine.infrastructure.api.v1.views.article_views import (
    ArticleCount,
    ArticleViewSet,
)
from apps.search_engine.infrastructure.api.v1.views.author_views import AuthorViews
from apps.search_engine.infrastructure.api.v1.views.coauthor_views import (
    CoAuthorsViewSet,
)
from apps.search_engine.infrastructure.api.v1.views.llm_search_views import (
    LLMSearchViewSet,
)
from apps.search_engine.infrastructure.api.v1.views.summary_views import SummaryView
from apps.search_engine.infrastructure.api.v1.views.topic_views import TopicViewSet

factory = APIRequestFactory()

ARTICLE_SERVICE = "apps.search_engine.infrastructure.api.v1.views.article_views.ArticleService"
AUTHOR_SERVICE = "apps.search_engine.infrastructure.api.v1.views.author_views.AuthorService"
AFFILIATION_SERVICE_AUTHOR_VIEW = (
    "apps.search_engine.infrastructure.api.v1.views.author_views.AffiliationService"
)
AFFILIATION_SERVICE = (
    "apps.search_engine.infrastructure.api.v1.views.affiliation_views.AffiliationService"
)
TOPIC_SERVICE = "apps.search_engine.infrastructure.api.v1.views.topic_views.TopicService"


class ArticleViewSetTests(APITestCase):
    def setUp(self):
        patcher = patch.object(ArticleViewSet, "article_service")
        self.mock_service = patcher.start()
        self.addCleanup(patcher.stop)

    def test_list_returns_articles(self):
        self.mock_service.find_all.return_value = []
        self.mock_service.find_total_articles.return_value = 0
        request = factory.get("/articles/?page=1&page_size=10")
        response = ArticleViewSet.as_view({"get": "list"})(request)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["total"], 0)

    def test_list_handles_errors(self):
        self.mock_service.find_all.side_effect = RuntimeError("boom")
        request = factory.get("/articles/")
        response = ArticleViewSet.as_view({"get": "list"})(request)
        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)

    def test_retrieve_returns_article_with_authors(self):
        article = type(
            "Obj",
            (),
            {
                "title": "T",
                "abstract": "A",
                "doi": "d",
                "publication_date": "2024",
                "author_count": 1,
                "affiliation_count": 0,
                "corpus": "",
                "scopus_id": "1",
                "affiliations": type("R", (), {"all": lambda self: []})(),
                "topics": type("R", (), {"all": lambda self: []})(),
            },
        )()
        self.mock_service.find_by_id.return_value = article
        self.mock_service.find_authors_by_article.return_value = [["author-list"]]
        request = factory.get("/articles/1/")
        response = ArticleViewSet.as_view({"get": "retrieve"})(request, pk="1")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["authors"], ["author-list"])

    def test_destroy_clears_database(self):
        with patch(
            "apps.search_engine.infrastructure.api.v1.views.article_views.clear_neo4j_database"
        ):
            request = factory.delete("/articles/")
            response = ArticleViewSet.as_view({"delete": "destroy"})(request)
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)

    def test_find_articles_by_author_id(self):
        self.mock_service.find_articles_by_author.return_value = []
        request = factory.get("/articles/find-articles-by-author-id/?author_id=1")
        response = ArticleViewSet.as_view(
            {"get": "find_articles_by_author_id"}
        )(request)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_most_relevant_articles_by_topic_invalid_payload(self):
        request = factory.post(
            "/articles/most-relevant-articles-by-topic/", {}, format="json"
        )
        response = ArticleViewSet.as_view(
            {"post": "most_relevant_articles_by_topic"}
        )(request)
        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)

    def test_most_relevant_articles_by_topic_success(self):
        with patch(
            "apps.search_engine.infrastructure.api.v1.views.article_views.MostRelevantArticlesUseCase"
        ) as mock_usecase_cls:
            mock_usecase_cls.return_value.execute.return_value = (
                [{"scopus_id": "1", "relevance": 0.9}],
                ["2024-01-01"],
            )
            self.mock_service.find_articles_by_ids.return_value = (
                [
                    {
                        "scopus_id": "1",
                        "title": "T",
                        "publication_date": "2024",
                        "author_count": 0,
                        "affiliation_count": 0,
                        "authors": [],
                        "affiliations": [],
                    }
                ],
                1,
            )
            request = factory.post(
                "/articles/most-relevant-articles-by-topic/",
                {"query": "ai", "page": 1, "size": 10},
                format="json",
            )
            response = ArticleViewSet.as_view(
                {"post": "most_relevant_articles_by_topic"}
            )(request)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["total"], 1)

    def test_article_count(self):
        with patch.object(ArticleCount, "article_service") as mock_service:
            mock_service.find_total_articles.return_value = 5
            request = factory.get("/articles/count/")
            response = ArticleCount.as_view()(request)
        self.assertEqual(response.data["total_articles"], 5)

    def test_article_count_handles_errors(self):
        with patch.object(ArticleCount, "article_service") as mock_service:
            mock_service.find_total_articles.side_effect = RuntimeError("boom")
            request = factory.get("/articles/count/")
            response = ArticleCount.as_view()(request)
        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)


class AuthorViewsTests(APITestCase):
    def setUp(self):
        patcher = patch.object(AuthorViews, "author_service")
        self.mock_author_service = patcher.start()
        self.addCleanup(patcher.stop)
        aff_patcher = patch.object(AuthorViews, "affiliation_service")
        self.mock_affiliation_service = aff_patcher.start()
        self.addCleanup(aff_patcher.stop)

    def test_list_returns_authors(self):
        self.mock_author_service.find_all.return_value = []
        request = factory.get("/authors/?page=1&page_size=10")
        response = AuthorViews.as_view({"get": "list"})(request)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["total"], 0)

    def test_find_by_query_success(self):
        self.mock_author_service.find_authors_by_query.return_value = ([], 0)
        request = factory.get("/authors/find_by_query/?query=ana")
        response = AuthorViews.as_view({"get": "find_by_query"})(request)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_find_by_query_handles_errors(self):
        self.mock_author_service.find_authors_by_query.side_effect = RuntimeError(
            "boom"
        )
        request = factory.get("/authors/find_by_query/?query=ana")
        response = AuthorViews.as_view({"get": "find_by_query"})(request)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_retrieve_success(self):
        self.mock_author_service.find_by_id.return_value = type(
            "Obj",
            (),
            {
                "scopus_id": "1",
                "first_name": "A",
                "last_name": "B",
                "auth_name": "B A.",
                "initials": "A.",
                "citation_count": 0,
                "current_affiliation": "",
                "affiliations": type("R", (), {"all": lambda self: []})(),
                "articles": type("R", (), {"all": lambda self: []})(),
                "co_authors": type("R", (), {"all": lambda self: []})(),
                "topics": type("R", (), {"all": lambda self: []})(),
            },
        )()
        request = factory.get("/authors/1/")
        response = AuthorViews.as_view({"get": "retrieve"})(request, pk="1")
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_retrieve_handles_errors(self):
        self.mock_author_service.find_by_id.side_effect = RuntimeError("missing")
        request = factory.get("/authors/1/")
        response = AuthorViews.as_view({"get": "retrieve"})(request, pk="1")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_most_relevant_authors_without_type_filter(self):
        import pandas as pd

        with patch(
            "apps.search_engine.infrastructure.api.v1.views.author_views.MostRelevantAuthorsByTopicUseCase"
        ) as mrbtu, patch(
            "apps.search_engine.infrastructure.api.v1.views.author_views.AffiliationByAuthorsUsecase"
        ) as abau, patch(
            "apps.search_engine.infrastructure.api.v1.views.author_views.AuthorsCommunityUseCase"
        ) as acuc:
            mrbtu.return_value.execute.return_value = pd.Series([0.5], index=["1"])
            abau.return_value.execute.return_value = []
            acuc.return_value.execute.return_value = {
                "nodes": [],
                "links": [],
                "size_nodes": 0,
                "size_links": 0,
            }
            request = factory.post(
                "/authors/most_relevant_authors/",
                {"topic": "ai", "authors_number": 5},
                format="json",
            )
            response = AuthorViews.as_view({"post": "most_relevant_authors"})(request)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_most_relevant_authors_with_type_filter(self):
        import pandas as pd

        with patch(
            "apps.search_engine.infrastructure.api.v1.views.author_views.MostRelevantAuthorsByTopicUseCase"
        ) as mrbtu, patch(
            "apps.search_engine.infrastructure.api.v1.views.author_views.AffiliationByAuthorsUsecase"
        ) as abau, patch(
            "apps.search_engine.infrastructure.api.v1.views.author_views.AuthorsByAffiliationsFiltersUseCase"
        ) as abfu, patch(
            "apps.search_engine.infrastructure.api.v1.views.author_views.AuthorsCommunityUseCase"
        ) as acuc:
            mrbtu.return_value.execute.return_value = pd.Series([0.5], index=["1"])
            abau.return_value.execute.return_value = []
            filtered_author = type("Obj", (), {"scopus_id": "1"})()
            abfu.return_value.execute.return_value = [filtered_author]
            acuc.return_value.execute.return_value = {
                "nodes": [],
                "links": [],
                "size_nodes": 0,
                "size_links": 0,
            }
            request = factory.post(
                "/authors/most_relevant_authors/",
                {"topic": "ai", "authors_number": 5, "type": "include"},
                format="json",
            )
            response = AuthorViews.as_view({"post": "most_relevant_authors"})(request)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_most_relevant_authors_handles_errors(self):
        request = factory.post(
            "/authors/most_relevant_authors/", {}, format="json"
        )
        response = AuthorViews.as_view({"post": "most_relevant_authors"})(request)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)


class AffiliationViewSetTests(APITestCase):
    def setUp(self):
        patcher = patch.object(AffiliationViewSet, "affiliation_service")
        self.mock_service = patcher.start()
        self.addCleanup(patcher.stop)

    def test_list_success(self):
        self.mock_service.find_all.return_value = []
        self.mock_service.find_total_affiliations.return_value = 0
        request = factory.get("/affiliations/")
        response = AffiliationViewSet.as_view({"get": "list"})(request)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_list_handles_errors(self):
        self.mock_service.find_all.side_effect = RuntimeError("boom")
        request = factory.get("/affiliations/")
        response = AffiliationViewSet.as_view({"get": "list"})(request)
        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)

    def test_retrieve_success(self):
        self.mock_service.find_by_id.return_value = type(
            "Obj", (), {"scopus_id": "1", "name": "Uni", "city": "", "country": ""}
        )()
        request = factory.get("/affiliations/1/")
        response = AffiliationViewSet.as_view({"get": "retrieve"})(request, pk="1")
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_retrieve_handles_errors(self):
        self.mock_service.find_by_id.side_effect = RuntimeError("boom")
        request = factory.get("/affiliations/1/")
        response = AffiliationViewSet.as_view({"get": "retrieve"})(request, pk="1")
        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)


class CoAuthorsViewSetTests(APITestCase):
    def test_coauthors_by_id_success(self):
        with patch.object(
            CoAuthorsViewSet, "co_author_service"
        ) as mock_service, patch.object(CoAuthorsViewSet, "author_service"):
            mock_service.find_coauthors_by_id.return_value = ([], [])
            request = factory.get("/coauthors/1/")
            response = CoAuthorsViewSet.as_view({"get": "coauthors_by_id"})(
                request, pk="1"
            )
        self.assertEqual(response.data["data"]["nodes"], [])

    def test_coauthors_by_id_does_not_exist(self):
        from apps.search_engine.domain.entities.author import Author

        with patch.object(
            CoAuthorsViewSet, "co_author_service"
        ) as mock_service, patch.object(CoAuthorsViewSet, "author_service"):
            mock_service.find_coauthors_by_id.side_effect = Author.DoesNotExist(
                Author
            )
            request = factory.get("/coauthors/1/")
            response = CoAuthorsViewSet.as_view({"get": "coauthors_by_id"})(
                request, pk="1"
            )
        self.assertEqual(response.data["error"], "Author not found.")

    def test_coauthors_by_id_handles_generic_error(self):
        with patch.object(
            CoAuthorsViewSet, "co_author_service"
        ) as mock_service, patch.object(CoAuthorsViewSet, "author_service"):
            mock_service.find_coauthors_by_id.side_effect = RuntimeError("boom")
            request = factory.get("/coauthors/1/")
            response = CoAuthorsViewSet.as_view({"get": "coauthors_by_id"})(
                request, pk="1"
            )
        self.assertEqual(response.data["error"], "boom")


class SummaryViewTests(APITestCase):
    def test_get_summary_success(self):
        with patch(
            "apps.search_engine.infrastructure.api.v1.views.summary_views.GetSummaryUseCase"
        ) as mock_usecase_cls:
            mock_usecase_cls.return_value.execute.return_value = {
                "authors": 1,
                "topics": 2,
                "articles": 3,
            }
            request = factory.get("/summary/")
            response = SummaryView.as_view()(request)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["authors"], 1)

    def test_get_summary_handles_errors(self):
        with patch(
            "apps.search_engine.infrastructure.api.v1.views.summary_views.GetSummaryUseCase"
        ) as mock_usecase_cls:
            mock_usecase_cls.return_value.execute.side_effect = RuntimeError("boom")
            request = factory.get("/summary/")
            response = SummaryView.as_view()(request)
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)


class TopicViewSetTests(APITestCase):
    def test_list_success(self):
        with patch.object(TopicViewSet, "topic_service") as mock_service:
            mock_service.find_all.return_value = []
            request = factory.get("/topics/")
            response = TopicViewSet.as_view({"get": "list"})(request)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["topics"], [])

    def test_list_raises_wrapped_exception(self):
        with patch.object(TopicViewSet, "topic_service") as mock_service:
            mock_service.find_all.side_effect = RuntimeError("boom")
            request = factory.get("/topics/")
            with self.assertRaises(Exception):
                TopicViewSet.as_view({"get": "list"})(request)


class LLMSearchViewSetTests(APITestCase):
    def test_semantic_search_requires_query(self):
        request = factory.post("/llm-search/semantic-search/", {}, format="json")
        response = LLMSearchViewSet.as_view({"post": "semantic_search"})(request)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_semantic_search_returns_transformed_results(self):
        with patch.object(LLMSearchViewSet, "llm_search_service") as mock_service:
            mock_service.search.return_value = [
                {
                    "title": "T",
                    "abstract": "A",
                    "author_count": 1,
                    "affiliation_count": 1,
                    "publication_date": "2024-01-01",
                    "article_id": "1",
                    "authors": [],
                    "affiliations": [],
                    "relevance_score": 0.9,
                }
            ]
            request = factory.post(
                "/llm-search/semantic-search/", {"query": "ai"}, format="json"
            )
            response = LLMSearchViewSet.as_view({"post": "semantic_search"})(request)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["total"], 1)
        self.assertIn("qa_context_hint", response.data)
        self.assertEqual(len(response.data["qa_context_hint"]["documents"]), 1)

    def test_semantic_search_handles_errors(self):
        with patch.object(LLMSearchViewSet, "llm_search_service") as mock_service:
            mock_service.search.side_effect = RuntimeError("boom")
            request = factory.post(
                "/llm-search/semantic-search/", {"query": "ai"}, format="json"
            )
            response = LLMSearchViewSet.as_view({"post": "semantic_search"})(request)
        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)

    def test_generate_qa_requires_query_and_documents(self):
        request = factory.post("/llm-search/generate-qa/", {}, format="json")
        response = LLMSearchViewSet.as_view({"post": "generate_qa"})(request)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

        request = factory.post(
            "/llm-search/generate-qa/", {"query": "ai"}, format="json"
        )
        response = LLMSearchViewSet.as_view({"post": "generate_qa"})(request)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_generate_qa_returns_pairs(self):
        from apps.search_engine.application.services.qa_generator_service import (
            GenerationLatency,
            QAGenerationResult,
        )

        fake = QAGenerationResult(
            query="ai",
            processed_docs=[],
            initial_raw_response="",
            final_raw_response=(
                "<pregunta>Q1</pregunta><respuesta>A1</respuesta>"
                "<pregunta>Q2</pregunta><respuesta>A2</respuesta>"
                "<pregunta>Q3</pregunta><respuesta>A3</respuesta>"
            ),
            intentos_utilizados=1,
            status="success",
            latency=GenerationLatency(
                compression_ms=1.0, generation_ms=2.0, total_generator_ms=3.0
            ),
            qa_pairs=[
                {"question": "Q1", "answer": "A1"},
                {"question": "Q2", "answer": "A2"},
                {"question": "Q3", "answer": "A3"},
            ],
        )
        with patch.object(LLMSearchViewSet, "qa_pipeline") as mock_pipeline:
            mock_pipeline.process.return_value = fake
            request = factory.post(
                "/llm-search/generate-qa/",
                {
                    "query": "ai",
                    "documents": [
                        {"scopus_id": "1", "title": "T", "abstract": "abs"},
                        {"scopus_id": "2", "title": "T2", "abstract": "abs2"},
                        {"scopus_id": "3", "title": "T3", "abstract": "abs3"},
                        {"scopus_id": "4", "title": "ignored", "abstract": "x"},
                    ],
                },
                format="json",
            )
            response = LLMSearchViewSet.as_view({"post": "generate_qa"})(request)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data["qa_pairs"]), 3)
        self.assertEqual(len(response.data["source_docs"]), 3)
        mock_pipeline.process.assert_called_once()
        docs_arg = mock_pipeline.process.call_args.kwargs["retrieved_docs"]
        self.assertEqual(len(docs_arg), 3)

    def test_generate_qa_rejects_when_no_valid_pairs(self):
        from apps.search_engine.application.services.qa_generator_service import (
            GenerationLatency,
            QAGenerationResult,
        )

        fake = QAGenerationResult(
            query="ai",
            processed_docs=[],
            initial_raw_response="",
            final_raw_response="texto invalido",
            intentos_utilizados=3,
            status="error: no se pudo extraer ningún par pregunta/respuesta válido",
            latency=GenerationLatency(
                compression_ms=1.0, generation_ms=2.0, total_generator_ms=3.0
            ),
            qa_pairs=[],
        )
        with patch.object(LLMSearchViewSet, "qa_pipeline") as mock_pipeline:
            mock_pipeline.process.return_value = fake
            request = factory.post(
                "/llm-search/generate-qa/",
                {
                    "query": "ai",
                    "documents": [{"scopus_id": "1", "title": "T", "abstract": "a"}],
                },
                format="json",
            )
            response = LLMSearchViewSet.as_view({"post": "generate_qa"})(request)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data["status"].startswith("error:"))
        self.assertIn("error", response.data)
        self.assertEqual(response.data["qa_pairs"], [])

    def test_generate_qa_returns_partial_pairs(self):
        from apps.search_engine.application.services.qa_generator_service import (
            GenerationLatency,
            QAGenerationResult,
        )

        fake = QAGenerationResult(
            query="ai",
            processed_docs=[],
            initial_raw_response="",
            final_raw_response="<pregunta>Q1</pregunta><respuesta>A1</respuesta>",
            intentos_utilizados=3,
            status="partial: se guardaron 1 de 3 pares válidos",
            latency=GenerationLatency(
                compression_ms=1.0, generation_ms=2.0, total_generator_ms=3.0
            ),
            qa_pairs=[{"question": "Q1", "answer": "A1"}],
        )
        with patch.object(LLMSearchViewSet, "qa_pipeline") as mock_pipeline:
            mock_pipeline.process.return_value = fake
            request = factory.post(
                "/llm-search/generate-qa/",
                {
                    "query": "ai",
                    "documents": [{"scopus_id": "1", "title": "T", "abstract": "a"}],
                },
                format="json",
            )
            response = LLMSearchViewSet.as_view({"post": "generate_qa"})(request)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data["status"].startswith("partial:"))
        self.assertEqual(len(response.data["qa_pairs"]), 1)
        self.assertEqual(response.data["qa_pairs"][0]["question"], "Q1")

    def test_generate_qa_soft_fails_without_500(self):
        with patch.object(LLMSearchViewSet, "qa_pipeline") as mock_pipeline:
            mock_pipeline.process.side_effect = RuntimeError("ollama down")
            request = factory.post(
                "/llm-search/generate-qa/",
                {
                    "query": "ai",
                    "documents": [{"scopus_id": "1", "title": "T", "abstract": "a"}],
                },
                format="json",
            )
            response = LLMSearchViewSet.as_view({"post": "generate_qa"})(request)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data["status"].startswith("error:"))
        self.assertEqual(response.data["qa_pairs"], [])

    def test_llm_search_service_lazily_instantiated(self):
        LLMSearchViewSet._llm_search_service = None
        with patch(
            "apps.search_engine.infrastructure.api.v1.views.llm_search_views.LLMSearchService"
        ) as mock_cls:
            view = LLMSearchViewSet()
            service = view.llm_search_service
            mock_cls.assert_called_once()
            self.assertIs(service, mock_cls.return_value)
        LLMSearchViewSet._llm_search_service = None