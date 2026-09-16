from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
from django.test import SimpleTestCase

from apps.search_engine.application.services.llm_search_service import (
    BM25Retriever,
    DenseRetriever,
    HydeBM25Retriever,
    LLMSearchService,
    QueryProcessor,
)


def _make_processor():
    with patch("apps.search_engine.application.services.llm_search_service.spacy.load"), patch(
        "apps.search_engine.application.services.llm_search_service.stopwords.words",
        return_value=["the", "a", "is"],
    ):
        return QueryProcessor(
            kw_model=MagicMock(), translator=MagicMock(), scibert_model=MagicMock()
        )


class QueryProcessorTests(SimpleTestCase):
    def setUp(self):
        self.processor = _make_processor()

    def test_detect_language_flags_spanish_stopwords(self):
        with patch(
            "apps.search_engine.application.services.llm_search_service.stopwords.words",
            return_value=["el", "la", "de"],
        ):
            lang = self.processor.detect_language("el gato negro")
        self.assertEqual(lang, "es")

    def test_detect_language_uses_langdetect_when_not_spanish(self):
        with patch(
            "apps.search_engine.application.services.llm_search_service.stopwords.words",
            return_value=["el", "la"],
        ), patch(
            "apps.search_engine.application.services.llm_search_service.langdetect.detect",
            return_value="fr",
        ):
            lang = self.processor.detect_language("bonjour tout le monde")
        self.assertEqual(lang, "fr")

    def test_detect_language_defaults_to_spanish_on_error(self):
        with patch(
            "apps.search_engine.application.services.llm_search_service.stopwords.words",
            side_effect=Exception("boom"),
        ):
            lang = self.processor.detect_language("text")
        self.assertEqual(lang, "es")

    def test_clean_text_normalizes_whitespace(self):
        cleaned = self.processor.clean_text("Hello,   World!!")
        self.assertEqual(cleaned, "hello world")

    def test_enhance_query_semantically_short_query_unchanged(self):
        result = self.processor.enhance_query_semantically("ai ml")
        self.assertEqual(result, "ai ml")

    def test_enhance_query_semantically_combines_tokens_and_keywords(self):
        token = MagicMock(is_stop=False, is_punct=False, text="quantum")
        self.processor.nlp = MagicMock(return_value=[token])
        self.processor.kw_model.extract_keywords.return_value = [
            ("quantum computing", 0.9)
        ]
        result = self.processor.enhance_query_semantically(
            "quantum computing research field"
        )
        self.assertIn("quantum", result)

    def test_enhance_query_semantically_falls_back_on_error(self):
        self.processor.nlp = MagicMock(side_effect=RuntimeError("boom"))
        result = self.processor.enhance_query_semantically("some long research query")
        self.assertEqual(result, "some long research query")

    def test_process_query_translates_and_extracts_keywords(self):
        with patch.object(self.processor, "detect_language", return_value="es"):
            self.processor.translator.translate.return_value = "hello world research"
            self.processor.kw_model.extract_keywords.return_value = [("hello", 0.5)]
            with patch.object(
                self.processor,
                "enhance_query_semantically",
                return_value="hello world research",
            ):
                enhanced, t_time, kb_time, keywords = self.processor.process_query(
                    "hola mundo"
                )
        self.assertIn("hello", enhanced)
        self.assertEqual(keywords, [("hello", 0.5)])

    def test_process_query_defaults_keywords_when_empty(self):
        with patch.object(self.processor, "detect_language", return_value="en"):
            self.processor.kw_model.extract_keywords.return_value = []
            with patch.object(
                self.processor, "enhance_query_semantically", return_value=""
            ):
                enhanced, _, _, keywords = self.processor.process_query("hello")
        self.assertTrue(len(keywords) == 1)

    def test_process_query_handles_keyword_extraction_error(self):
        with patch.object(self.processor, "detect_language", return_value="en"):
            self.processor.kw_model.extract_keywords.side_effect = RuntimeError("boom")
            enhanced, t_time, kb_time, keywords = self.processor.process_query("hello")
        self.assertEqual(kb_time, 0)


class BM25RetrieverTests(SimpleTestCase):
    def test_retrieve_ranks_by_score(self):
        df = pd.DataFrame(
            {
                "title": ["A", "B", "C"],
                "abstract": ["a1", "b1", "c1"],
                "content": [
                    "machine learning models",
                    "cooking recipes",
                    "gardening tips",
                ],
            }
        )
        retriever = BM25Retriever(df)
        results = retriever.retrieve("machine learning", top_k=3)
        scores_by_title = {r["title"]: r["score"] for r in results}
        self.assertEqual(len(results), 3)
        self.assertGreater(scores_by_title["A"], scores_by_title["B"])


class HydeBM25RetrieverTests(SimpleTestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "article_id": ["a1", "a2"],
                "title": ["Quantum AI", "Cooking"],
                "abstract": ["quantum machine learning", "pasta recipes"],
                "content": ["quantum machine learning", "pasta recipes"],
            }
        )
        self.retriever = HydeBM25Retriever(self.df, llm_model="fake-model")
        self.retriever._client = MagicMock()

    def test_generate_hypothetical_document_calls_ollama(self):
        self.retriever._client.chat.return_value = {
            "message": {"content": "  Hypothetical abstract.  "}
        }
        doc = self.retriever._generate_hypothetical_document("quantum computing")
        self.assertEqual(doc, "Hypothetical abstract.")
        self.retriever._client.chat.assert_called_once()

    def test_generate_hypothetical_document_returns_empty_on_error(self):
        self.retriever._client.chat.side_effect = RuntimeError("ollama down")
        doc = self.retriever._generate_hypothetical_document("query")
        self.assertEqual(doc, "")

    def test_expand_query_appends_hypothetical(self):
        with patch.object(
            self.retriever,
            "_generate_hypothetical_document",
            return_value="fake abstract",
        ):
            expanded = self.retriever.expand_query("quantum")
        self.assertEqual(expanded, "quantum\nfake abstract")

    def test_expand_query_keeps_original_when_hyde_fails(self):
        with patch.object(
            self.retriever, "_generate_hypothetical_document", return_value=""
        ):
            expanded = self.retriever.expand_query("quantum")
        self.assertEqual(expanded, "quantum")

    def test_retrieve_returns_ranked_docs_and_timings(self):
        with patch.object(
            self.retriever, "expand_query", return_value="quantum machine"
        ):
            result = self.retriever.retrieve("quantum", top_k=2)
        self.assertIn("hyde_query", result)
        self.assertEqual(result["hyde_query"], "quantum machine")
        self.assertEqual(len(result["docs"]), 2)
        titles = {doc["title"] for doc in result["docs"]}
        self.assertEqual(titles, {"Quantum AI", "Cooking"})
        self.assertIn("doc_id", result["docs"][0])
        self.assertIn("score", result["docs"][0])
        self.assertIn("t_hyde_ms", result)
        self.assertIn("t_search_ms", result)
        self.assertGreaterEqual(result["docs"][0]["score"], result["docs"][1]["score"])


class DenseRetrieverTests(SimpleTestCase):
    def test_retrieve_without_candidates(self):
        df = pd.DataFrame(
            {
                "title": ["A", "B"],
                "abstract": ["a1", "b1"],
                "content": ["c1", "c2"],
            }
        )
        model = MagicMock()
        model.encode.return_value = MagicMock(
            cpu=lambda: MagicMock(numpy=lambda: np.array([[1.0, 0.0]]))
        )
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        retriever = DenseRetriever(model, embeddings, df)
        results = retriever.retrieve("query", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["title"], "A")

    def test_retrieve_with_candidates(self):
        df = pd.DataFrame(
            {
                "title": ["A", "B"],
                "abstract": ["a1", "b1"],
                "content": ["c1", "c2"],
            }
        )
        model = MagicMock()
        model.encode.return_value = MagicMock(
            cpu=lambda: MagicMock(numpy=lambda: np.array([[1.0, 0.0]]))
        )
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        retriever = DenseRetriever(model, embeddings, df)
        candidates = [{"id": 0}, {"id": 1}]
        results = retriever.retrieve("query", candidates=candidates, top_k=1)
        self.assertEqual(len(results), 1)


class LLMSearchServiceTests(SimpleTestCase):
    def setUp(self):
        LLMSearchService._instance = None
        LLMSearchService._initialized = False

    def tearDown(self):
        LLMSearchService._instance = None
        LLMSearchService._initialized = False

    def test_is_singleton(self):
        with patch.object(LLMSearchService, "initialize_components"):
            first = LLMSearchService()
            second = LLMSearchService()
        self.assertIs(first, second)

    def test_initialize_components_only_runs_once(self):
        with patch.object(
            LLMSearchService, "initialize_components"
        ) as mock_init:
            LLMSearchService()
            LLMSearchService()
        mock_init.assert_called_once()

    def test_initialize_components_downloads_missing_spacy_model(self):
        service = LLMSearchService.__new__(LLMSearchService)
        service.scibert_model_path = "x"
        service.keybert_path = "y"
        service.embeddings_path = "z"
        with patch(
            "apps.search_engine.application.services.llm_search_service.spacy.util.is_package",
            return_value=False,
        ), patch(
            "apps.search_engine.application.services.llm_search_service.spacy.cli.download"
        ) as mock_download, patch(
            "apps.search_engine.application.services.llm_search_service.SentenceTransformer"
        ), patch(
            "apps.search_engine.application.services.llm_search_service.KeyBERT"
        ), patch(
            "apps.search_engine.application.services.llm_search_service.GoogleTranslator"
        ), patch(
            "apps.search_engine.application.services.llm_search_service.QueryProcessor"
        ), patch(
            "apps.search_engine.application.services.llm_search_service.np.load",
            return_value=np.array([[1.0]]),
        ), patch.object(
            service, "get_corpus_from_neo4j", return_value=pd.DataFrame()
        ):
            service.initialize_components()
        mock_download.assert_called_once_with("en_core_web_sm")

    def test_initialize_components_wraps_spacy_errors(self):
        service = LLMSearchService.__new__(LLMSearchService)
        with patch(
            "apps.search_engine.application.services.llm_search_service.spacy.util.is_package",
            side_effect=RuntimeError("boom"),
        ):
            with self.assertRaises(RuntimeError):
                service.initialize_components()

    def test_get_corpus_from_neo4j_builds_dataframe(self):
        service = LLMSearchService.__new__(LLMSearchService)
        fake_row = (
            "1",
            "Title",
            "Abstract",
            "2024-01-01",
            2,
            1,
            ["Ana"],
            ["Uni"],
            ["ai"],
        )
        with patch(
            "apps.search_engine.application.services.llm_search_service.db.cypher_query",
            return_value=([fake_row], None),
        ):
            df = service.get_corpus_from_neo4j(limit=10)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["article_id"], "1")
        self.assertIn("Title", df.iloc[0]["content"])

    def test_get_corpus_from_neo4j_wraps_errors(self):
        service = LLMSearchService.__new__(LLMSearchService)
        with patch(
            "apps.search_engine.application.services.llm_search_service.db.cypher_query",
            side_effect=RuntimeError("boom"),
        ):
            with self.assertRaises(RuntimeError):
                service.get_corpus_from_neo4j()

    def test_search_orchestrates_retrievers(self):
        service = LLMSearchService.__new__(LLMSearchService)
        service.query_processor = MagicMock()
        service.query_processor.process_query.return_value = ("enhanced", 0, 0, [])
        service.df = pd.DataFrame(
            {
                "title": ["A"],
                "abstract": ["a1"],
                "publication_date": ["2024"],
                "author_count": [1],
                "affiliation_count": [1],
                "authors": [["Ana"]],
                "affiliations": [["Uni"]],
                "article_id": ["1"],
                "content": ["some content"],
            }
        )
        service.scibert_model = MagicMock()
        service.corpus_embeddings = np.array([[1.0, 0.0]])

        with patch(
            "apps.search_engine.application.services.llm_search_service.BM25Retriever"
        ) as mock_bm25_cls, patch(
            "apps.search_engine.application.services.llm_search_service.DenseRetriever"
        ) as mock_dense_cls:
            mock_bm25_cls.return_value.retrieve.return_value = [{"id": 0}]
            mock_dense_cls.return_value.retrieve.return_value = [
                {"id": 0, "score": 0.9}
            ]
            results = service.search("ai", top_k=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "A")
        self.assertEqual(results[0]["article_id"], "1")

    def test_search_with_hyde_expands_query(self):
        service = LLMSearchService.__new__(LLMSearchService)
        service.query_processor = MagicMock()
        service.query_processor.process_query.return_value = ("enhanced", 0, 0, [])
        service.df = pd.DataFrame(
            {
                "title": ["A"],
                "abstract": ["a1"],
                "publication_date": ["2024"],
                "author_count": [1],
                "affiliation_count": [1],
                "authors": [["Ana"]],
                "affiliations": [["Uni"]],
                "article_id": ["1"],
                "content": ["some content"],
            }
        )
        service.scibert_model = MagicMock()
        service.corpus_embeddings = np.array([[1.0, 0.0]])

        with patch(
            "apps.search_engine.application.services.llm_search_service.HydeBM25Retriever"
        ) as mock_hyde_cls, patch(
            "apps.search_engine.application.services.llm_search_service.BM25Retriever"
        ) as mock_bm25_cls, patch(
            "apps.search_engine.application.services.llm_search_service.DenseRetriever"
        ) as mock_dense_cls:
            mock_hyde_cls.return_value.expand_query.return_value = "enhanced\nhypothetical"
            mock_bm25_cls.return_value.retrieve.return_value = [{"id": 0}]
            mock_dense_cls.return_value.retrieve.return_value = [
                {"id": 0, "score": 0.9}
            ]
            results = service.search("ai", top_k=5, use_hyde=True)

        mock_hyde_cls.return_value.expand_query.assert_called_once_with("enhanced")
        mock_bm25_cls.return_value.retrieve.assert_called_once()
        self.assertEqual(
            mock_bm25_cls.return_value.retrieve.call_args.args[0],
            "enhanced\nhypothetical",
        )
        self.assertEqual(len(results), 1)

    def test_search_wraps_errors(self):
        service = LLMSearchService.__new__(LLMSearchService)
        service.query_processor = MagicMock()
        service.query_processor.process_query.side_effect = RuntimeError("boom")
        with self.assertRaises(RuntimeError):
            service.search("ai")
