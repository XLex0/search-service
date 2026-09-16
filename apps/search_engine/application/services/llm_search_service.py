import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import langdetect
import numpy as np
import ollama
import pandas as pd
import spacy
import torch
from deep_translator import GoogleTranslator
from keybert import KeyBERT
from neomodel import db
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_HYDE_MODEL = "qwen2.5:1.5b"


def _ollama_client() -> ollama.Client:
    return ollama.Client(host=os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST))


def _hyde_model_name() -> str:
    return os.environ.get("OLLAMA_HYDE_MODEL", os.environ.get("OLLAMA_QA_MODEL", DEFAULT_HYDE_MODEL))


class QueryProcessor:
    def __init__(
        self,
        kw_model: KeyBERT,
        translator: GoogleTranslator,
        scibert_model: SentenceTransformer,
    ):
        self.kw_model = kw_model
        self.translator = translator
        self.scibert_model = scibert_model
        self.nlp = spacy.load("en_core_web_sm")
        self.stop_words = set(stopwords.words("english"))

    def detect_language(self, text: str) -> str:
        try:
            text_lower = text.lower()
            words = text_lower.split()
            spanish_stopwords = set(stopwords.words("spanish"))

            if any(word in spanish_stopwords for word in words):
                logger.info("Spanish word(s) detected, treating as Spanish")
                return "es"

            lang = langdetect.detect(text)
            logger.info(f"Detected language: {lang}")
            return lang

        except Exception as e:
            logger.warning(f"Language detection error: {e}. Using 'en' as default")
            return "es"

    def clean_text(self, text: str) -> str:
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip().lower()

    def enhance_query_semantically(self, query: str) -> str:
        word_count = len(query.split())
        if word_count <= 3:
            logger.info(
                f"Short query detected ({word_count} words). Preserving original"
            )
            return query.lower()

        try:
            doc = self.nlp(query)
            keywords = self.kw_model.extract_keywords(
                query,
                top_n=5,
                keyphrase_ngram_range=(1, 3),
                stop_words="english",
                use_maxsum=True,
                nr_candidates=20,
            )

            terms = []
            used_words = set()

            for word in doc:
                if (
                    not word.is_stop
                    and not word.is_punct
                    and word.text.lower() not in used_words
                ):
                    terms.append(word.text.lower())
                    used_words.add(word.text.lower())

            for keyword, _ in keywords:
                words = keyword.lower().split()
                if any(word not in used_words for word in words):
                    for word in words:
                        if word not in used_words:
                            terms.append(word)
                            used_words.add(word)

            enhanced_query = " ".join(terms)
            if not enhanced_query.strip():
                return query.lower()

            return enhanced_query

        except Exception as e:
            logger.error(f"Query enhancement error: {str(e)}")
            return query.lower()

    def process_query(
        self, query: str
    ) -> tuple[str, float, float, list[tuple[str, float]]]:
        logger.info("=" * 50)
        logger.info("Starting query processing")
        logger.info(f"Original query: '{query}'")

        translation_time = 0
        translated_query = query
        try:
            lang = self.detect_language(query)
            if lang != "en":
                translation_start = time.time()
                translated_query = self.translator.translate(query)
                translation_time = time.time() - translation_start
                logger.info(f"Translated query: '{translated_query}'")
        except Exception as e:
            logger.error(f"Translation error: {e}")

        cleaned_query = self.clean_text(translated_query)
        keybert_start = time.time()

        try:
            keywords = self.kw_model.extract_keywords(
                cleaned_query,
                top_n=5,
                keyphrase_ngram_range=(1, 3),
                use_maxsum=True,
                nr_candidates=20,
            )

            if not keywords:
                keywords = [(cleaned_query, 1.0)]

            enhanced_query = self.enhance_query_semantically(cleaned_query)
            enhanced_query = " ".join(
                [word for word in enhanced_query.split() if word not in self.stop_words]
            )

            if not enhanced_query:
                enhanced_query = cleaned_query

            keybert_time = time.time() - keybert_start
            logger.info(f"Enhanced query: '{enhanced_query}'")

            return enhanced_query, translation_time, keybert_time, keywords

        except Exception as e:
            logger.error(f"Keyword extraction error: {str(e)}")
            return cleaned_query, translation_time, 0, [(cleaned_query, 1.0)]


class BM25Retriever:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.tokenized_docs = [doc.lower().split() for doc in df["content"]]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def retrieve(self, query: str, top_k: int = 1000) -> list[dict]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append(
                {
                    "id": idx,
                    "title": self.df.iloc[idx]["title"],
                    "abstract": self.df.iloc[idx]["abstract"],
                    "content": self.df.iloc[idx]["content"],
                    "score": float(scores[idx]),
                }
            )
        return results


class HydeBM25Retriever:
    """HyDE + BM25 retriever ported from end_to_end.py (no evaluation)."""

    def __init__(self, df: pd.DataFrame, llm_model: str | None = None):
        self.df = df
        self.llm_model = llm_model or _hyde_model_name()
        self._client = _ollama_client()
        self.bm25 = None

    def _ensure_bm25_index(self):
        if self.bm25 is not None:
            return
        corpus_texts = [
            f"{row.get('title', '')} {row.get('abstract', row.get('content', ''))}".lower()
            for _, row in self.df.iterrows()
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
            res = self._client.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.7, "num_predict": 150},
            )
            return res["message"]["content"].strip()
        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}")
            return ""

    def expand_query(self, query: str) -> str:
        """Build HyDE-augmented query: original + hypothetical abstract."""
        hypothetical_doc = self._generate_hypothetical_document(query)
        if hypothetical_doc:
            return f"{query}\n{hypothetical_doc}"
        return query

    def retrieve(self, query: str, top_k: int = 5) -> dict:
        self._ensure_bm25_index()
        t0 = time.time()
        hyde_query = self.expand_query(query)
        t_hyde = (time.time() - t0) * 1000

        t1 = time.time()
        tokenized_query = hyde_query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        t_search = (time.time() - t1) * 1000

        docs = []
        for idx in top_indices:
            row = self.df.iloc[idx]
            docs.append(
                {
                    "id": int(idx),
                    "doc_id": str(row.get("article_id", idx)),
                    "score": float(scores[idx]),
                    "title": row.get("title", ""),
                    "abstract": row.get("abstract", ""),
                    "content": row.get("content", ""),
                    "text": row.get("abstract", row.get("content", "")),
                }
            )

        return {
            "hyde_query": hyde_query,
            "docs": docs,
            "t_hyde_ms": t_hyde,
            "t_search_ms": t_search,
        }


class DenseRetriever:
    def __init__(
        self,
        model: SentenceTransformer,
        corpus_embeddings: np.ndarray,
        df: pd.DataFrame,
    ):
        self.model = model
        self.corpus_embeddings = torch.from_numpy(corpus_embeddings)
        self.df = df

    def retrieve(
        self, query: str, candidates: list[dict] = None, top_k: int = 200
    ) -> list[dict]:
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        query_embedding = query_embedding.cpu().numpy().reshape(1, -1)

        if candidates:
            candidate_ids = [c["id"] for c in candidates]
            candidate_embeddings = self.corpus_embeddings[candidate_ids]
            similarities = cosine_similarity(
                query_embedding, candidate_embeddings.cpu().numpy()
            )[0]
            top_k_indices = np.argsort(similarities)[-top_k:][::-1]
            top_indices = [candidate_ids[i] for i in top_k_indices]
            top_scores = similarities[top_k_indices]
        else:
            similarities = cosine_similarity(
                query_embedding, self.corpus_embeddings.cpu().numpy()
            )[0]
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            top_scores = similarities[top_indices]

        results = []
        for idx, score in zip(top_indices, top_scores, strict=False):
            results.append(
                {
                    "id": idx,
                    "title": self.df.iloc[idx]["title"],
                    "abstract": self.df.iloc[idx]["abstract"],
                    "content": self.df.iloc[idx]["content"],
                    "score": float(score),
                }
            )
        return results


class LLMSearchService:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            BASE_DIR = "resources/"
            self.scibert_model_path = os.path.join(
                BASE_DIR, "models", "scibert_scivocab_uncased"
            )
            self.keybert_path = os.path.join(BASE_DIR, "models", "keybert")
            self.embeddings_path = os.path.join(
                BASE_DIR, "embeddings", "centinela_genkeywords.npy"
            )
            self.initialize_components()
            self.__class__._initialized = True

    def initialize_components(self):
        """Initialize all required models and components"""
        try:
            import spacy

            if not spacy.util.is_package("en_core_web_sm"):
                spacy.cli.download("en_core_web_sm")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {e}")
            raise

        try:
            # Load models and embeddings first
            self.scibert_model = SentenceTransformer(self.scibert_model_path)
            keybert_model = SentenceTransformer(self.keybert_path)
            self.kw_model = KeyBERT(model=keybert_model)
            self.translator = GoogleTranslator(source="auto", target="en")

            # Load embeddings to know the size we should use
            self.corpus_embeddings = np.load(self.embeddings_path)
            embeddings_size = len(self.corpus_embeddings)
            logger.info(f"Loaded embeddings with size: {embeddings_size}")

            # Get corpus limited to embeddings size
            self.df = self.get_corpus_from_neo4j(limit=embeddings_size)
            logger.info(f"Retrieved {len(self.df)} documents from Neo4j")

            # Initialize processors
            self.query_processor = QueryProcessor(
                self.kw_model, self.translator, self.scibert_model
            )

        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise

    def get_corpus_from_neo4j(self, limit: int = None) -> pd.DataFrame:
        try:
            # Modified query to include LIMIT
            query = """
            MATCH (a:Article)
            OPTIONAL MATCH (a)-[:WROTE]-(au:Author)
            OPTIONAL MATCH (a)-[:BELONGS_TO]->(af:Affiliation)
            OPTIONAL MATCH (a)-[:USES]->(t:Topic)
            RETURN a.scopus_id as article_id,
                a.title as title,
                a.abstract as abstract,
                a.publication_date as publication_date,
                count(DISTINCT au) as author_count,
                count(DISTINCT af) as affiliation_count,
                collect(DISTINCT au.auth_name) as authors,
                collect(DISTINCT af.name) as affiliations,
                collect(DISTINCT t.name) as topics
            """
            if limit:
                query += f" LIMIT {limit}"

            results, meta = db.cypher_query(query)
            df = pd.DataFrame(
                [
                    {
                        "article_id": row[0],
                        "title": row[1] or "",
                        "abstract": row[2] or "",
                        "publication_date": row[3] or "",
                        "author_count": row[4],
                        "affiliation_count": row[5],
                        "authors": row[6],
                        "affiliations": row[7],
                        "topics": row[8],
                        "content": f"{row[1] or ''} {row[2] or ''} {' '.join(row[8])}",
                    }
                    for row in results
                ]
            )

            return df

        except Exception as e:
            logger.error(f"Error retrieving corpus from Neo4j: {e}")
            raise

    def search(self, query: str, top_k: int = 20, use_hyde: bool = False) -> list[dict]:
        """Perform semantic search using BM25 + Dense Retrieval.

        When ``use_hyde`` is True, expands the query with a HyDE hypothetical
        abstract (Ollama) before the existing BM25 → Dense pipeline.
        When False, keeps the current behavior unchanged.
        """
        try:
            # Process query
            enhanced_query, translation_time, keybert_time, keywords = (
                self.query_processor.process_query(query)
            )

            # Use the cached DataFrame that matches embeddings size
            df = self.df

            retrieval_query = enhanced_query
            if use_hyde:
                hyde_retriever = HydeBM25Retriever(df)
                retrieval_query = hyde_retriever.expand_query(enhanced_query)
                logger.info("HyDE enabled; using expanded query for retrieval")

            # Initialize retrievers
            bm25_retriever = BM25Retriever(df)
            dense_retriever = DenseRetriever(
                self.scibert_model, self.corpus_embeddings, df
            )

            # Perform retrievals concurrently
            with ThreadPoolExecutor() as executor:
                bm25_future = executor.submit(
                    bm25_retriever.retrieve, retrieval_query, top_k=500
                )
                bm25_results = bm25_future.result()

                dense_future = executor.submit(
                    dense_retriever.retrieve,
                    query=retrieval_query,
                    candidates=bm25_results,
                    top_k=top_k,
                )
                final_results = dense_future.result()

            # Format results
            formatted_results = []
            for i, result in enumerate(final_results, 1):
                article_data = df.iloc[result["id"]]
                formatted_results.append(
                    {
                        "rank": i,
                        "title": article_data["title"],
                        "abstract": article_data["abstract"],
                        "publication_date": article_data["publication_date"],
                        "author_count": article_data["author_count"],
                        "affiliation_count": article_data["affiliation_count"],
                        "authors": article_data["authors"],
                        "affiliations": article_data["affiliations"],
                        "relevance_score": float(result["score"]),
                        "article_id": str(article_data["article_id"]),
                    }
                )

            return formatted_results

        except Exception as e:
            logger.error(f"Search error: {e}")
            raise
