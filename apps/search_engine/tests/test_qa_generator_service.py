from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase

from apps.search_engine.application.services.qa_generator_service import (
    EXPECTED_QA_PAIRS,
    AcademicQAGenerator,
    ContextCompressor,
    QAGenerationPipeline,
    QAGenerationResult,
    normalize_docs_for_qa,
    parse_qa_pairs,
)

VALID_XML = (
    "<pregunta>Q1</pregunta><respuesta>A1</respuesta>"
    "<pregunta>Q2</pregunta><respuesta>A2</respuesta>"
    "<pregunta>Q3</pregunta><respuesta>A3</respuesta>"
)

PARTIAL_ONE_VALID_TWO_EMPTY = (
    "<pregunta>Q1 buena</pregunta><respuesta>A1 buena</respuesta>"
    "<pregunta></pregunta><respuesta>sin pregunta</respuesta>"
    "<pregunta>sin respuesta</pregunta><respuesta></respuesta>"
)

TWO_FIXED = (
    "<pregunta>Q2 fija</pregunta><respuesta>A2 fija</respuesta>"
    "<pregunta>Q3 fija</pregunta><respuesta>A3 fija</respuesta>"
)


def _chat_response(content: str) -> dict:
    return {"message": {"content": content}}


class ParseQaPairsTests(SimpleTestCase):
    def test_parse_empty_returns_empty_list(self):
        self.assertEqual(parse_qa_pairs(""), [])
        self.assertEqual(parse_qa_pairs(None), [])

    def test_parse_extracts_three_pairs(self):
        pairs = parse_qa_pairs(VALID_XML)
        self.assertEqual(len(pairs), 3)
        self.assertEqual(pairs[0]["question"], "Q1")
        self.assertEqual(pairs[0]["answer"], "A1")
        self.assertEqual(pairs[2]["question"], "Q3")

    def test_parse_strips_whitespace(self):
        raw = "<pregunta>  Q  </pregunta><respuesta>\nA\n</respuesta>"
        pairs = parse_qa_pairs(raw)
        self.assertEqual(pairs[0]["question"], "Q")
        self.assertEqual(pairs[0]["answer"], "A")

    def test_parse_case_insensitive_tags(self):
        raw = "<PREGUNTA>Q</PREGUNTA><RESPUESTA>A</RESPUESTA>"
        pairs = parse_qa_pairs(raw)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["question"], "Q")


class NormalizeDocsForQaTests(SimpleTestCase):
    def test_maps_scopus_id_and_limits(self):
        docs = [
            {"scopus_id": "1", "title": "T1", "abstract": "a1"},
            {"scopus_id": "2", "title": "T2", "abstract": "a2"},
            {"scopus_id": "3", "title": "T3", "abstract": "a3"},
            {"scopus_id": "4", "title": "T4", "abstract": "a4"},
        ]
        normalized = normalize_docs_for_qa(docs, limit=3)
        self.assertEqual(len(normalized), 3)
        self.assertEqual(normalized[0]["article_id"], "1")
        self.assertEqual(normalized[0]["title"], "T1")

    def test_falls_back_to_text_and_content(self):
        docs = [
            {"article_id": 99, "title": "T", "text": "from text"},
            {"doc_id": "d1", "title": "T2", "content": "from content"},
        ]
        normalized = normalize_docs_for_qa(docs, limit=2)
        self.assertEqual(normalized[0]["article_id"], "99")
        self.assertEqual(normalized[0]["abstract"], "from text")
        self.assertEqual(normalized[1]["article_id"], "d1")
        self.assertEqual(normalized[1]["abstract"], "from content")

    def test_empty_fields_become_empty_strings(self):
        normalized = normalize_docs_for_qa([{"id": "x"}], limit=1)
        self.assertEqual(normalized[0]["article_id"], "x")
        self.assertEqual(normalized[0]["title"], "")
        self.assertEqual(normalized[0]["abstract"], "")


class ContextCompressorTests(SimpleTestCase):
    def test_compress_uses_ollama_and_sets_text(self):
        compressor = ContextCompressor(slm_model="fake-model")
        compressor._client = MagicMock()
        compressor._client.chat.return_value = _chat_response("resumen corto")
        docs = [{"article_id": "1", "abstract": "long abstract about AI"}]

        compressed, latency_ms = compressor.compress("ai", docs)

        self.assertEqual(len(compressed), 1)
        self.assertEqual(compressed[0]["text"], "resumen corto")
        self.assertGreaterEqual(latency_ms, 0)
        compressor._client.chat.assert_called_once()

    def test_compress_reads_text_or_content_fields(self):
        compressor = ContextCompressor(slm_model="fake-model")
        compressor._client = MagicMock()
        compressor._client.chat.return_value = _chat_response("ok")
        docs = [
            {"text": "from text field"},
            {"content": "from content field"},
        ]

        compressed, _ = compressor.compress("q", docs)

        self.assertEqual(len(compressed), 2)
        calls = compressor._client.chat.call_args_list
        self.assertIn("from text field", calls[0].kwargs["messages"][0]["content"])
        self.assertIn("from content field", calls[1].kwargs["messages"][0]["content"])

    def test_compress_falls_back_to_original_on_error(self):
        compressor = ContextCompressor(slm_model="fake-model")
        compressor._client = MagicMock()
        compressor._client.chat.side_effect = RuntimeError("ollama down")
        docs = [{"abstract": "original abstract"}]

        compressed, _ = compressor.compress("q", docs)

        self.assertEqual(compressed[0]["text"], "original abstract")


class AcademicQAGeneratorDiagnosticTests(SimpleTestCase):
    def setUp(self):
        self.gen = AcademicQAGenerator(slm_model="fake-model")
        self.gen._client = MagicMock()

    def test_evaluar_valid_xml(self):
        valid, failed, msg, _ = self.gen._evaluar_respuesta(VALID_XML)
        self.assertEqual(len(valid), 3)
        self.assertEqual(failed, [])
        self.assertEqual(msg, "")

    def test_evaluar_missing_tags(self):
        valid, failed, msg, _ = self.gen._evaluar_respuesta("solo texto plano sin xml")
        self.assertEqual(valid, [])
        self.assertEqual(failed, [])
        self.assertIn("etiquetas XML", msg)

    def test_evaluar_keeps_valid_and_marks_failed(self):
        valid, failed, msg, _ = self.gen._evaluar_respuesta(PARTIAL_ONE_VALID_TWO_EMPTY)
        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0]["question"], "Q1 buena")
        self.assertEqual(len(failed), 2)
        self.assertIn("inválidos", msg)

    def test_diagnosticar_complete_when_three_valid(self):
        ok, msg, _ = self.gen._diagnosticar_fallo(VALID_XML)
        self.assertTrue(ok)
        self.assertEqual(msg, "")

    def test_diagnosticar_incomplete_when_partial(self):
        ok, msg, _ = self.gen._diagnosticar_fallo(PARTIAL_ONE_VALID_TWO_EMPTY)
        self.assertFalse(ok)
        self.assertTrue(msg)

    def test_reparar_etiquetas_normalizes_case_and_spaces(self):
        raw = (
            "< Pregunta >Q1</ Pregunta >"
            "< RESPUESTA >A1</RESPUESTA>"
            "<pregunta>Q2</pregunta><respuesta>A2</respuesta>"
            "<pregunta>Q3</pregunta><respuesta>A3</respuesta>"
        )
        repaired = self.gen._reparar_etiquetas(raw)
        valid, failed, _, texto = self.gen._evaluar_respuesta(repaired)
        self.assertEqual(len(valid), 3)
        self.assertEqual(failed, [])
        self.assertIn("<pregunta>", texto)

    def test_format_context_uses_doc_fields(self):
        docs = [
            {
                "doc_id": "42",
                "title": "Paper",
                "text": "Abstract text",
            }
        ]
        formatted = self.gen._format_context(docs)
        self.assertIn("Doc 42", formatted)
        self.assertIn("Paper", formatted)
        self.assertIn("Abstract text", formatted)


class AcademicQAGeneratorGenerateTests(SimpleTestCase):
    def setUp(self):
        self.gen = AcademicQAGenerator(slm_model="fake-model")
        self.gen._client = MagicMock()
        self.docs = [{"article_id": "1", "title": "T", "abstract": "abs"}]

    def test_generate_success_first_attempt(self):
        self.gen._client.chat.return_value = _chat_response(VALID_XML)
        initial, final, intentos, latency, status, pairs = self.gen.generate(
            "query", self.docs, use_iterative=True
        )
        self.assertEqual(status, "success")
        self.assertEqual(intentos, 1)
        self.assertEqual(len(pairs), EXPECTED_QA_PAIRS)
        self.assertIn("<pregunta>Q1</pregunta>", final)
        self.assertEqual(initial, VALID_XML)
        self.assertGreaterEqual(latency, 0)
        self.gen._client.chat.assert_called_once()

    def test_generate_strips_markdown_fence(self):
        fenced = f"```xml\n{VALID_XML}\n```"
        self.gen._client.chat.return_value = _chat_response(fenced)
        _, final, intentos, _, status, pairs = self.gen.generate(
            "query", self.docs, use_iterative=False
        )
        self.assertEqual(status, "success")
        self.assertEqual(intentos, 1)
        self.assertEqual(len(pairs), 3)
        self.assertNotIn("```", final)
        self.assertIn("<pregunta>", final)

    def test_generate_retries_when_iterative(self):
        bad = "<pregunta>only one</pregunta><respuesta>a</respuesta>"
        self.gen._client.chat.side_effect = [
            _chat_response(bad),
            _chat_response(
                "<pregunta>Q2</pregunta><respuesta>A2</respuesta>"
                "<pregunta>Q3</pregunta><respuesta>A3</respuesta>"
            ),
        ]
        initial, final, intentos, _, status, pairs = self.gen.generate(
            "query", self.docs, use_iterative=True
        )
        self.assertEqual(status, "success")
        self.assertEqual(intentos, 2)
        self.assertEqual(initial, bad)
        self.assertEqual(len(pairs), 3)
        self.assertEqual(pairs[0]["question"], "only one")
        self.assertEqual(self.gen._client.chat.call_count, 2)
        # Second call asks to rewrite the missing ones only
        second_user = self.gen._client.chat.call_args_list[1].kwargs["messages"][1][
            "content"
        ]
        self.assertIn("exactamente 2", second_user)

    def test_generate_keeps_valid_and_retries_only_failed(self):
        self.gen._client.chat.side_effect = [
            _chat_response(PARTIAL_ONE_VALID_TWO_EMPTY),
            _chat_response(TWO_FIXED),
        ]
        _, final, intentos, _, status, pairs = self.gen.generate(
            "query", self.docs, use_iterative=True
        )
        self.assertEqual(status, "success")
        self.assertEqual(intentos, 2)
        self.assertEqual(len(pairs), 3)
        self.assertEqual(pairs[0]["question"], "Q1 buena")
        self.assertEqual(pairs[1]["question"], "Q2 fija")
        self.assertEqual(pairs[2]["question"], "Q3 fija")
        self.assertIn("Q1 buena", final)
        correction_prompt = self.gen._client.chat.call_args_list[1].kwargs["messages"][
            1
        ]["content"]
        self.assertIn("pares fallidos", correction_prompt)
        self.assertIn("sin pregunta", correction_prompt)
        self.assertIn("sin respuesta", correction_prompt)

    def test_generate_returns_partial_when_retries_exhausted(self):
        self.gen._client.chat.side_effect = [
            _chat_response(PARTIAL_ONE_VALID_TWO_EMPTY),
            _chat_response("<pregunta></pregunta><respuesta>x</respuesta>"),
            _chat_response("texto basura"),
        ]
        _, final, intentos, _, status, pairs = self.gen.generate(
            "query", self.docs, use_iterative=True
        )
        self.assertEqual(intentos, 3)
        self.assertEqual(len(pairs), 1)
        self.assertTrue(status.startswith("partial:"))
        self.assertIn("Q1 buena", final)

    def test_generate_stops_after_one_attempt_without_iterative(self):
        bad = "texto invalido"
        self.gen._client.chat.return_value = _chat_response(bad)
        _, final, intentos, _, status, pairs = self.gen.generate(
            "query", self.docs, use_iterative=False
        )
        self.assertEqual(intentos, 1)
        self.assertEqual(pairs, [])
        self.assertTrue(status.startswith("error:"))
        self.gen._client.chat.assert_called_once()

    def test_generate_partial_without_iterative_keeps_valid(self):
        self.gen._client.chat.return_value = _chat_response(PARTIAL_ONE_VALID_TWO_EMPTY)
        _, _, intentos, _, status, pairs = self.gen.generate(
            "query", self.docs, use_iterative=False
        )
        self.assertEqual(intentos, 1)
        self.assertEqual(len(pairs), 1)
        self.assertTrue(status.startswith("partial:"))

    def test_generate_handles_client_exception(self):
        self.gen._client.chat.side_effect = RuntimeError("connection refused")
        initial, final, intentos, _, status, pairs = self.gen.generate(
            "query", self.docs, use_iterative=True
        )
        self.assertEqual(initial, "")
        self.assertEqual(pairs, [])
        self.assertTrue(status.startswith("error:"))
        self.assertIn("connection refused", status)


class QAGenerationPipelineTests(SimpleTestCase):
    def test_process_with_compression(self):
        pipeline = QAGenerationPipeline(model_name="fake-model")
        pipeline.compressor = MagicMock()
        pipeline.generator = MagicMock()
        pipeline.compressor.compress.return_value = (
            [{"text": "compressed"}],
            10.0,
        )
        pairs = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
            {"question": "Q3", "answer": "A3"},
        ]
        pipeline.generator.generate.return_value = (
            VALID_XML,
            VALID_XML,
            1,
            20.0,
            "success",
            pairs,
        )

        result = pipeline.process(
            query="ai",
            retrieved_docs=[{"abstract": "a"}],
            use_compression=True,
            use_iterative=True,
        )

        self.assertIsInstance(result, QAGenerationResult)
        self.assertEqual(result.status, "success")
        self.assertEqual(result.final_raw_response, VALID_XML)
        self.assertEqual(result.qa_pairs, pairs)
        self.assertEqual(result.latency.compression_ms, 10.0)
        self.assertEqual(result.latency.generation_ms, 20.0)
        pipeline.compressor.compress.assert_called_once()
        pipeline.generator.generate.assert_called_once()

    def test_process_skips_compression_when_disabled(self):
        pipeline = QAGenerationPipeline(model_name="fake-model")
        pipeline.compressor = MagicMock()
        pipeline.generator = MagicMock()
        docs = [{"abstract": "raw"}]
        pipeline.generator.generate.return_value = (
            VALID_XML,
            VALID_XML,
            1,
            5.0,
            "success",
            [{"question": "Q1", "answer": "A1"}],
        )

        result = pipeline.process(
            query="ai",
            retrieved_docs=docs,
            use_compression=False,
            use_iterative=False,
        )

        pipeline.compressor.compress.assert_not_called()
        self.assertEqual(result.processed_docs, docs)
        self.assertEqual(result.latency.compression_ms, 0.0)
        self.assertEqual(
            pipeline.generator.generate.call_args.kwargs["use_iterative"], False
        )

    def test_process_propagates_generator_error_status(self):
        pipeline = QAGenerationPipeline(model_name="fake-model")
        pipeline.compressor = MagicMock()
        pipeline.generator = MagicMock()
        pipeline.compressor.compress.return_value = ([], 0.0)
        pipeline.generator.generate.return_value = (
            "",
            "",
            1,
            1.0,
            "error: boom",
            [],
        )

        result = pipeline.process("q", [], use_compression=True)
        self.assertEqual(result.status, "error: boom")
        self.assertEqual(result.final_raw_response, "")
        self.assertEqual(result.qa_pairs, [])

    @patch(
        "apps.search_engine.application.services.qa_generator_service._ollama_client"
    )
    def test_pipeline_init_wires_compressor_and_generator(self, mock_client):
        mock_client.return_value = MagicMock()
        pipeline = QAGenerationPipeline(model_name="custom-model")
        self.assertEqual(pipeline.compressor.slm_model, "custom-model")
        self.assertEqual(pipeline.generator.slm_model, "custom-model")
