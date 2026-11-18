import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from fastapi.testclient import TestClient
import httpx

from doclens.api import create_app
from doclens.corpus import ROOT, Corpus, markdown_blocks, rst_blocks
from doclens.evaluation import cases_from, evidence_map, metrics
from doclens.generation import Generator
from doclens.retrieval import MODES
from doclens.settings import Settings


class CorpusTests(TestCase):
    def test_frozen_inputs_and_stable_citations(self):
        corpus = Corpus()
        self.assertGreaterEqual(len(corpus.sources), 30)
        self.assertGreater(corpus.verify(), len(corpus.sources))
        first = corpus.chunks()
        second = corpus.chunks()
        self.assertEqual(first, second)
        self.assertEqual(len(first), len({chunk.id for chunk in first}))
        for chunk in first:
            self.assertGreaterEqual(chunk.end, chunk.start)
            self.assertIn(f"#L{chunk.start}-L{chunk.end}", chunk.url)
            self.assertIn(chunk.source.collection.revision, chunk.url)

    def test_markup_cleaning_and_example_preservation(self):
        blocks = list(markdown_blocks(
            "---\ntitle: Test\n---\n## Usage\nSee [the API](./api).\n"
            "{* ../../docs_src/demo.py *}\n```python\nif x < y:\n    print(x)\n```\n",
            {"docs_src/demo.py": "def example():\n    return 42"},
        ))
        self.assertEqual([block.start for block in blocks], [4, 5, 6, 7])
        self.assertEqual(blocks[1].text, "See the API.")
        self.assertIn("    return 42", blocks[2].text)
        self.assertIn("x < y", blocks[3].text)
        rst = list(rst_blocks("Tools\n=====\n\nRun it::\n\n    >>> help()\n"))
        self.assertEqual(rst[0].text, "Tools")
        self.assertEqual(rst[-1].language, "python")
        self.assertEqual(rst[-1].text, ">>> help()")

    def test_checksum_rejects_corrupt_file(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_bytes((ROOT / "manifest.json").read_bytes())
            corpus = Corpus(root)
            _, _, local, _ = next(corpus.assets())
            local.parent.mkdir()
            local.write_text("unexpected bytes")
            with self.assertRaisesRegex(ValueError, "Source checksum mismatch"):
                corpus.verify()

    def test_benchmark_references_resolve_in_both_chunkers(self):
        cases = cases_from(ROOT / "evaluation.json")
        self.assertEqual(len(cases), 60)
        for strategy in ("heading", "recursive"):
            with self.subTest(strategy=strategy):
                targets = evidence_map(cases, Corpus().chunks(strategy))
                self.assertEqual(len(targets), len(cases))
        self.assertEqual(metrics(["a", "a", "b"], [{"a"}, {"b"}])[:2], (1, 1))
        self.assertLess(metrics(["a"], [{"a"}, {"b"}])[2], 1)


class ApiTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app(Settings())
        cls.client = cls.enterClassContext(TestClient(cls.app))

    def test_home_health_and_source_inventory(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("DocLens", response.text)
        self.assertEqual(self.client.get("/static/app.js").status_code, 200)
        health = self.client.get("/health").json()
        sources = self.client.get("/sources").json()
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["documents"], len(sources["documents"]))
        self.assertGreater(health["passages"], 500)
        self.assertTrue(all(source["license"] for source in sources["documents"]))

    def test_retrievers_return_sorted_candidates_with_filters(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                response = self.client.post("/search", json={
                    "question": "HTTPException status code",
                    "topic": "fastapi", "collection": "fastapi",
                    "version": "0.115.12", "mode": mode, "top_k": 3,
                })
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertEqual(data["mode"], mode)
                self.assertEqual(len(data["evidence"]), 3)
                self.assertEqual([hit["number"] for hit in data["evidence"]], [1, 2, 3])
                self.assertTrue(all(hit["topic"] == "fastapi" for hit in data["evidence"]))
                self.assertTrue(any(hit["source_id"] == "fastapi-http-errors" for hit in data["evidence"]))
        missing = self.client.post("/search", json={"question": "testing", "version": "0.0.1"})
        self.assertEqual(missing.json()["evidence"], [])

    def test_ask_links_answer_to_real_evidence(self):
        response = self.client.post("/ask", json={"question": "How do FastAPI dependencies work?"})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertFalse(result["abstained"])
        self.assertEqual(result["generation"], "extractive")
        self.assertIn("inject", result["answer"].lower())
        self.assertTrue(result["citations"])
        passages = {hit["number"]: hit for hit in result["evidence"]}
        for citation in result["citations"]:
            self.assertIn(f"[{citation['number']}]", result["answer"])
            self.assertEqual(citation["url"], passages[citation["number"]]["url"])
            self.assertTrue(citation["url"].startswith("https://github.com/"))

    def test_cross_topic_and_abstention(self):
        cross = self.client.post("/ask", json={
            "question": "How can I build a FastAPI endpoint that performs semantic search using Sentence Transformers?"
        }).json()
        self.assertFalse(cross["abstained"])
        self.assertEqual({"fastapi", "rag"}, {cite["topic"] for cite in cross["citations"]})
        result = self.client.post("/ask", json={"question": "Will it rain in Paris tomorrow?"}).json()
        self.assertTrue(result["abstained"])
        self.assertEqual(result["citations"], [])

    def test_validation(self):
        for body in ({"question": "   "}, {"question": "docs", "mode": "unknown"},
                     {"question": "docs", "top_k": 20}, {"question": "x" * 1001}):
            self.assertEqual(self.client.post("/ask", json=body).status_code, 422)

    def test_provider_contract_and_fallback(self):
        question = "How do FastAPI dependencies work?"
        index = self.app.state.index
        hits = index.search(question)
        for provider in ("openai", "ollama"):
            with self.subTest(provider=provider):
                generator = Generator(Settings(provider=provider))
                message = {"content": json.dumps({"answer": "FastAPI injects dependency results. [1]",
                                                  "abstained": False})}
                payload = {"choices": [{"message": message}]} if provider == "openai" else {"message": message}
                response = httpx.Response(200, json=payload, request=httpx.Request("POST", "https://test.invalid"))
                with patch("doclens.generation.httpx.post", return_value=response) as post:
                    answer = generator.answer(question, hits, index)
                self.assertEqual(answer.numbers, (1,))
                self.assertEqual(answer.mode, provider)
                prompt = json.loads(post.call_args.kwargs["json"]["messages"][1]["content"])
                self.assertEqual(prompt["question"], question)
                self.assertEqual(prompt["evidence"][0]["text"], hits[0].chunk.text)
                self.assertEqual(prompt["evidence"][0]["number"], 1)
                message["content"] = json.dumps({"answer": "Unsupported claim. [999]", "abstained": False})
                with patch("doclens.generation.httpx.post", return_value=httpx.Response(
                    200, json=payload, request=httpx.Request("POST", "https://test.invalid")
                )):
                    fallback = generator.answer(question, hits, index)
                self.assertEqual(fallback.mode, "extractive")
                self.assertTrue(fallback.note)
                self.assertNotIn("Unsupported claim", fallback.text)
