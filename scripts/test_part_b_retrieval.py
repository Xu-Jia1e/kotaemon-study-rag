"""Offline checks for the local Part B index safeguards."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import fitz

import part_b_retrieval as retrieval
import part_b_project_pipeline as project_pipeline


class RetrievalSafetyTests(unittest.TestCase):
    def test_manual_subquery_rrf_deduplicates_and_preserves_source_scores(self):
        first = [{"id": "A", "score": 0.8, "score_type": "vector_store_similarity"},
                 {"id": "B", "score": 0.7, "score_type": "vector_store_similarity"}]
        second = [{"id": "B", "score": 0.6, "score_type": "vector_store_similarity"},
                  {"id": "C", "score": 0.5, "score_type": "vector_store_similarity"}]
        result = project_pipeline.fuse_query_results([first, second], top_k=3)
        self.assertEqual([item["id"] for item in result], ["B", "A", "C"])
        self.assertEqual(len(result[0]["subquery_hits"]), 2)
        self.assertEqual([hit["rank"] for hit in result[0]["subquery_hits"]], [2, 1])
        self.assertEqual([hit["original_score"] for hit in result[0]["subquery_hits"]], [0.7, 0.6])
        self.assertAlmostEqual(result[0]["score"], 1 / 62 + 1 / 61)
        self.assertEqual(result[0]["score_type"], "rrf")

    def test_rrf_combines_ranks_without_adding_raw_scores(self):
        vector = [SimpleNamespace(doc_id="A", score=0.9), SimpleNamespace(doc_id="B", score=0.8)]
        text = [SimpleNamespace(doc_id="B", score=-1), SimpleNamespace(doc_id="C", score=-1)]
        result = retrieval.reciprocal_rank_fusion(vector, text, limit=3)
        self.assertEqual([e["doc"].doc_id for e in result], ["B", "A", "C"])
        self.assertAlmostEqual(result[0]["score"], 1 / 62 + 1 / 61)
        self.assertEqual((result[0]["vector_rank"], result[0]["text_rank"]), (2, 1))
        self.assertEqual(result[0]["vector_score"], 0.8)
        self.assertIs(result[0]["doc"], vector[1])

    def test_rrf_handles_one_empty_branch_and_duplicate_ids(self):
        docs = [SimpleNamespace(doc_id="A", score=0.7), SimpleNamespace(doc_id="A", score=0.7),
                SimpleNamespace(doc_id="B", score=0.6)]
        result = retrieval.reciprocal_rank_fusion(docs, [], limit=5)
        self.assertEqual([e["doc"].doc_id for e in result], ["A", "B"])
        self.assertAlmostEqual(result[0]["score"], 1 / 61)
        self.assertIsNone(result[0]["text_rank"])

    def test_pdf_page_ids_and_metadata_are_stable(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.pdf"
            pdf = fitz.open()
            page = pdf.new_page()
            page.insert_text((72, 72), "A sample passage.")
            pdf.save(path)
            pdf.close()
            digest, docs = retrieval.read_pdf(path)
            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0].doc_id, f"{digest}-page-1")
            self.assertEqual(docs[0].metadata["file_id"], digest)
            self.assertEqual(docs[0].metadata["page_label"], "1")

    def test_build_refuses_to_replace_existing_index(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pdf = root / "source.pdf"
            pdf.touch()
            index = root / "existing"
            index.mkdir()
            sentinel = index / "important.txt"
            sentinel.write_text("keep", encoding="utf-8")
            args = SimpleNamespace(pdf=[pdf], env_file=root / "env", index_dir=index)
            with patch.object(retrieval, "create_embedding"), patch.object(
                retrieval, "read_pdf", return_value=("id", [object()])
            ):
                with self.assertRaises(FileExistsError):
                    retrieval.build(args)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_incomplete_index_fails_before_embedding(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(index_dir=Path(temp), env_file=Path("unused"), file=None)
            with patch.object(retrieval, "create_embedding") as embedding:
                with self.assertRaisesRegex(ValueError, "missing or incomplete"):
                    retrieval.query(args)
                embedding.assert_not_called()

    def test_missing_document_cannot_misalign_scores(self):
        store = object.__new__(retrieval.CheckedDocumentStore)
        doc = SimpleNamespace(doc_id="second")
        with patch.object(retrieval.LanceDBDocumentStore, "get", return_value=[doc]):
            with self.assertRaisesRegex(RuntimeError, "inconsistent"):
                store.get(["first", "second"])


if __name__ == "__main__":
    unittest.main()
