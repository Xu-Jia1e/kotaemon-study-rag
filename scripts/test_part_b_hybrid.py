"""Regression test for overlapping hybrid retrieval results (upstream #858)."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openai.types.create_embedding_response import CreateEmbeddingResponse

from kotaemon.base import Document
from kotaemon.embeddings import AzureOpenAIEmbeddings
from kotaemon.indices import VectorRetrieval
from kotaemon.storages import ChromaVectorStore, InMemoryDocumentStore


class HybridOverlapTest(unittest.TestCase):
    def test_overlapping_chunks_appear_once_with_vector_score(self):
        fixture = Path(__file__).resolve().parents[1] / "libs/kotaemon/tests/resources/embedding_openai.json"
        response = CreateEmbeddingResponse.model_validate(json.loads(fixture.read_text()))
        docs = [Document(text=name, id_=name) for name in ("A", "B", "C")]
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            vector_store = ChromaVectorStore(path=temp)
            doc_store = InMemoryDocumentStore()
            doc_store.add(docs)
            embedding = AzureOpenAIEmbeddings(
                azure_deployment="test", azure_endpoint="https://test.openai.azure.com/",
                api_key="test", api_version="test",
            )
            retriever = VectorRetrieval(vector_store=vector_store, doc_store=doc_store,
                                        embedding=embedding, retrieval_mode="hybrid")
            with patch("openai.resources.embeddings.Embeddings.create", return_value=response), \
                 patch.object(ChromaVectorStore, "query", return_value=([], [0.91, 0.73], ["A", "C"])), \
                 patch.object(InMemoryDocumentStore, "query", return_value=docs[:2]):
                result = retriever("query", top_k=10, scope=["A", "B", "C"])
            self.assertEqual([d.doc_id for d in result], ["B", "A", "C"])
            self.assertEqual([d.score for d in result], [-1.0, 0.91, 0.73])

    def test_pdf_thumbnail_conversion_preserves_ranking_and_duplicate_page_chunks(self):
        fixture = Path(__file__).resolve().parents[1] / "libs/kotaemon/tests/resources/embedding_openai.json"
        response = CreateEmbeddingResponse.model_validate(json.loads(fixture.read_text()))
        docs = [Document(text=name, id_=name, metadata={"thumbnail_doc_id": thumbnail})
                for name, thumbnail in (("A", "TA"), ("B", "TB"), ("C", "TA"))]
        thumbnails = [Document(text="Page thumbnail", id_=name, metadata={"type": "thumbnail"})
                      for name in ("TA", "TB")]
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            vector_store = ChromaVectorStore(path=temp)
            doc_store = InMemoryDocumentStore()
            doc_store.add(docs + thumbnails)
            embedding = AzureOpenAIEmbeddings(
                azure_deployment="test", azure_endpoint="https://test.openai.azure.com/",
                api_key="test", api_version="test",
            )
            retriever = VectorRetrieval(vector_store=vector_store, doc_store=doc_store,
                                        embedding=embedding, retrieval_mode="vector")
            with patch("openai.resources.embeddings.Embeddings.create", return_value=response), \
                 patch.object(ChromaVectorStore, "query", return_value=([], [0.9, 0.8, 0.7], ["A", "B", "C"])):
                result = retriever("query", top_k=3)
            self.assertEqual([d.doc_id for d in result], ["A", "B", "C"])
            self.assertEqual([d.score for d in result], [0.9, 0.8, 0.7])
            self.assertEqual([d.content for d in result], ["A", "B", "C"])


if __name__ == "__main__":
    unittest.main()
