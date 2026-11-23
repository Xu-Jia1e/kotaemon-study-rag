from __future__ import annotations

import json

import pytest

from evaluation.adapters import a_index
from evaluation.adapters.a_index import (
    REPO_ROOT,
    _chroma_collections,
    load_index,
    normalize_manifest,
)

A_MANIFEST = {
    "complete": True,
    "reader_mode": "default",
    "chunk_size": 512,
    "chunk_overlap": 128,
    "embedding_model": "text-embedding-v4",
    "dimensions": 2048,
    "collection": "index_1",
    "files": {"11-hash2.pdf": {"source_id": "s1", "error": None}},
}

D_MANIFEST = {
    "format": 1,
    "chunk_size": 1024,
    "chunk_overlap": 256,
    "embedding": {"model": "text-embedding-v4", "dimensions": 2048},
    "files": [{"name": "11-hash2.pdf", "sha256": "abc", "bytes": 1}],
    "vector_count": 106,
    "indexed_documents": 158,
}


def _write_manifest(index_dir, manifest):
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "index.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_normalize_a_manifest():
    normalized = normalize_manifest(A_MANIFEST)
    assert normalized["schema"] == "a"
    assert normalized["chunk_size"] == 512
    assert normalized["chunk_overlap"] == 128
    assert normalized["embedding_model"] == "text-embedding-v4"
    assert normalized["embedding_dimensions"] == 2048
    assert normalized["vector_collection"] == "index_1"
    assert normalized["doc_collection"] == "index_1"
    assert normalized["files"][0]["source_id"] == "s1"
    assert normalized["complete"] is True


def test_normalize_d_manifest():
    normalized = normalize_manifest(D_MANIFEST)
    assert normalized["schema"] == "d"
    assert normalized["embedding_model"] == "text-embedding-v4"
    assert normalized["embedding_dimensions"] == 2048
    assert normalized["vector_collection"] == "default"
    assert normalized["doc_collection"] == "docstore"
    assert normalized["files"][0]["sha256"] == "abc"
    assert normalized["vector_count_expected"] == 106


def test_real_1024_index_still_loads():
    index_dir = REPO_ROOT / ".eval-index" / "1024-256"
    if not index_dir.is_dir():
        pytest.skip("local 1024/256 index is not present")
    bundle = load_index(index_dir)
    assert bundle.vector_store.count() == 106
    assert bundle.normalized["schema"] == "d"
    assert bundle.normalized["chunk_size"] == 1024
    assert bundle.normalized["chunk_overlap"] == 256
    assert bundle.normalized["embedding_model"] == "text-embedding-v4"
    assert bundle.normalized["embedding_dimensions"] == 2048


def test_missing_collection_raises_without_creating_it(tmp_path):
    _write_manifest(tmp_path, A_MANIFEST)
    with pytest.raises(RuntimeError, match="not found"):
        load_index(tmp_path)
    assert not (tmp_path / "vectors" / "chroma.sqlite3").exists()


def test_empty_collection_is_rejected(tmp_path, monkeypatch):
    _write_manifest(tmp_path, A_MANIFEST)

    class FakeVectorStore:
        def count(self):
            return 0

    monkeypatch.setattr(a_index, "_chroma_collections", lambda path: ["index_1"])
    monkeypatch.setattr(a_index, "_define_models", lambda: (object, object))
    monkeypatch.setattr(
        a_index,
        "_new_stores",
        lambda path, **kwargs: (FakeVectorStore(), object()),
    )
    with pytest.raises(RuntimeError, match="empty Chroma collection"):
        load_index(tmp_path)


def test_a_manifest_opens_index_1_for_both_stores(tmp_path, monkeypatch):
    _write_manifest(tmp_path, A_MANIFEST)
    calls = {}

    class FakeVectorStore:
        def count(self):
            return 5

    monkeypatch.setattr(a_index, "_chroma_collections", lambda path: ["index_1"])
    monkeypatch.setattr(a_index, "_define_models", lambda: (object, object))

    def fake_stores(path, *, vector_collection, doc_collection):
        calls["vector"] = vector_collection
        calls["doc"] = doc_collection
        return FakeVectorStore(), object()

    monkeypatch.setattr(a_index, "_new_stores", fake_stores)
    monkeypatch.setattr(a_index, "_lance_count", lambda path, name: 10)
    bundle = load_index(tmp_path)
    assert calls == {"vector": "index_1", "doc": "index_1"}
    assert bundle.normalized["schema"] == "a"


def test_vector_count_mismatch_raises(tmp_path, monkeypatch):
    _write_manifest(tmp_path, {**A_MANIFEST, "vector_count": 99})

    class FakeVectorStore:
        def count(self):
            return 5

    monkeypatch.setattr(a_index, "_chroma_collections", lambda path: ["index_1"])
    monkeypatch.setattr(a_index, "_define_models", lambda: (object, object))
    monkeypatch.setattr(
        a_index,
        "_new_stores",
        lambda path, **kwargs: (FakeVectorStore(), object()),
    )
    with pytest.raises(RuntimeError, match="vector count mismatch"):
        load_index(tmp_path)


def test_chroma_collection_listing_is_read_only(tmp_path):
    assert _chroma_collections(tmp_path) == []
    assert not (tmp_path / "chroma.sqlite3").exists()
