"""D-owned adapter around A's ``IndexDocumentPipeline``.

Goal: build and inspect a *real* 1024/256 (or 512/128) index with A's own code,
without modifying ``libs/kotaemon`` or ``libs/ktem``.

How the wiring works
--------------------
``IndexDocumentPipeline`` is a component, not a CLI. ``route()`` builds an
``IndexPipeline`` from ``self.chunk_size`` / ``self.chunk_overlap``, and
``stream()`` writes chunks into the injected vector store, document store and
SQLite tables. The repository already demonstrates this assembly in
``scripts/part_b_project_pipeline.py``; this module provides the same recipe for
evaluation indexes, plus an offline ``verify_index`` that reads an index without
calling any embedding API.

Hard dependency: a DashScope-compatible embedding endpoint + key (Bailian
``text-embedding-v4``, 2048 dims). Without it, ``build_index`` raises before any
request is made. ``check_wiring`` needs no key and creates no index.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]

MODEL = "text-embedding-v4"
DIMENSIONS = 2048
ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MANIFEST = "index.json"
SUPPORTED_SUFFIXES = {".pdf", ".pptx", ".docx", ".md", ".html", ".htm"}


@dataclass
class IndexBundle:
    """Handles to a built index; ``embedding`` is optional for read-only use."""

    index_dir: Path
    vector_store: Any
    doc_store: Any
    source_model: type | None
    index_model: type | None
    manifest: dict[str, Any]
    embedding: Any = None
    normalized: dict[str, Any] | None = None


DEFAULT_VECTOR_COLLECTION = "default"
DEFAULT_DOC_COLLECTION = "docstore"


def _chroma_collections(path: Path) -> list[str]:
    """List Chroma collections without creating or opening the store."""
    database = path / "chroma.sqlite3"
    if not database.is_file():
        return []
    connection = sqlite3.connect(f"file:{database}?mode=ro&immutable=1", uri=True)
    try:
        return [row[0] for row in connection.execute("SELECT name FROM collections")]
    finally:
        connection.close()


def _store_count(store: Any) -> int:
    count = getattr(store, "count", None)
    if not callable(count):
        raise RuntimeError(f"{type(store).__name__} does not expose count()")
    return int(count())


def _lance_count(documents_path: Path, collection: str) -> int:
    """Read-only row count for a LanceDB table (its store.count() is a stub)."""
    table_path = documents_path / f"{collection}.lance"
    if not table_path.is_dir():
        raise RuntimeError(
            f"LanceDB table {collection!r} not found at {documents_path}"
        )
    import lance

    return int(lance.dataset(str(table_path)).count_rows())


def normalize_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical view over D's old manifest and A's new manifest.

    D schema: nested ``embedding`` + list ``files`` + ``vector_count``.
    A schema: flat ``embedding_model``/``dimensions`` + dict ``files`` +
    ``collection`` + ``complete``.
    """
    embedding = manifest.get("embedding") or {}
    model = embedding.get("model") or manifest.get("embedding_model")
    dimensions = embedding.get("dimensions") or manifest.get("dimensions")
    raw_files = manifest.get("files")
    files: list[dict[str, Any]] = []
    if isinstance(raw_files, dict):
        for name in sorted(raw_files):
            info = raw_files.get(name) or {}
            files.append(
                {
                    "name": name,
                    "source_id": info.get("source_id"),
                    "error": info.get("error"),
                }
            )
        schema = "a"
    elif isinstance(raw_files, list):
        for info in raw_files:
            files.append(
                {
                    "name": info.get("name"),
                    "sha256": info.get("sha256"),
                    "bytes": info.get("bytes"),
                }
            )
        schema = "d"
    else:
        schema = "a" if "embedding_model" in manifest else "d"

    collection = manifest.get("collection")
    return {
        "schema": schema,
        "chunk_size": manifest.get("chunk_size"),
        "chunk_overlap": manifest.get("chunk_overlap"),
        "reader_mode": manifest.get("reader_mode"),
        "embedding_model": model,
        "embedding_dimensions": dimensions,
        "vector_collection": collection or DEFAULT_VECTOR_COLLECTION,
        "doc_collection": collection or DEFAULT_DOC_COLLECTION,
        "files": files,
        "complete": manifest.get("complete"),
        "vector_count_expected": manifest.get("vector_count"),
        "documents_expected": manifest.get("indexed_documents"),
        "raw": dict(manifest),
    }


def manifest_embedding(manifest: Mapping[str, Any]) -> dict[str, Any]:
    normalized = normalize_manifest(manifest)
    return {
        "model": normalized["embedding_model"],
        "dimensions": normalized["embedding_dimensions"],
    }


def _ensure_path() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def _prepare_settings(index_dir: Path) -> None:
    """Point ktem's settings at ``index_dir`` before importing ktem.

    Must run in a fresh process per index: ``theflow`` caches the settings module
    and ``ktem.db.engine`` builds its engine once at import.
    """
    _ensure_path()
    index_dir = index_dir.resolve()
    import os

    os.environ["DSH_EVAL_INDEX_DIR"] = str(index_dir)
    os.environ.setdefault("THEFLOW_SETTINGS_MODULE", "evaluation.adapters._settings")


_MODELS: tuple[type, type] | None = None


def _new_id() -> str:
    import uuid

    return uuid.uuid4().hex


def _define_models() -> tuple[type, type]:
    """Define the two tables ``IndexPipeline`` writes (mirrors Part B's harness)."""
    global _MODELS
    if _MODELS is not None:
        return _MODELS

    from sqlalchemy import Column, Integer, JSON, String
    from sqlalchemy.ext.mutable import MutableDict
    from sqlalchemy.orm import declarative_base

    base = declarative_base()

    class Source(base):  # type: ignore[misc, valid-type]
        __tablename__ = "source"

        id = Column(String, primary_key=True, default=_new_id)
        name = Column(String, unique=True)
        path = Column(String)
        size = Column(Integer, default=0)
        user = Column(String, default="")
        note = Column(MutableDict.as_mutable(JSON), default=dict)

    class Index(base):  # type: ignore[misc, valid-type]
        __tablename__ = "index_record"

        id = Column(Integer, primary_key=True, autoincrement=True)
        source_id = Column(String)
        target_id = Column(String)
        relation_type = Column(String)
        user = Column(String, default="")

    _MODELS = (Source, Index)

    from ktem.db.engine import engine

    base.metadata.create_all(engine)
    return _MODELS


def embedding_config(env_file: str | Path | None = None) -> dict[str, Any]:
    """Validate embedding configuration without creating a client or a request."""
    from dotenv import dotenv_values

    values = dotenv_values(env_file) if env_file else {}
    api_key = values.get("DASHSCOPE_API_KEY") or __import__("os").environ.get(
        "DASHSCOPE_API_KEY"
    )
    model = values.get("EMBEDDING_MODEL_NAME") or MODEL
    endpoint = (values.get("EMBEDDING_API_BASE") or ENDPOINT).rstrip("/")
    if not api_key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY is not configured (pass --env-file or set the "
            "environment variable); no index can be built without it"
        )
    if model != MODEL or endpoint != ENDPOINT:
        raise RuntimeError(
            f"this index requires {MODEL} at {ENDPOINT}; got model={model!r} "
            f"endpoint={endpoint!r}"
        )
    return {"api_key": api_key, "model": model, "endpoint": endpoint}


def _embedding_client(config: dict[str, Any]):
    from kotaemon.embeddings import OpenAIEmbeddings

    return OpenAIEmbeddings(
        api_key=config["api_key"],
        base_url=config["endpoint"],
        model=config["model"],
        dimensions=DIMENSIONS,
    )


def create_embedding(env_file: str | Path | None = None):
    """Build the Bailian embedding client from a local file or environment.

    Never logs the key. Raises ``RuntimeError`` when it is absent so callers can
    report a precise blocker instead of attempting a request.
    """
    return _embedding_client(embedding_config(env_file))


def _new_stores(
    index_dir: Path,
    *,
    vector_collection: str = DEFAULT_VECTOR_COLLECTION,
    doc_collection: str = DEFAULT_DOC_COLLECTION,
):
    from kotaemon.storages import ChromaVectorStore, LanceDBDocumentStore

    vector_store = ChromaVectorStore(
        path=str(index_dir / "vectors"), collection_name=vector_collection
    )
    doc_store = LanceDBDocumentStore(
        path=str(index_dir / "documents"), collection_name=doc_collection
    )
    return vector_store, doc_store


def build_index(
    documents: Sequence[str | Path],
    index_dir: str | Path,
    *,
    chunk_size: int = 1024,
    chunk_overlap: int = 256,
    reader_mode: str = "default",
    env_file: str | Path | None = None,
    embedding: Any | None = None,
) -> IndexBundle:
    """Build one real index with A's pipeline. Requires an embedding key."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    index_dir = Path(index_dir).resolve()
    paths = [Path(document).resolve() for document in documents]
    if not paths:
        raise ValueError("at least one document is required")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"document not found: {path}")
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"unsupported document type: {path.suffix} ({path.name})")
    if index_dir.exists() and any(index_dir.iterdir()):
        raise FileExistsError(
            f"index directory already exists and is not empty: {index_dir} "
            "(refusing to overwrite an existing index)"
        )

    if embedding is None:
        config = embedding_config(env_file)  # fail fast before creating anything
    else:
        config = None

    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "files").mkdir(exist_ok=True)
    _prepare_settings(index_dir)
    from ktem.index.file.pipelines import IndexDocumentPipeline

    Source, Index = _define_models()
    vector_store, doc_store = _new_stores(index_dir)
    embedding = embedding or _embedding_client(config)

    pipeline = IndexDocumentPipeline(embedding=embedding, reader_mode=reader_mode)
    pipeline.Source, pipeline.Index = Source, Index
    pipeline.VS, pipeline.DS = vector_store, doc_store
    pipeline.FSPath, pipeline.user_id, pipeline.private = index_dir / "files", "", False
    pipeline.chunk_size, pipeline.chunk_overlap = chunk_size, chunk_overlap

    debug: list[Any] = []
    generator = pipeline.stream(paths)
    try:
        while True:
            debug.append(next(generator))
    except StopIteration as stop:
        file_ids, errors, documents_indexed = stop.value

    failures = [
        {"index": position, "error": error}
        for position, error in enumerate(errors)
        if error is not None
    ]
    if failures:
        raise RuntimeError(f"indexing failed for some documents: {failures}")

    vector_count = vector_store.count() if hasattr(vector_store, "count") else None
    manifest = {
        "format": 1,
        "built_by": "evaluation.adapters.a_index",
        "pipeline": "ktem.index.file.pipelines.IndexDocumentPipeline",
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "reader_mode": reader_mode,
        "embedding": {"model": MODEL, "dimensions": DIMENSIONS, "endpoint": ENDPOINT},
        "files": [
            {
                "name": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
            for path in paths
        ],
        "file_ids": file_ids,
        "vector_count": vector_count,
        "indexed_documents": len(documents_indexed),
    }
    (index_dir / MANIFEST).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return IndexBundle(
        index_dir=index_dir,
        vector_store=vector_store,
        doc_store=doc_store,
        source_model=Source,
        index_model=Index,
        manifest=manifest,
        embedding=embedding,
        normalized=normalize_manifest(manifest),
    )


def load_index(index_dir: str | Path, *, embedding: Any | None = None) -> IndexBundle:
    """Open an existing index for reading (no embedding required to read).

    Refuses to create/open a missing or empty Chroma collection. ``ChromaVectorStore``
    itself uses ``get_or_create_collection``, so existence is checked read-only
    against ``vectors/chroma.sqlite3`` first.
    """
    index_dir = Path(index_dir).resolve()
    manifest_path = index_dir / MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(f"index manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    normalized = normalize_manifest(manifest)
    if normalized["complete"] is False:
        raise RuntimeError("manifest reports an incomplete build (complete=false)")
    if not normalized["embedding_model"] or not normalized["embedding_dimensions"]:
        raise RuntimeError("manifest does not record embedding model/dimensions")

    available = _chroma_collections(index_dir / "vectors")
    collection = normalized["vector_collection"]
    if collection not in available:
        raise RuntimeError(
            f"Chroma collection {collection!r} not found in "
            f"{index_dir / 'vectors'}; available={available}"
        )

    _prepare_settings(index_dir)
    vector_store, doc_store = _new_stores(
        index_dir,
        vector_collection=collection,
        doc_collection=normalized["doc_collection"],
    )
    vector_count = _store_count(vector_store)
    if vector_count == 0:
        raise RuntimeError(
            f"refusing to open empty Chroma collection {collection!r} "
            f"at {index_dir}"
        )
    expected = normalized["vector_count_expected"]
    if expected is not None and int(expected) != vector_count:
        raise RuntimeError(
            f"vector count mismatch: manifest={expected} store={vector_count}"
        )
    try:
        document_count = _lance_count(
            index_dir / "documents", normalized["doc_collection"]
        )
    except Exception as error:  # noqa: BLE001 - surface a precise load failure
        raise RuntimeError(
            f"LanceDB table {normalized['doc_collection']!r} is not readable: {error}"
        ) from error
    if document_count == 0:
        raise RuntimeError(
            f"refusing to open empty LanceDB table "
            f"{normalized['doc_collection']!r} at {index_dir}"
        )
    return IndexBundle(
        index_dir=index_dir,
        vector_store=vector_store,
        doc_store=doc_store,
        source_model=None,
        index_model=None,
        manifest=manifest,
        embedding=embedding,
        normalized=normalized,
    )


def verify_index(index_dir: str | Path, sample: int = 5) -> dict[str, Any]:
    """Read an index back without any embedding call and check its metadata."""
    bundle = load_index(index_dir)
    manifest = bundle.manifest
    normalized = bundle.normalized or normalize_manifest(manifest)
    vector_count = bundle.vector_store.count()
    expected = normalized["vector_count_expected"]
    if expected is not None and vector_count != expected:
        raise RuntimeError(
            f"vector count mismatch: manifest={expected} store={vector_count}"
        )
    document_count = _lance_count(
        bundle.index_dir / "documents", normalized["doc_collection"]
    )
    warnings: list[str] = []
    documents_expected = normalized["documents_expected"]
    if documents_expected is not None and int(documents_expected) != document_count:
        warnings.append(
            f"manifest indexed_documents={documents_expected} but docstore has "
            f"{document_count} rows (thumbnails included)"
        )

    collection = bundle.vector_store._collection
    ids = collection.get(include=[])["ids"][:sample]
    docs = bundle.doc_store.get(ids) if ids else []
    checked = []
    for doc in docs:
        metadata = doc.metadata or {}
        checked.append(
            {
                "chunk_id": doc.doc_id,
                "file_name": metadata.get("file_name"),
                "page_label": metadata.get("page_label"),
                "section": metadata.get("section"),
                "text_head": (doc.text or "")[:60],
                "has_text": bool((doc.text or "").strip()),
            }
        )
    missing = {
        "chunk_id": sum(1 for row in checked if not row["chunk_id"]),
        "file_name": sum(1 for row in checked if not row["file_name"]),
        "page_label": sum(1 for row in checked if row["page_label"] is None),
        "text": sum(1 for row in checked if not row["has_text"]),
    }
    return {
        "index_dir": str(bundle.index_dir),
        "manifest": manifest,
        "normalized": {key: value for key, value in normalized.items() if key != "raw"},
        "vector_count": vector_count,
        "document_count": document_count,
        "manifest_warnings": warnings,
        "sampled": len(checked),
        "missing_fields": missing,
        "sample": checked,
    }


def check_wiring(
    document: str | Path,
    *,
    chunk_size: int = 1024,
    chunk_overlap: int = 256,
    reader_mode: str = "default",
) -> dict[str, Any]:
    """Verify A's pipeline can be assembled by D without an embedding key.

    Creates a throw-away index directory, constructs the pipeline and calls
    ``route()`` only. No embedding request is made, no index is produced, and the
    temporary directory is removed afterwards.
    """
    document = Path(document).resolve()
    if not document.is_file():
        raise FileNotFoundError(document)

    temp = Path(tempfile.mkdtemp(prefix="dsh-wiring-"))
    index_dir = temp
    try:
        _prepare_settings(index_dir)
        from ktem.index.file.pipelines import IndexDocumentPipeline, IndexPipeline

        Source, Index = _define_models()
        vector_store, doc_store = _new_stores(index_dir)
        (index_dir / "files").mkdir(exist_ok=True)

        pipeline = IndexDocumentPipeline(
            embedding=_NullEmbedding(), reader_mode=reader_mode
        )
        pipeline.Source, pipeline.Index = Source, Index
        pipeline.VS, pipeline.DS = vector_store, doc_store
        pipeline.FSPath = index_dir / "files"
        pipeline.user_id, pipeline.private = "", False
        pipeline.chunk_size, pipeline.chunk_overlap = chunk_size, chunk_overlap

        routed = pipeline.route(document)
        assert isinstance(routed, IndexPipeline)
        splitter = routed.splitter
        return {
            "document": str(document),
            "pipeline_class": type(pipeline).__name__,
            "routed_class": type(routed).__name__,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "splitter_chunk_size": getattr(splitter, "chunk_size", None),
            "splitter_chunk_overlap": getattr(splitter, "chunk_overlap", None),
            "splitter_class": type(splitter).__name__,
            "writes": {
                "vector_store_count": vector_store.count(),
                "files_dir_empty": not any((index_dir / "files").iterdir()),
            },
        }
    finally:
        # Chroma/SQLite may still hold handles on Windows; best-effort cleanup.
        shutil.rmtree(index_dir, ignore_errors=True)


class _NullEmbedding:
    """Placeholder used only by ``check_wiring``; never invoked."""

    def __call__(self, *args, **kwargs):  # pragma: no cover - defensive
        raise RuntimeError("check_wiring must not call the embedding model")

    def invoke(self, *args, **kwargs):  # pragma: no cover - defensive
        raise RuntimeError("check_wiring must not call the embedding model")


def collect_documents(docs_dir: str | Path) -> list[Path]:
    """Return supported documents in ``docs_dir``, sorted by name."""
    docs_dir = Path(docs_dir).resolve()
    if not docs_dir.is_dir():
        raise NotADirectoryError(docs_dir)
    return sorted(
        path
        for path in docs_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def summarise(bundle: IndexBundle) -> dict[str, Any]:
    """Compact view for CLI output."""
    normalized = bundle.normalized or normalize_manifest(bundle.manifest)
    return {
        "index_dir": str(bundle.index_dir),
        "schema": normalized["schema"],
        "chunk_size": normalized["chunk_size"],
        "chunk_overlap": normalized["chunk_overlap"],
        "embedding_model": normalized["embedding_model"],
        "embedding_dimensions": normalized["embedding_dimensions"],
        "vector_collection": normalized["vector_collection"],
        "doc_collection": normalized["doc_collection"],
        "files": [entry["name"] for entry in normalized["files"]],
        "vector_count": normalized["vector_count_expected"],
        "indexed_documents": normalized["documents_expected"],
    }


def _iter_manifest_files(bundle: IndexBundle) -> Iterable[str]:
    normalized = bundle.normalized or normalize_manifest(bundle.manifest)
    for entry in normalized["files"]:
        yield str(entry["name"])
