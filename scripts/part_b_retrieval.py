"""Build and query a local, page-level PDF vector index.

The index format is independent of the Kotaemon application's own index.
"""

import argparse
import contextlib
import hashlib
import json
import sys
from pathlib import Path

import fitz
from dotenv import dotenv_values
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

from kotaemon.base import Document
from kotaemon.embeddings import OpenAIEmbeddings
from kotaemon.indices import VectorIndexing, VectorRetrieval
from kotaemon.storages import ChromaVectorStore, LanceDBDocumentStore

MODEL = "text-embedding-v4"
DIMENSIONS = 2048
ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MANIFEST = "index.json"
RRF_K = 60
RRF_CANDIDATES = 20


class CheckedDocumentStore(LanceDBDocumentStore):
    """Fail closed if vector and document stores no longer contain the same IDs."""

    def get(self, ids):
        expected = [ids] if isinstance(ids, str) else list(ids)
        docs = super().get(ids)
        if [doc.doc_id for doc in docs] != expected:
            raise RuntimeError("index is inconsistent: retrieved document IDs are missing")
        return docs


def reciprocal_rank_fusion(vector_docs, text_docs, limit, rank_constant=RRF_K):
    """Fuse unique IDs by rank; prefer vector documents for overlapping IDs."""
    entries = {}
    for branch, docs in (("vector", vector_docs), ("text", text_docs)):
        seen = set()
        for rank, doc in enumerate(docs, start=1):
            if doc.doc_id in seen:
                continue
            seen.add(doc.doc_id)
            entry = entries.setdefault(doc.doc_id, {
                "doc": doc, "score": 0.0, "vector_rank": None,
                "text_rank": None, "vector_score": None,
            })
            entry["score"] += 1.0 / (rank_constant + rank)
            entry[f"{branch}_rank"] = rank
            if branch == "vector":
                entry["doc"] = doc
                entry["vector_score"] = doc.score
    return sorted(entries.values(), key=lambda e: (
        -e["score"], e["vector_rank"] or float("inf"), e["doc"].doc_id
    ))[:limit]


def create_embedding(env_file):
    config = dotenv_values(env_file)
    api_key = config.get("DASHSCOPE_API_KEY")
    model = config.get("EMBEDDING_MODEL_NAME") or MODEL
    endpoint = (config.get("EMBEDDING_API_BASE") or ENDPOINT).rstrip("/")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY is missing from the configuration file")
    if model != MODEL or endpoint != ENDPOINT:
        raise ValueError("this index requires text-embedding-v4 at the DashScope endpoint")
    return OpenAIEmbeddings(
        api_key=api_key, base_url=endpoint, model=model, dimensions=DIMENSIONS
    )


def read_pdf(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    docs = []
    with fitz.open(path) as pdf:
        for page in pdf:
            content = page.get_text().strip()
            if content:
                docs.append(Document(
                    text=content,
                    id_=f"{digest}-page-{page.number + 1}",
                    metadata={"file_name": path.name, "file_id": digest,
                              "page_label": str(page.number + 1)},
                ))
    if not docs:
        raise ValueError(f"No extractable text in {path.name}")
    return digest, docs


def build(args):
    paths = [p.resolve() for p in args.pdf]
    if any(not p.is_file() or p.suffix.lower() != ".pdf" for p in paths):
        raise ValueError("each input must be an existing PDF file")
    if len({p.name.casefold() for p in paths}) != len(paths):
        raise ValueError("PDF filenames must be unique within an index")
    # Validate inputs before making an index directory or sending requests.
    embedding = create_embedding(args.env_file)
    source_docs = [read_pdf(p) for p in paths]
    if len({digest for digest, _ in source_docs}) != len(source_docs):
        raise ValueError("identical PDF content cannot be indexed twice")
    docs = [doc for _, pages in source_docs for doc in pages]
    args.index_dir.mkdir(parents=True, exist_ok=False)
    vectors = ChromaVectorStore(path=str(args.index_dir / "vectors"))
    documents = LanceDBDocumentStore(path=str(args.index_dir / "documents"))
    VectorIndexing(vector_store=vectors, doc_store=documents, embedding=embedding)(docs)
    if vectors.count() != len(docs):
        raise RuntimeError("vector count does not match the extracted pages")
    manifest = {
        "format": 1, "model": MODEL, "dimensions": DIMENSIONS,
        "endpoint": ENDPOINT, "chunking": "pdf-extractable-text-by-page",
        "files": [{"name": p.name, "sha256": digest, "pages": len(pages)}
                  for p, (digest, pages) in zip(paths, source_docs)],
        "count": len(docs),
    }
    (args.index_dir / MANIFEST).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"index_dir": str(args.index_dir.resolve()), **manifest}


def query(args):
    path = args.index_dir
    manifest_file = path / MANIFEST
    if not manifest_file.is_file() or not (path / "vectors" / "chroma.sqlite3").is_file():
        raise ValueError("index is missing or incomplete; build a new index")
    if not (path / "documents" / "docstore.lance").is_dir():
        raise ValueError("index document store is missing")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if (manifest.get("format"), manifest.get("model"), manifest.get("dimensions"),
        manifest.get("endpoint")) != (1, MODEL, DIMENSIONS, ENDPOINT):
        raise ValueError("index embedding configuration is incompatible")
    selected = None
    if args.file:
        selected = next((f for f in manifest["files"] if f["name"] == args.file), None)
        if selected is None:
            raise ValueError("requested file is not in the index")
    embedding = create_embedding(args.env_file)
    # get_collection checks existence before the wrapper's get_or_create call.
    import chromadb
    chromadb.PersistentClient(path=str(path / "vectors")).get_collection("default")
    vectors = ChromaVectorStore(path=str(path / "vectors"))
    if vectors.count() != manifest["count"]:
        raise ValueError("index vector count does not match its manifest")
    documents = CheckedDocumentStore(path=str(path / "documents"))
    options = {}
    if selected:
        options["filters"] = MetadataFilters(filters=[
            MetadataFilter(key="file_id", value=selected["sha256"])
        ])
    if args.mode in ("hybrid", "rrf"):
        # VectorRetrieval requires a nonempty scope for the text branch.
        collection_kwargs = {"include": []}
        if selected:
            collection_kwargs["where"] = {"file_id": selected["sha256"]}
        scope = vectors._collection.get(**collection_kwargs)["ids"]
        if not scope:
            raise ValueError("no indexed chunks found for the selected scope")
        options["scope"] = scope
    if args.mode == "rrf":
        candidate_count = max(args.top_k, RRF_CANDIDATES)
        vector_retriever = VectorRetrieval(vector_store=vectors, doc_store=documents,
                                           embedding=embedding, retrieval_mode="vector")
        text_retriever = VectorRetrieval(vector_store=vectors, doc_store=documents,
                                         embedding=embedding, retrieval_mode="text")
        vector_docs = vector_retriever(args.question, top_k=candidate_count, **options)
        text_docs = text_retriever(args.question, top_k=candidate_count, **options)
        fused = reciprocal_rank_fusion(vector_docs, text_docs, args.top_k)
        results = [entry["doc"] for entry in fused]
    else:
        retriever = VectorRetrieval(vector_store=vectors, doc_store=documents,
                                    embedding=embedding, retrieval_mode=args.mode)
        results = retriever(args.question, top_k=args.top_k, **options)
    if selected and any(d.metadata.get("file_id") != selected["sha256"] for d in results):
        raise RuntimeError("retrieval returned a document outside the selected file")
    output = {
        "question": args.question, "model": MODEL, "dimensions": DIMENSIONS,
        "retrieval_mode": args.mode, "file_filter": args.file,
        "results": [{"id": d.doc_id, "file_name": d.metadata.get("file_name"),
                     "page": d.metadata.get("page_label"), "section": d.metadata.get("section"),
                     "score": None if d.score == -1.0 else d.score,
                     "score_type": "text_unscored" if d.score == -1.0 else "vector_store_similarity",
                     "text": d.text} for d in results],
    }
    if args.mode == "rrf":
        output["rrf"] = {"rank_constant": RRF_K, "candidate_limit_per_branch": candidate_count}
        for result, entry in zip(output["results"], fused):
            result.update(score=entry["score"], score_type="rrf_rank_fusion",
                          vector_rank=entry["vector_rank"], text_rank=entry["text_rank"],
                          vector_score=entry["vector_score"])
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "query"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--index-dir", type=Path, required=True)
        cmd.add_argument("--env-file", type=Path, required=True)
    sub.choices["build"].add_argument("--pdf", type=Path, action="append", required=True)
    q = sub.choices["query"]
    q.add_argument("--question", required=True)
    q.add_argument("--top-k", type=int, default=5)
    q.add_argument("--mode", choices=["vector", "hybrid", "rrf"], default="vector")
    q.add_argument("--file", help="restrict retrieval to this indexed filename")
    args = parser.parse_args()
    if not args.env_file.is_file():
        parser.error("configuration file does not exist")
    if args.command == "query" and (args.top_k < 1 or not args.question.strip()):
        parser.error("--top-k must be positive and --question must not be empty")
    try:
        # Kotaemon logs to stdout; keep the JSON output machine-readable.
        with contextlib.redirect_stdout(sys.stderr):
            result = build(args) if args.command == "build" else query(args)
    except (ValueError, FileExistsError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
