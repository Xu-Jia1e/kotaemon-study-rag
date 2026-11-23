"""Run a small vector-only smoke test on a public PDF using Kotaemon components.

This creates a temporary, page-level index; it does not use the app's saved index.
The supplied env file must contain DASHSCOPE_API_KEY and may contain
EMBEDDING_MODEL_NAME and EMBEDDING_API_BASE. Never print the credential.
"""

import argparse
import json
import tempfile
from pathlib import Path

import fitz
from dotenv import dotenv_values

from kotaemon.base import Document
from kotaemon.embeddings import OpenAIEmbeddings
from kotaemon.indices import VectorIndexing, VectorRetrieval
from kotaemon.storages import ChromaVectorStore, InMemoryDocumentStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if not args.env_file.is_file() or not args.pdf.is_file():
        parser.error("the env file and PDF must exist")

    config = dotenv_values(args.env_file)
    api_key = config.get("DASHSCOPE_API_KEY")
    if not api_key:
        parser.error("DASHSCOPE_API_KEY is missing from the env file")
    base_url = config.get("EMBEDDING_API_BASE") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model = config.get("EMBEDDING_MODEL_NAME") or "text-embedding-v4"
    if base_url.rstrip("/") != "https://dashscope.aliyuncs.com/compatible-mode/v1":
        parser.error("unexpected embedding endpoint")
    if model != "text-embedding-v4":
        parser.error("this baseline expects text-embedding-v4")

    with fitz.open(args.pdf) as pdf:
        docs = []
        for page in pdf:
            text = page.get_text().strip()
            if text:
                docs.append(Document(text=text, id_=f"page-{page.number + 1}", metadata={
                    "file_name": args.pdf.name, "page_label": str(page.number + 1)
                }))
    if not docs:
        raise SystemExit("No extractable text found in the PDF")

    embedding = OpenAIEmbeddings(model=model, base_url=base_url, api_key=api_key, dimensions=2048)
    with tempfile.TemporaryDirectory(prefix="kotaemon-baseline-", ignore_cleanup_errors=True) as directory:
        vectors = ChromaVectorStore(path=directory)
        documents = InMemoryDocumentStore()
        VectorIndexing(vector_store=vectors, doc_store=documents, embedding=embedding)(docs)
        retriever = VectorRetrieval(vector_store=vectors, doc_store=documents, embedding=embedding, retrieval_mode="vector")
        results = retriever(args.question, top_k=args.top_k)
        print(json.dumps({
            "model": model, "dimensions": 2048, "indexed_pages": len(docs),
            "question": args.question,
            "results": [{"id": d.doc_id, "file_name": d.metadata.get("file_name"),
                         "page": d.metadata.get("page_label"), "score": d.score,
                         "text": d.text} for d in results],
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
