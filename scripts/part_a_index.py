"""Build an isolated study-material index with the default file readers."""

import argparse
import json
import os
import sys
from pathlib import Path

MODEL = "text-embedding-v4"
DIMENSIONS = 2048
BAILIAN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def build_index(args: argparse.Namespace, api_key: str) -> bool:
    index_dir = args.index_dir
    # Configure storage before importing ktem, which opens its database on import.
    os.environ["THEFLOW_SETTINGS_MODULE"] = "theflow.settings.default"
    from theflow.settings import settings

    settings.KH_DATABASE = f"sqlite:///{(index_dir / 'pipeline.sqlite3').as_posix()}"
    settings.KH_FILESTORAGE_PATH = str(index_dir / "files")
    settings.KH_DOCSTORE = {
        "__type__": "kotaemon.storages.LanceDBDocumentStore",
        "path": str(index_dir / "documents"),
    }
    settings.KH_VECTORSTORE = {
        "__type__": "kotaemon.storages.ChromaVectorStore",
        "path": str(index_dir / "vectors"),
    }

    from ktem.index.file.index import FileIndex
    from ktem.index.file.pipelines import IndexDocumentPipeline

    from kotaemon.embeddings import OpenAIEmbeddings

    index = FileIndex(None, 1, "Part A", {})
    index.on_create()
    resources = index._resources
    embedding = OpenAIEmbeddings(
        api_key=api_key,
        base_url=os.environ.get("EMBEDDING_API_BASE") or BAILIAN_BASE_URL,
        model=MODEL,
        dimensions=DIMENSIONS,
        timeout=20,
        max_retries=0,  # OpenAIEmbeddings already retries failed requests.
    )
    pipeline = IndexDocumentPipeline(
        embedding=embedding,
        reader_mode="default",
        run_embedding_in_thread=False,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        Source=resources["Source"],
        Index=resources["Index"],
        VS=resources["VectorStore"],
        DS=resources["DocStore"],
        FSPath=resources["FileStoragePath"],
        user_id="",
    )

    stream = pipeline.stream(args.files)
    while True:
        try:
            message = next(stream)
        except StopIteration as end:
            source_ids, errors, _ = end.value
            break
        if message.channel == "index":
            print(f"{message.content['file_name']}: {message.content['status']}")

    complete = all(source_ids) and not any(errors)
    manifest = {
        "complete": complete,
        "reader_mode": "default",
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "embedding_model": MODEL,
        "dimensions": DIMENSIONS,
        "collection": "index_1",
        "files": {
            path.name: {"source_id": source_id, "error": error}
            for path, source_id, error in zip(args.files, source_ids, errors)
        },
    }
    manifest_path = index_dir / "index.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(manifest_path)
    return complete


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, required=True, help="New directory")
    parser.add_argument("--chunk-size", type=int, required=True, help="In tokens")
    parser.add_argument("--chunk-overlap", type=int, required=True, help="In tokens")
    parser.add_argument("files", type=Path, nargs="+", help="Local files to index")
    args = parser.parse_args()

    if not 0 < args.chunk_overlap < args.chunk_size:
        parser.error("require 0 < --chunk-overlap < --chunk-size")
    args.files = [path.resolve() for path in args.files]
    if any(not path.is_file() for path in args.files):
        parser.error("all inputs must be existing files")
    if len({path.name for path in args.files}) != len(args.files):
        parser.error("input file names must be unique")

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        parser.error("set the DASHSCOPE_API_KEY environment variable")

    args.index_dir = args.index_dir.resolve()
    try:
        args.index_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"index directory already exists: {args.index_dir}")
    return 0 if build_index(args, api_key) else 1


if __name__ == "__main__":
    sys.exit(main())
