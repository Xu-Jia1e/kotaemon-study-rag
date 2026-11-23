"""Build a real evaluation index with A's ``IndexDocumentPipeline``.

Examples::

    # Interface check only: no key, no index, no requests
    python -m evaluation.scripts.build_indexes --check \
        --docs-dir downloads/real-samples

    # Real 1024/256 index (requires DASHSCOPE_API_KEY in --env-file or env)
    python -m evaluation.scripts.build_indexes \
        --docs-dir downloads/real-samples \
        --out-dir .eval-index/1024-256 \
        --chunk-size 1024 --chunk-overlap 256

    # Read an existing index back without any embedding call
    python -m evaluation.scripts.build_indexes --verify-only \
        --out-dir .eval-index/1024-256

One index per invocation: ``theflow`` caches its settings module, so 1024/256 and
512/128 must be built by separate processes (two commands).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

PROFILES = {
    "1024-256": (1024, 256),
    "512-128": (512, 128),
}


def _resolve_documents(args: argparse.Namespace) -> list[Path]:
    from evaluation.adapters.a_index import collect_documents

    if args.docs:
        documents = [Path(item) for item in args.docs]
    elif args.docs_dir:
        documents = collect_documents(args.docs_dir)
    else:
        raise SystemExit("provide --docs or --docs-dir")
    if not documents:
        raise SystemExit("no supported documents found")
    return documents


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", nargs="+", help="explicit document paths")
    parser.add_argument("--docs-dir", help="directory to scan for documents")
    parser.add_argument("--out-dir", help="index directory to create")
    parser.add_argument(
        "--profile", choices=sorted(PROFILES), help="named chunking profile"
    )
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--chunk-overlap", type=int)
    parser.add_argument("--reader-mode", default="default")
    parser.add_argument("--env-file", help="local .env with DASHSCOPE_API_KEY")
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the pipeline wiring only (no embedding, no index)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="read an existing index back and check chunk metadata",
    )
    args = parser.parse_args(argv)

    from evaluation.adapters.a_index import check_wiring, build_index, verify_index

    if args.verify_only:
        if not args.out_dir:
            raise SystemExit("--verify-only requires --out-dir")
        report = verify_index(args.out_dir)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    documents = _resolve_documents(args)

    if args.profile:
        chunk_size, chunk_overlap = PROFILES[args.profile]
    else:
        chunk_size = args.chunk_size or 1024
        chunk_overlap = args.chunk_overlap or 256

    if args.check:
        wiring = check_wiring(
            documents[0],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            reader_mode=args.reader_mode,
        )
        from evaluation.adapters.a_index import DIMENSIONS, MODEL

        wiring["documents"] = [path.name for path in documents]
        wiring["document_count"] = len(documents)
        wiring["embedding_model"] = MODEL
        wiring["embedding_dimensions"] = DIMENSIONS
        print(json.dumps(wiring, ensure_ascii=False, indent=2))
        return 0

    if not args.out_dir:
        raise SystemExit("building an index requires --out-dir")

    try:
        bundle = build_index(
            documents,
            args.out_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            reader_mode=args.reader_mode,
            env_file=args.env_file,
        )
    except RuntimeError as error:
        print(f"BLOCKED: {error}", file=sys.stderr)
        return 2

    from evaluation.adapters.a_index import summarise

    print(json.dumps(summarise(bundle), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
