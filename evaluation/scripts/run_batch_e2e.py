"""Resilient real E2E batch runner: retrieval.json -> C answers -> records.

Unlike ``run_answer_e2e.py`` (the 4-case pilot), this runner:

* isolates failures per question (one failure never aborts the batch);
* appends each finished case to ``per_case.jsonl`` with flush+fsync;
* supports ``--resume`` keyed on a configuration fingerprint;
* retries transient 429/5xx errors a bounded number of times with backoff;
* writes a run manifest with hashes/commit/model (never API keys);
* can rebuild all outputs (``answers.json``, ``predictions.jsonl``,
  ``failures.json``, ``report_auto.json``) from ``per_case.jsonl`` alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = REPO_ROOT / "evaluation" / "datasets" / "qa_cases_v2.jsonl"
DEFAULT_RETRIEVAL = Path(
    r"E:/dsh_workplace/.eval-data/retrieval-real/1024-256-vector-k5/retrieval.json"
)
DEFAULT_OUT = Path(r"E:/dsh_workplace/.eval-data/batch-real")
ADAPTER_VERSION = "c-answer-v1"


def _load_cases(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _build_config(args: argparse.Namespace, retrieval: dict, dataset: Path) -> dict:
    from evaluation.adapters.c_answer import DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
    from evaluation.batch import sha256_file

    nested = retrieval.get("config")
    nested = nested if isinstance(nested, dict) else {}

    chunking = nested.get("chunking")
    if not (isinstance(chunking, str) and chunking.strip()):
        legacy_chunking = retrieval.get("chunking")
        if isinstance(legacy_chunking, str) and legacy_chunking.strip():
            chunking = legacy_chunking
        elif isinstance(legacy_chunking, dict):
            size = legacy_chunking.get("chunk_size")
            overlap = legacy_chunking.get("chunk_overlap")
            chunking = f"{size}-{overlap}" if size is not None else "None-None"
        else:
            chunking = "None-None"

    index_sha = nested.get("index_manifest_sha256")
    if not index_sha:
        index_dir = nested.get("index_dir") or retrieval.get("index_dir")
        index_sha = sha256_file(Path(index_dir) / "index.json") if index_dir else None

    legacy_embedding = retrieval.get("embedding")
    legacy_embedding = legacy_embedding if isinstance(legacy_embedding, dict) else {}
    embedding_model = (
        nested.get("embedding_model")
        or retrieval.get("embedding_model")
        or legacy_embedding.get("model")
    )
    embedding_dimensions = (
        nested.get("embedding_dimensions")
        or retrieval.get("embedding_dimensions")
        or legacy_embedding.get("dimensions")
    )

    return {
        "dataset_sha256": sha256_file(dataset),
        "retrieval_sha256": sha256_file(args.retrieval),
        "index_manifest_sha256": index_sha,
        "chunking": chunking,
        "top_k": retrieval.get("top_k"),
        "retrieval_method": retrieval.get("mode", "vector"),
        "llm_model": DEEPSEEK_MODEL,
        "llm_endpoint": DEEPSEEK_BASE_URL,
        "embedding_model": embedding_model,
        "embedding_dimensions": embedding_dimensions,
        "adapter_version": ADAPTER_VERSION,
        "simulated": bool(retrieval.get("simulated")),
    }


def _write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _unused_answerer(question, items):  # pragma: no cover - rebuild-only path
    raise RuntimeError("rebuild-only does not call the answerer")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--env-file", default=str(REPO_ROOT / ".env"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rebuild-only", action="store_true")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--case-ids", help="comma separated subset")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)

    from evaluation.batch import BatchRunner, build_manifest, now_iso

    retrieval = json.loads(args.retrieval.read_text(encoding="utf-8"))
    dataset_cases = {case["id"]: case for case in _load_cases(args.cases)}
    retrieval_cases = {case["case_id"]: case for case in retrieval["cases"]}
    if not args.rebuild_only and retrieval.get("simulated"):
        raise SystemExit("refusing to treat simulated retrieval as real data")

    wanted = list(retrieval_cases)
    if args.case_ids:
        wanted = [item.strip() for item in args.case_ids.split(",") if item.strip()]
        missing = [case_id for case_id in wanted if case_id not in retrieval_cases]
        if missing:
            raise SystemExit(f"case ids missing from retrieval file: {missing}")
    if args.limit:
        wanted = wanted[: args.limit]

    cases = [dataset_cases[case_id] for case_id in wanted if case_id in dataset_cases]
    if len(cases) != len(wanted):
        raise SystemExit("retrieval file references cases outside the dataset")

    config = _build_config(args, retrieval, args.cases)
    runner = BatchRunner(
        cases=cases,
        retrieval_cases={case_id: retrieval_cases[case_id] for case_id in wanted},
        answer_fn=_unused_answerer,
        out_dir=args.out_dir,
        config=config,
        simulated=bool(retrieval.get("simulated")),
        max_retries=args.max_retries,
        backoff_s=args.retry_backoff,
    )

    manifest_path = args.out_dir / "run_manifest.json"
    started_at = now_iso()
    code_files = [
        REPO_ROOT / "evaluation" / "batch.py",
        REPO_ROOT / "evaluation" / "reporting.py",
        REPO_ROOT / "evaluation" / "metrics" / "retrieval.py",
        REPO_ROOT / "evaluation" / "metrics" / "answer.py",
        REPO_ROOT / "evaluation" / "metrics" / "citation.py",
        REPO_ROOT / "evaluation" / "adapters" / "base.py",
        REPO_ROOT / "evaluation" / "adapters" / "_settings.py",
        REPO_ROOT / "evaluation" / "adapters" / "a_index.py",
        REPO_ROOT / "evaluation" / "adapters" / "b_a_index.py",
        REPO_ROOT / "evaluation" / "adapters" / "c_answer.py",
        REPO_ROOT / "evaluation" / "scripts" / "run_batch_e2e.py",
    ]

    def manifest(status: str, counts: dict | None, finished_at: str | None) -> dict:
        index_dir = retrieval.get("index_dir")
        return build_manifest(
            repo_root=REPO_ROOT,
            config=config,
            code_files=code_files,
            dataset_path=args.cases,
            index_manifest_path=Path(index_dir) / "index.json" if index_dir else None,
            retrieval_path=args.retrieval,
            simulated=bool(retrieval.get("simulated")),
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            counts=counts,
        )

    _write_manifest(manifest_path, manifest("running", None, None))

    stats = {"ok": 0, "failed": 0, "skipped": 0}
    try:
        if not args.rebuild_only:
            from evaluation.adapters.c_answer import EvidenceAnswerer

            answerer = EvidenceAnswerer(args.env_file)

            def answer_fn(question, items):
                return answerer.answer(question, items)

            runner.answer_fn = answer_fn
            stats = runner.run(case_ids=wanted, resume=args.resume)

        rebuilt = runner.rebuild()
    except BaseException as error:
        interrupted_counts: dict = {}
        try:
            interrupted_counts = runner.rebuild()
        except Exception:  # noqa: BLE001 - best effort on the way out
            pass
        _write_manifest(
            manifest_path,
            manifest("interrupted", interrupted_counts, now_iso()),
        )
        print(
            f"INTERRUPTED ({type(error).__name__}); partial outputs rebuilt from "
            f"per_case.jsonl at {args.out_dir}",
            file=sys.stderr,
        )
        raise

    from evaluation.adapters.c_answer import llm_call_count

    calls = 0 if args.rebuild_only else llm_call_count()
    (args.out_dir / "llm_calls.json").write_text(
        json.dumps(
            {"llm_calls": calls, "per_case_upper_bound": 2 * len(wanted)},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {
        **rebuilt,
        "skipped": stats.get("skipped", 0),
        "resume": bool(args.resume),
        "rebuild_only": bool(args.rebuild_only),
        "llm_calls": calls,
    }
    _write_manifest(manifest_path, manifest("finished", summary, now_iso()))
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"-> {args.out_dir}")
    return 0 if rebuilt["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
