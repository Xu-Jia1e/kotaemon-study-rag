"""Resilient 30-question real retrieval over A's index (no LLM calls).

Same P0 mechanics as ``run_batch_e2e.py``: per-case isolation, append+fsync after
each question, ``--resume`` keyed on a config fingerprint, bounded retries with
backoff, a manifest without secrets, and ``--rebuild-only`` reconstruction.

Writes to ``<out-dir>/<tag>/``; the 4-case pilot results are never touched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX = REPO_ROOT / ".eval-index" / "1024-256"
DEFAULT_CASES = REPO_ROOT / "evaluation" / "datasets" / "qa_cases_v2.jsonl"
DEFAULT_OUT = Path(r"E:/dsh_workplace/.eval-data/retrieval-real-30")
ADAPTER_VERSION = "b-a-index-v2"


def _load_cases(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _append(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _latest(path: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    if not path.is_file():
        return latest
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            latest[str(record["case_id"])] = record
    return latest


def _complete(record: dict, fingerprint: str) -> bool:
    return (
        record.get("status") == "ok"
        and record.get("config_fingerprint") == fingerprint
        and isinstance(record.get("results"), list)
    )


def _rebuild(
    *,
    out_dir: Path,
    cases: list[dict],
    fingerprint: str,
    config: dict,
    top_k: int,
    simulated: bool,
    stats: dict,
) -> dict:
    from evaluation.metrics.retrieval import evaluate_retrieval

    latest = _latest(out_dir / "per_case.jsonl")
    ok = {
        case_id: record
        for case_id, record in latest.items()
        if _complete(record, fingerprint)
    }
    failed = {
        case_id: record
        for case_id, record in latest.items()
        if record.get("status") == "failed"
    }
    missing = [case["id"] for case in cases if case["id"] not in latest]

    rankings = {case_id: record["results"] for case_id, record in ok.items()}
    selected = [case for case in cases if case["id"] in ok]
    report = evaluate_retrieval(selected, rankings, k=top_k)

    by_category = {}
    for category in ("direct", "synthesis", "detail", "unanswerable"):
        subset = [case for case in selected if case.get("category") == category]
        if subset:
            by_category[category] = evaluate_retrieval(
                subset, rankings, k=top_k
            ).to_dict()

    case_by_id = {case["id"]: case for case in cases}
    payload = {
        "schema_version": 2,
        "simulated": simulated,
        "mode": "vector",
        "top_k": top_k,
        "config_fingerprint": fingerprint,
        "config": config,
        "cases": [
            {
                "case_id": case_id,
                "category": case_by_id[case_id].get("category"),
                "question": case_by_id[case_id].get("question"),
                "latency_s": record.get("latency_s"),
                "attempts": record.get("attempts"),
                "results": record["results"],
            }
            for case_id, record in ok.items()
            if case_id in case_by_id
        ],
        "metrics": report.to_dict(),
        "by_category": by_category,
        "counts": {
            "total": len(cases),
            "ok": len(ok),
            "failed": len(failed),
            "missing": len(missing),
        },
    }
    _write_json(out_dir / "retrieval.json", payload)
    _write_json(
        out_dir / "failures.json",
        {
            "total": len(failed),
            "missing": missing,
            "cases": {
                case_id: {
                    "attempts": record.get("attempts"),
                    "error": record.get("error"),
                }
                for case_id, record in failed.items()
            },
        },
    )
    _write_json(
        out_dir / "summary.json",
        {
            "config_fingerprint": fingerprint,
            "simulated": simulated,
            "counts": payload["counts"],
            "metrics": payload["metrics"],
            "by_category": by_category,
            "statuses": {
                "real_or_simulated": "simulated" if simulated else "real",
                "failed": sorted(failed),
                "missing": sorted(missing),
            },
            "run": stats,
        },
    )
    stamps = sorted(
        str(record.get("created_at"))
        for record in ok.values()
        if record.get("created_at")
    )
    return {
        "ok": len(ok),
        "failed": len(failed),
        "missing": len(missing),
        "metrics": {
            key: payload["metrics"][key]
            for key in ("hit_at_k", "mrr_at_k", "evidence_recall_at_k", "cases_scored")
        },
        "record_started_at": stamps[0] if stamps else None,
        "record_finished_at": stamps[-1] if stamps else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--env-file", default=str(REPO_ROOT / ".env"))
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tag", default="1024-256-vector-k5")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rebuild-only", action="store_true")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--case-ids", help="comma separated subset")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)

    from evaluation.batch import (
        build_manifest,
        config_fingerprint,
        is_transient_error,
        now_iso,
        retry_call,
        sha256_file,
    )

    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")
    manifest_path = args.index_dir / "index.json"
    if not manifest_path.is_file():
        raise SystemExit(f"index manifest not found: {manifest_path}")
    index_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    cases = _load_cases(args.cases)
    by_id = {case["id"]: case for case in cases}
    wanted = list(by_id)
    if args.case_ids:
        wanted = [item.strip() for item in args.case_ids.split(",") if item.strip()]
    if args.limit:
        wanted = wanted[: args.limit]
    unknown = [case_id for case_id in wanted if case_id not in by_id]
    if unknown:
        raise SystemExit(f"unknown case ids: {unknown}")

    out_dir = args.out_dir / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    per_case = out_dir / "per_case.jsonl"

    from evaluation.adapters.a_index import normalize_manifest

    normalized_manifest = normalize_manifest(index_manifest)
    if normalized_manifest["complete"] is False:
        raise SystemExit(f"index manifest reports incomplete build: {manifest_path}")
    config = {
        "dataset_sha256": sha256_file(args.cases),
        "index_manifest_sha256": sha256_file(manifest_path),
        "chunking": (
            f"{normalized_manifest['chunk_size']}-"
            f"{normalized_manifest['chunk_overlap']}"
        ),
        "top_k": args.top_k,
        "retrieval_method": "vector",
        "embedding_model": normalized_manifest["embedding_model"],
        "embedding_dimensions": normalized_manifest["embedding_dimensions"],
        "adapter_version": ADAPTER_VERSION,
        "simulated": False,
    }
    fingerprint = config_fingerprint(config)

    code_files = [
        REPO_ROOT / "evaluation" / "batch.py",
        REPO_ROOT / "evaluation" / "metrics" / "retrieval.py",
        REPO_ROOT / "evaluation" / "adapters" / "base.py",
        REPO_ROOT / "evaluation" / "adapters" / "_settings.py",
        REPO_ROOT / "evaluation" / "adapters" / "a_index.py",
        REPO_ROOT / "evaluation" / "adapters" / "b_a_index.py",
        REPO_ROOT / "evaluation" / "scripts" / "run_batch_retrieval.py",
    ]

    def manifest(status: str, counts: dict | None, finished_at: str | None) -> dict:
        return build_manifest(
            repo_root=REPO_ROOT,
            config=config,
            code_files=code_files,
            dataset_path=args.cases,
            index_manifest_path=manifest_path,
            retrieval_path=per_case,
            simulated=False,
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            counts=counts,
        )

    started_at = now_iso()
    _write_json(out_dir / "run_manifest.json", manifest("running", None, None))

    stats = {"ok": 0, "failed": 0, "skipped": 0}
    try:
        if not args.rebuild_only:
            from evaluation.adapters.b_a_index import AIndexVectorAdapter

            adapter = AIndexVectorAdapter(args.index_dir, env_file=args.env_file)
            latest = _latest(per_case)
            for case_id in wanted:
                case = by_id[case_id]
                if args.resume and _complete(latest.get(case_id, {}), fingerprint):
                    stats["skipped"] += 1
                    continue
                started = time.perf_counter()
                try:
                    items, attempts = retry_call(
                        lambda: adapter.retrieve(
                            str(case["question"]), top_k=args.top_k, case=case
                        ),
                        max_retries=args.max_retries,
                        backoff_s=args.retry_backoff,
                        log=lambda message: print(f"{case_id}: {message}"),
                    )
                except Exception as error:  # noqa: BLE001 - isolation is the point
                    _append(
                        per_case,
                        {
                            "case_id": case_id,
                            "status": "failed",
                            "simulated": False,
                            "config_fingerprint": fingerprint,
                            "attempts": args.max_retries + 1,
                            "question": case.get("question"),
                            "category": case.get("category"),
                            "error": {
                                "type": type(error).__name__,
                                "message": str(error)[:500],
                                "retryable": is_transient_error(error),
                            },
                            "created_at": now_iso(),
                        },
                    )
                    stats["failed"] += 1
                    continue
                _append(
                    per_case,
                    {
                        "case_id": case_id,
                        "status": "ok",
                        "simulated": False,
                        "config_fingerprint": fingerprint,
                        "attempts": attempts,
                        "question": case.get("question"),
                        "category": case.get("category"),
                        "latency_s": round(time.perf_counter() - started, 3),
                        "results": [item.to_dict() for item in items],
                        "created_at": now_iso(),
                    },
                )
                stats["ok"] += 1
        result = _rebuild(
            out_dir=out_dir,
            cases=cases,
            fingerprint=fingerprint,
            config=config,
            top_k=args.top_k,
            simulated=False,
            stats=stats,
        )
    except BaseException as error:
        try:
            _rebuild(
                out_dir=out_dir,
                cases=cases,
                fingerprint=fingerprint,
                config=config,
                top_k=args.top_k,
                simulated=False,
                stats=stats,
            )
        except Exception:  # noqa: BLE001 - best effort on the way out
            pass
        _write_json(
            out_dir / "run_manifest.json",
            manifest("interrupted", {"interrupted": 1}, now_iso()),
        )
        print(
            f"INTERRUPTED ({type(error).__name__}); partial outputs kept in {out_dir}",
            file=sys.stderr,
        )
        raise

    finished_at = (
        result.get("record_finished_at")
        if args.rebuild_only and result.get("record_finished_at")
        else now_iso()
    )
    if args.rebuild_only and result.get("record_started_at"):
        started_at = result["record_started_at"]
    _write_json(
        out_dir / "run_manifest.json", manifest("finished", result, finished_at)
    )
    print(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"-> {out_dir}")
    return 0 if result["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
