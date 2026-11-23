"""Resilient batch execution for real end-to-end runs (D-side only).

Design goals
------------
* one question must never abort the batch (per-case ``try/except``);
* every finished case is appended to ``per_case.jsonl`` and flushed/fsynced, so a
  crash keeps completed work;
* ``--resume`` skips only *successful, complete* records produced with the same
  configuration fingerprint; failed cases are retried;
* transient errors (HTTP 429/5xx, rate-limit/timeout) get a bounded number of
  retries with exponential backoff; other errors fail immediately;
* ``answers.json``, ``predictions.jsonl``, ``failures.json`` and the automatic
  report can always be rebuilt from ``per_case.jsonl``;
* the run manifest never stores API keys (whitelisted config only).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

TRANSIENT_STATUS = {429, 500, 502, 503, 504}
TRANSIENT_HINTS = (
    "rate limit",
    "ratelimit",
    "too many requests",
    "timeout",
    "timed out",
    "temporarily",
    "connection reset",
    "connection aborted",
    "service unavailable",
    "internal server error",
)
REQUIRED_RECORD_KEYS = (
    "case_id",
    "status",
    "attempts",
    "config_fingerprint",
    "simulated",
    "answer_text",
    "evidence_index",
)
PUBLIC_CONFIG_KEYS = (
    "dataset_sha256",
    "retrieval_sha256",
    "index_manifest_sha256",
    "chunking",
    "top_k",
    "retrieval_method",
    "llm_model",
    "llm_endpoint",
    "embedding_model",
    "embedding_dimensions",
    "adapter_version",
    "simulated",
)
PREDICTION_KEYS = ("chunk_id", "file", "locator", "score", "text")


def sha256_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def public_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Whitelist config fields so a secret can never leak into the manifest."""
    return {key: config[key] for key in PUBLIC_CONFIG_KEYS if key in config}


def config_fingerprint(config: Mapping[str, Any]) -> str:
    payload = json.dumps(public_config(config), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def git_commit(repo_root: str | Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except Exception:  # pragma: no cover - git may be unavailable
        return None
    return completed.stdout.strip() or None if completed.returncode == 0 else None


def build_manifest(
    *,
    repo_root: str | Path,
    config: Mapping[str, Any],
    code_files: Sequence[str | Path],
    dataset_path: str | Path,
    index_manifest_path: str | Path | None,
    retrieval_path: str | Path,
    simulated: bool,
    status: str,
    started_at: str,
    finished_at: str | None = None,
    counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": status,
        "simulated": bool(simulated),
        "started_at": started_at,
        "finished_at": finished_at,
        "git": {"commit": git_commit(repo_root)},
        "code": {Path(path).name: sha256_file(path) for path in code_files},
        "dataset": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
        },
        "index": {
            "manifest": str(index_manifest_path) if index_manifest_path else None,
            "sha256": sha256_file(index_manifest_path),
        },
        "retrieval": {
            "path": str(retrieval_path),
            "sha256": sha256_file(retrieval_path),
        },
        "config": public_config(config),
        "config_fingerprint": config_fingerprint(config),
        "counts": dict(counts or {}),
        "secrets_recorded": False,
    }


def _is_transient(error: BaseException) -> bool:
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if isinstance(status, int) and status in TRANSIENT_STATUS:
        return True
    name = type(error).__name__.lower()
    if any(hint.replace(" ", "") in name for hint in TRANSIENT_HINTS):
        return True
    message = str(error).lower()
    return any(hint in message for hint in TRANSIENT_HINTS)


def is_transient_error(error: BaseException) -> bool:
    """Public alias used by the retrieval batch runner."""
    return _is_transient(error)


def retry_call(
    action: Callable[[], Any],
    *,
    max_retries: int = 2,
    backoff_s: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] | None = None,
) -> tuple[Any, int]:
    """Run ``action`` with bounded exponential backoff on transient errors."""
    attempts = 0
    while True:
        attempts += 1
        try:
            return action(), attempts
        except Exception as error:  # noqa: BLE001 - classified below
            if attempts <= max_retries and _is_transient(error):
                delay = backoff_s * (2 ** (attempts - 1))
                if log is not None:
                    log(f"transient {type(error).__name__}, retry in {delay:.1f}s")
                sleep(delay)
                continue
            raise


class BatchRunner:
    """Run ``answer_fn`` per case with isolation, retries, resume and rebuild."""

    def __init__(
        self,
        *,
        cases: Sequence[Mapping[str, Any]],
        retrieval_cases: Mapping[str, Mapping[str, Any]],
        answer_fn: Callable[[str, Sequence[Mapping[str, Any]]], Mapping[str, Any]],
        out_dir: str | Path,
        config: Mapping[str, Any],
        simulated: bool = False,
        max_retries: int = 2,
        backoff_s: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        log: Callable[[str], None] = print,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.cases = {str(case["id"]): case for case in cases}
        self.retrieval_cases = {
            str(key): dict(value) for key, value in retrieval_cases.items()
        }
        self.answer_fn = answer_fn
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.per_case_path = self.out_dir / "per_case.jsonl"
        self.config = dict(config)
        self.fingerprint = config_fingerprint(self.config)
        self.simulated = bool(simulated)
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.sleep = sleep
        self.log = log

    # ---------------------------------------------------------- persistence
    def _append_record(self, record: Mapping[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self.per_case_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def read_records(self) -> list[dict[str, Any]]:
        if not self.per_case_path.is_file():
            return []
        records: list[dict[str, Any]] = []
        for line in self.per_case_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records

    def latest_records(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for record in self.read_records():
            latest[str(record.get("case_id"))] = record
        return latest

    def is_complete(self, record: Mapping[str, Any] | None) -> bool:
        if not record or record.get("status") != "ok":
            return False
        if record.get("config_fingerprint") != self.fingerprint:
            return False
        if any(key not in record for key in REQUIRED_RECORD_KEYS):
            return False
        if not isinstance(record.get("answer_text"), str):
            return False
        if not isinstance(record.get("evidence_index"), list):
            return False
        return True

    # ------------------------------------------------------------- execution
    def _record_ok(
        self, case: Mapping[str, Any], result: Mapping[str, Any], attempts: int
    ) -> dict[str, Any]:
        return {
            "case_id": str(case["id"]),
            "status": "ok",
            "simulated": self.simulated,
            "config_fingerprint": self.fingerprint,
            "attempts": attempts,
            "category": case.get("category"),
            "question_type": case.get("question_type"),
            "question": case.get("question"),
            "retrieved_count": len(
                self.retrieval_cases.get(str(case["id"]), {}).get("results", [])
            ),
            "answer_text": result.get("answer_text", ""),
            "evidence_index": result.get("evidence_index") or [],
            "citation_status": result.get("citation_status"),
            "citation_validation": result.get("citation_validation"),
            "latency_s": result.get("latency_s"),
            "created_at": now_iso(),
        }

    def _record_failed(
        self, case: Mapping[str, Any], error: BaseException, attempts: int
    ) -> dict[str, Any]:
        return {
            "case_id": str(case["id"]),
            "status": "failed",
            "simulated": self.simulated,
            "config_fingerprint": self.fingerprint,
            "attempts": attempts,
            "category": case.get("category"),
            "question_type": case.get("question_type"),
            "question": case.get("question"),
            "error": {
                "type": type(error).__name__,
                "message": str(error)[:500],
                "retryable": _is_transient(error),
            },
            "created_at": now_iso(),
        }

    def run(
        self, case_ids: Sequence[str] | None = None, resume: bool = False
    ) -> dict[str, Any]:
        selected = list(case_ids or self.cases)
        latest = self.latest_records()
        stats = {"total": len(selected), "ok": 0, "failed": 0, "skipped": 0}
        for case_id in selected:
            case = self.cases[case_id]
            if resume and self.is_complete(latest.get(case_id)):
                stats["skipped"] += 1
                continue
            items = self.retrieval_cases.get(case_id, {}).get("results", [])
            attempts = 0
            while True:
                attempts += 1
                try:
                    result = self.answer_fn(str(case["question"]), items)
                except Exception as error:  # noqa: BLE001 - isolation is the point
                    if attempts <= self.max_retries and _is_transient(error):
                        delay = self.backoff_s * (2 ** (attempts - 1))
                        self.log(
                            f"{case_id}: transient {type(error).__name__}, "
                            f"retry {attempts}/{self.max_retries} in {delay:.1f}s"
                        )
                        self.sleep(delay)
                        continue
                    record = self._record_failed(case, error, attempts)
                    self._append_record(record)
                    stats["failed"] += 1
                    break
                record = self._record_ok(case, result, attempts)
                self._append_record(record)
                stats["ok"] += 1
                break
        return stats

    # -------------------------------------------------------------- rebuild
    def rebuild(self) -> dict[str, Any]:
        latest = self.latest_records()
        ok_records = {
            case_id: record
            for case_id, record in latest.items()
            if record.get("status") == "ok" and self.is_complete(record)
        }
        failed = {
            case_id: record
            for case_id, record in latest.items()
            if record.get("status") == "failed"
        }
        missing = [
            case_id for case_id in self.cases if case_id not in latest
        ]

        answers = {
            "schema_version": 2,
            "simulated": self.simulated,
            "config_fingerprint": self.fingerprint,
            "generated_at": now_iso(),
            "cases": {
                case_id: {
                    "question": record.get("question"),
                    "answer_text": record["answer_text"],
                    "evidence_index": record["evidence_index"],
                    "citation_status": record.get("citation_status"),
                    "citation_validation": record.get("citation_validation"),
                    "latency_s": record.get("latency_s"),
                    "attempts": record.get("attempts"),
                }
                for case_id, record in ok_records.items()
            },
        }
        (self.out_dir / "answers.json").write_text(
            json.dumps(answers, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        with (self.out_dir / "predictions.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            for case_id, record in ok_records.items():
                handle.write(
                    json.dumps(
                        {
                            "case_id": case_id,
                            "text": record["answer_text"],
                            "evidence_index": record["evidence_index"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        failures = {
            "total": len(failed),
            "missing": missing,
            "cases": {
                case_id: {
                    "attempts": record.get("attempts"),
                    "error": record.get("error"),
                    "updated_at": record.get("created_at"),
                }
                for case_id, record in failed.items()
            },
        }
        (self.out_dir / "failures.json").write_text(
            json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        report = self._build_report(ok_records)
        summary = {
            "config_fingerprint": self.fingerprint,
            "simulated": self.simulated,
            "counts": {
                "cases": len(self.cases),
                "ok": len(ok_records),
                "failed": len(failed),
                "missing": len(missing),
                "scored": 0,
                "unscored": len(ok_records),
            },
            "statuses": {
                "real_or_simulated": "simulated" if self.simulated else "real",
                "failed": sorted(failed),
                "missing": sorted(missing),
            },
            "report": report,
        }
        (self.out_dir / "report_auto.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return {
            "ok": len(ok_records),
            "failed": len(failed),
            "missing": len(missing),
            "unscored": len(ok_records),
        }

    def _build_report(
        self, ok_records: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, Any]:
        from evaluation.reporting import aggregate, build_case_records

        selected_cases = [
            self.cases[case_id] for case_id in ok_records if case_id in self.cases
        ]
        rankings = {
            case_id: list(payload.get("results") or [])
            for case_id, payload in self.retrieval_cases.items()
            if case_id in ok_records
        }
        predictions = {
            case_id: {
                "text": record["answer_text"],
                "evidence_index": record["evidence_index"],
            }
            for case_id, record in ok_records.items()
        }
        config = self.config
        records = build_case_records(
            selected_cases,
            rankings,
            predictions,
            {},  # answer_correct is human-only
            {},  # citation_supported is human-only
            chunking_strategy=str(config.get("chunking", "unknown")),
            retrieval_method=str(config.get("retrieval_method", "unknown")),
            top_k=int(config.get("top_k", 0)),
            reranking=str(config.get("reranking", "off")),
            simulated=self.simulated,
        )
        return aggregate(records)
