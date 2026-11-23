"""Adapter for Part B's standalone retrieval index.

Part B ships ``scripts/part_b_retrieval.py`` with ``build`` / ``query``
subcommands and a JSON-on-stdout contract. We deliberately do **not** import or
modify that module: it validates API keys at import time and binds process-wide
settings, so the adapter shells out to its CLI and only parses the JSON.

When A's main index becomes queryable, write a sibling adapter with the same
``retrieve()`` signature and switch ``--adapter`` in the run script; no metric
code needs to change.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from evaluation.adapters.base import RetrievedItem, from_part_b_result

Runner = Callable[[Sequence[str]], str]


class PartBRetrievalAdapter:
    """Query Part B's index through its CLI.

    Args:
        index_dir: directory created by ``part_b_retrieval.py build``.
        env_file: local ``.env`` holding ``DASHSCOPE_API_KEY`` (Part B's
            requirement). Never stored in the repository.
        mode: ``vector`` | ``hybrid`` | ``rrf``.
        repo_root: repository root, used to locate the script.
        python: interpreter used for the subprocess (defaults to this one).
        runner: injectable ``args -> stdout`` callable for tests.
    """

    name = "part-b"

    def __init__(
        self,
        index_dir: str | Path,
        env_file: str | Path | None = None,
        mode: str = "vector",
        repo_root: str | Path | None = None,
        python: str | None = None,
        runner: Runner | None = None,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.env_file = Path(env_file) if env_file else None
        self.mode = mode
        self.repo_root = Path(repo_root) if repo_root else Path.cwd()
        self.python = python or sys.executable
        self._runner = runner

    @property
    def script_path(self) -> Path:
        return self.repo_root / "scripts" / "part_b_retrieval.py"

    def _build_args(self, question: str, top_k: int, file: str | None) -> list[str]:
        args = [
            self.python,
            str(self.script_path),
            "query",
            "--index-dir",
            str(self.index_dir),
            "--question",
            question,
            "--top-k",
            str(top_k),
            "--mode",
            self.mode,
        ]
        if self.env_file is not None:
            args.extend(["--env-file", str(self.env_file)])
        if file:
            args.extend(["--file", file])
        return args

    def _run(self, args: Sequence[str]) -> str:
        if self._runner is not None:
            return self._runner(args)
        if not self.index_dir.is_dir():
            raise FileNotFoundError(
                f"Part B index not found: {self.index_dir}. "
                "Build it first with scripts/part_b_retrieval.py build."
            )
        if not self.script_path.is_file():
            raise FileNotFoundError(f"Part B script not found: {self.script_path}")
        completed = subprocess.run(
            list(args),
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "part_b_retrieval.py failed "
                f"(exit {completed.returncode}): {completed.stderr.strip()[:800]}"
            )
        return completed.stdout

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        case: Mapping[str, Any] | None = None,
    ) -> Sequence[RetrievedItem]:
        file = None
        if case is not None:
            files = case.get("expected_files") or []
            if len(files) == 1:
                file = str(files[0])
        stdout = self._run(self._build_args(question, top_k, file))
        payload = _parse_payload(stdout)
        results = payload.get("results")
        if not isinstance(results, list):
            raise ValueError("Part B output has no 'results' list")
        return [from_part_b_result(row) for row in results]


def _parse_payload(stdout: str) -> dict[str, Any]:
    """Parse the JSON object printed by Part B (tolerating log prefixes)."""
    text = stdout.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("could not find a JSON object in Part B output") from None
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Part B output JSON is not an object")
    return payload
