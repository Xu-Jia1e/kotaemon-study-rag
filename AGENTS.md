# AGENTS.md

Fork of [Cinnamon/kotaemon](https://github.com/Cinnamon/kotaemon) customized for Chinese/English study-material RAG (PDF/PPTX/DOCX/Markdown/HTML ingestion, OCR, citation QA). Upstream docs live in `README.upstream.md`; the project-specific overview and acceptance results are in `README.md`.

## Layout

- `libs/kotaemon/kotaemon/` — core library (loaders, embeddings, indices, vectorstore, LLMs, QA). Custom code here differs from upstream.
- `libs/ktem/ktem/` — Gradio app, DB models, UI pages, RAG pipelines.
- `app.py` — app entrypoint; `sso_app*.py` for SSO/demo modes.
- `test-materials/` — acceptance fixtures (do not treat as unit tests; see below).

## Setup & run

Requires Python 3.10 (`.python-version`) and `uv` (workspace of `libs/kotaemon` + `libs/ktem`).

```sh
uv sync --locked --package kotaemon --package ktem
uv run --no-sync python app.py
```

App reads `flowsettings.py`; runtime data, SQLite DB, docstore (LanceDB), and vectorstore (Chroma) are created under `ktem_app_data/` (gitignored). Runtime config can be overridden via `.env` / `settings.yaml` (see `flowsettings.py`, `.env.example`, `settings.yaml.example`).

## Tests

Run from `libs/kotaemon` (pytest.ini defines `testpaths = tests`):

```sh
cd libs/kotaemon && pytest              # single file: pytest tests/test_reader.py -k name
```

Tests skip automatically when optional deps (unstructured, paddleocr, cohere, voyageai, llama_cpp, sentence-transformers) are absent — don't assume a skip means a broken test. `ktem` tests live in `libs/ktem/ktem_tests/`.

## Lint

Pre-commit is the source of truth: black, isort (`--profile black`), flake8 (`max-line-length=88`, ignore `E203`), autoflake, mypy, codespell, plus yaml/toml checks.

```sh
pre-commit run --all-files
```

## Repo-specific quirks

- **Commits are conventional commits** (`.commitlintrc`, enforced by CI `pr-lint`). Format `<type>(<scope>): <subject>`; types: build/chore/ci/docs/feat/fix/perf/refactor/revert/style/test. Subject not empty, no trailing period.
- **Embedding batching**: `OpenAIEmbeddings` special-cases Bailian `text-embedding-v4` — batches of 10 and forces `encoding_format="float"` (`libs/kotaemon/kotaemon/embeddings/openai.py`). Preserve this when touching embedding code.
- **DeepSeek thinking conflict**: `ChatOpenAI` disables thinking (`extra_body={"thinking":{"type":"disabled"}}`) when forcing tool calls against `api.deepseek.com` (`libs/kotaemon/kotaemon/llms/chats/openai.py`). Removing this breaks citation requests.
- **PDF/PPTX/DOCX loaders** do local OCR via RapidOCR (`loaders/utils/image_ocr.py`), preserve page/slide/section anchors, and restore math/fraction structure. PDF preview uses `PDF_LOADER_DPI` env (default 40).
- **Storage split**: Chroma = vectors, LanceDB = chunks/previews, SQLite = source linkage. Forced reindex clears old records.
- **Evidence labels**: QA renders `Page`/`Slide` and optional `section` in citations (`indices/qa/format_context.py`); keep labels correct when editing retrieval output.

## Acceptance (not unit tests)

`test-materials/` holds fixed public sample documents, expected chunk/vector counts (`ingestion-results.json`), and six QA cases (`search-cases.md`). Download samples via `python test-materials/download_samples.py` (verifies SHA-256, writes to `downloads/`, gitignored). These validate real model endpoints (DeepSeek + Bailian), so they are not run in CI.
