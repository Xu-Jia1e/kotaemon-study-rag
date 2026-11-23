import os
import sys
import tempfile
from pathlib import Path

# The flow settings singleton (``theflow.settings.settings``) is initialised once
# per process, on first access. Some test modules import ``kotaemon`` (and thus
# ``theflow``) during collection; if that happens before the QA tests run, the
# settings are frozen without ``flowsettings`` and ``ktem``'s ``KH_DATABASE``
# lookup fails during import. Configure settings here, in conftest, because it is
# imported before any test module is collected.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
# Only point the flow settings loader at the repository ``flowsettings`` when it
# actually exists; otherwise let ``theflow`` fall back to its defaults (which
# provide ``STORAGE``/``BASE_BACKEND``) and inject ``KH_*`` values below.
_HAS_FLOWSETTINGS = (_REPO_ROOT / "flowsettings.py").is_file()
if _HAS_FLOWSETTINGS:
    os.environ.setdefault("THEFLOW_SETTINGS_MODULE", "flowsettings")


def _ensure_flow_settings() -> None:
    """Make sure ``theflow.settings`` exposes ``KH_DATABASE`` and friends.

    Prefer the repository ``flowsettings``. Import ``theflow`` first so that
    ``flowsettings``'s ``from theflow.settings.default import *`` does not
    re-enter a partially-initialised ``theflow`` package (which would raise
    ``Setting STORAGE not found``). If ``flowsettings`` still cannot be loaded
    (e.g. a minimal CI environment), fall back to an isolated sqlite database
    under a temporary directory so collection never needs a real database or
    ``.env``.
    """
    if _HAS_FLOWSETTINGS:
        try:
            import theflow  # noqa: F401  (initialise the package before settings)

            import flowsettings  # noqa: F401
        except Exception:  # pragma: no cover - only on incomplete environments
            pass

    from theflow.settings import settings

    if "KH_DATABASE" not in vars(settings):
        fallback = Path(tempfile.gettempdir()) / "kotaemon_tests"
        fallback.mkdir(parents=True, exist_ok=True)
        settings.KH_DATABASE = f"sqlite:///{fallback / 'sql.db'}"
    if "BASE_BACKEND" not in vars(settings):
        settings.BASE_BACKEND = "sqlite"


_ensure_flow_settings()


import pytest  # noqa: E402


@pytest.fixture(scope="function")
def mock_google_search(monkeypatch):
    import googlesearch

    def result(*args, **kwargs):
        yield googlesearch.SearchResult(
            url="https://www.cinnamon.is/en/",
            title="Cinnamon AI",
            description="Cinnamon AI is an enterprise AI company.",
        )

    monkeypatch.setattr(googlesearch, "search", result)


@pytest.fixture(scope="function")
def mock_wikipedia(monkeypatch):
    """Avoid live Wikipedia/MediaWiki calls in unit tests."""

    class _FakePage:
        content = "Cinnamon AI is an enterprise AI company."
        url = "https://en.wikipedia.org/wiki/Cinnamon"

    def _page(_title: str) -> _FakePage:
        return _FakePage()

    def _search(_query: str) -> list[str]:
        return ["Cinnamon"]

    monkeypatch.setattr("wikipedia.page", _page)
    monkeypatch.setattr("wikipedia.search", _search)


@pytest.fixture(scope="function")
def mock_langchain_search_tools(monkeypatch):
    """Stub LangChain community search tools used in agent tests."""

    def _wikipedia_run(_query: str) -> str:
        return "Cinnamon AI is an enterprise AI company."

    def _duckduckgo_run(_query: str) -> str:
        return "Cinnamon AI is an enterprise AI company."

    monkeypatch.setattr(
        "langchain_community.tools.wikipedia.tool.WikipediaQueryRun._run",
        lambda self, query: _wikipedia_run(query),
    )
    monkeypatch.setattr(
        "langchain_community.tools.ddg_search.tool.DuckDuckGoSearchRun._run",
        lambda self, query: _duckduckgo_run(query),
    )


def if_haystack_not_installed():
    try:
        import haystack  # noqa: F401
    except ImportError:
        return True
    else:
        return False


def if_sentence_bert_not_installed():
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return True
    else:
        return False


def if_sentence_fastembed_not_installed():
    try:
        import fastembed  # noqa: F401
    except ImportError:
        return True
    else:
        return False


def if_unstructured_pdf_not_installed():
    try:
        import unstructured  # noqa: F401
        from unstructured.partition.pdf import partition_pdf  # noqa: F401
    except ImportError:
        return True
    else:
        return False


def if_cohere_not_installed():
    try:
        import cohere  # noqa: F401
    except ImportError:
        return True
    else:
        return False


def if_llama_cpp_not_installed():
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        return True
    else:
        return False


def if_voyageai_not_installed():
    try:
        import voyageai  # noqa: F401
    except ImportError:
        return True
    else:
        return False


def if_paddleocr_not_installed():
    try:
        import paddle  # noqa: F401
        import paddleocr  # noqa: F401
    except ImportError:
        return True
    else:
        return False


skip_when_haystack_not_installed = pytest.mark.skipif(
    if_haystack_not_installed(), reason="Haystack is not installed"
)

skip_when_sentence_bert_not_installed = pytest.mark.skipif(
    if_sentence_bert_not_installed(), reason="SBert is not installed"
)

skip_when_fastembed_not_installed = pytest.mark.skipif(
    if_sentence_fastembed_not_installed(), reason="fastembed is not installed"
)

skip_when_unstructured_pdf_not_installed = pytest.mark.skipif(
    if_unstructured_pdf_not_installed(), reason="unstructured is not installed"
)

skip_when_cohere_not_installed = pytest.mark.skipif(
    if_cohere_not_installed(), reason="cohere is not installed"
)

skip_openai_lc_wrapper_test = pytest.mark.skipif(
    True, reason="OpenAI LC wrapper test is skipped"
)

skip_llama_cpp_not_installed = pytest.mark.skipif(
    if_llama_cpp_not_installed(), reason="llama_cpp is not installed"
)

skip_when_voyageai_not_installed = pytest.mark.skipif(
    if_voyageai_not_installed(), reason="voyageai is not installed"
)

skip_when_paddleocr_not_installed = pytest.mark.skipif(
    if_paddleocr_not_installed(), reason="paddle/paddleocr is not installed"
)
