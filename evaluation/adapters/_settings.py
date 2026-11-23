"""Isolated theflow settings for D-owned index builds.

The application's ``ktem.db.engine`` binds ``KH_DATABASE`` at import time. To keep
an evaluation index out of ``ktem_app_data/user_data/sql.db``, ``a_index`` sets
``DSH_EVAL_INDEX_DIR`` and ``THEFLOW_SETTINGS_MODULE`` *before* importing any
``ktem`` module; this module then rewrites ``KH_DATABASE`` to a SQLite file inside
the evaluation index directory.
"""

from __future__ import annotations

import os
from pathlib import Path

from flowsettings import *  # noqa: F401,F403

_index_dir = Path(os.environ["DSH_EVAL_INDEX_DIR"]).resolve()
KH_DATABASE = f"sqlite:///{(_index_dir / 'pipeline.sqlite3').as_posix()}"
KH_CHUNKS_OUTPUT_DIR = None
KH_ENABLE_ALEMBIC = False
