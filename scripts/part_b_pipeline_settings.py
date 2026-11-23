"""Isolate the original application's database for the Part B experiment."""

import os
from pathlib import Path

from flowsettings import *  # noqa: F401,F403

_data = Path(os.environ["PART_B_PIPELINE_DIR"]).resolve()
KH_DATABASE = f"sqlite:///{(_data / 'pipeline.sqlite3').as_posix()}"
KH_CHUNKS_OUTPUT_DIR = None
KH_ENABLE_ALEMBIC = False
