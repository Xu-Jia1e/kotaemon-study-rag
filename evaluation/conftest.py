"""Pytest bootstrap: make the repository root importable.

``THEFLOW_SETTINGS_MODULE`` is set because importing some project modules (for
example ``kotaemon.indices.qa``) pulls in ``ktem``, whose settings layer needs a
settings module. Pointing it at the repository's ``flowsettings.py`` keeps the
tests offline and avoids the ``Setting KH_DATABASE not found`` error.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("THEFLOW_SETTINGS_MODULE", "flowsettings")

