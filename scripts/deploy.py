"""Thin launcher for the OpenWAM policy server.

Kept for source-tree workflows (``bash scripts/deploy.sh``) where the openwam
package is not pip-installed: injects the repo root + third_party into
``sys.path``, then delegates to the canonical CLI in
``openwam.deploy.server`` — the same code path as the ``openwam-serve``
console script, with the identical flag set (see ``--help``).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party"))

from openwam.deploy.server import main  # noqa: E402

if __name__ == "__main__":
    main()
