"""Runtime-patched wrapper around RoboTwin's ``script/eval_policy.py``.

Used by step_analysis.sh to override the hardcoded ``test_num = 100`` without
forking the upstream script. Set ``ROBOTWIN_TEST_NUM`` to pick a different
episode count (default 5).

Must be launched with cwd = ``$ROBOTWIN_PATH`` so the relative paths inside
eval_policy.py (``./``, ``./policy``, ``./description/utils``) resolve. The
original script's argparse + main() run inside the ``exec()`` below, so the
CLI contract is identical to ``eval_policy.py``.
"""

import os
import sys

# Mirror the path setup eval_policy.py does itself, so imports work when this
# file is invoked via `python /abs/path/eval_policy_steps.py` with cwd at the
# RoboTwin root. ``./script`` is added because the original file relies on
# Python's implicit ``sys.path[0] = dirname(script)`` to resolve imports like
# ``from test_render import Sapien_TEST``; that resolution points at our
# wrapper's directory instead, so we restore it explicitly.
sys.path.append("./")
sys.path.append("./policy")
sys.path.append("./description/utils")
sys.path.append("./script")

TEST_NUM = int(os.environ.get("ROBOTWIN_TEST_NUM", "5"))

SOURCE_PATH = os.path.abspath("./script/eval_policy.py")
if not os.path.isfile(SOURCE_PATH):
    raise RuntimeError(
        f"eval_policy.py not found at {SOURCE_PATH}. "
        "Launch this wrapper with cwd set to the RoboTwin repo root."
    )

with open(SOURCE_PATH, "r", encoding="utf-8") as f:
    src = f.read()

needle = "test_num = 100"
if needle not in src:
    raise RuntimeError(
        f"Expected '{needle}' in {SOURCE_PATH} — RoboTwin may have changed. "
        "Update this patch script to match the new line."
    )
src = src.replace(needle, f"test_num = {TEST_NUM}  # patched by eval_policy_steps.py")

# Also un-silence the expert-check `except Exception` that only prints
# "error occurs !" with no traceback — it hides real failures (missing
# assets, pytorch3d, SAPIEN state, etc.). Gated by ROBOTWIN_VERBOSE_ERRORS
# so the default stays aligned with upstream behavior.
if os.environ.get("ROBOTWIN_VERBOSE_ERRORS", "1") != "0":
    # traceback is already imported at the top of eval_policy.py, so we can
    # rely on it being in scope here.
    verbose_needle = 'print("error occurs !")'
    verbose_repl = 'print("error occurs !", repr(e)); print(traceback.format_exc())'
    if verbose_needle in src:
        src = src.replace(verbose_needle, verbose_repl)

code = compile(src, SOURCE_PATH, "exec")
exec(code, {"__name__": "__main__", "__file__": SOURCE_PATH})
