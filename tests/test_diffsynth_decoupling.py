"""Verify that open_wam/ does not directly import from diffsynth.

After Phase 3, all diffsynth access should go through third_party/diffsynth/
via sys.path injection, not through direct `from diffsynth...` import statements
in open_wam/ source files.
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_no_direct_diffsynth_import_in_open_wam():
    """open_wam/ source files must not contain 'from diffsynth' or 'import diffsynth'.

    Exception: architecture wrappers (open_wam/models/architectures/),
    backbone wrappers (open_wam/models/backbone/), inference optimization
    wrappers, and the package-native inference/training boundary modules are
    allowed to import from diffsynth via sys.path, as they are designated
    boundary layers.
    """
    open_wam_dir = PROJECT_ROOT / "open_wam"
    # Directories allowed to import diffsynth (boundary wrappers)
    allowed_dirs = {
        open_wam_dir / "models" / "architectures",
        open_wam_dir / "models" / "backbone",
        open_wam_dir / "inference" / "optimizations",
    }
    allowed_files = {
        open_wam_dir / "inference" / "model_loader.py",
        open_wam_dir / "inference" / "schedule.py",
        open_wam_dir / "training" / "legacy.py",
    }
    violations = []

    for py_file in open_wam_dir.rglob("*.py"):
        # Skip files in allowed wrapper directories
        if py_file in allowed_files or any(py_file.is_relative_to(d) for d in allowed_dirs):
            continue
        with open(py_file) as f:
            for i, line in enumerate(f, 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if "from diffsynth" in stripped or "import diffsynth" in stripped:
                    rel = py_file.relative_to(PROJECT_ROOT)
                    violations.append(f"{rel}:{i}: {stripped}")

    assert violations == [], (
        "open_wam/ should not directly import from diffsynth "
        "(except in designated boundary layers).\n"
        "Violations:\n" + "\n".join(violations)
    )


def test_diffsynth_moved_to_third_party():
    """diffsynth/ should no longer exist at project root."""
    assert not (PROJECT_ROOT / "diffsynth").is_dir(), (
        "diffsynth/ still exists at project root — should be third_party/diffsynth/"
    )
    assert (PROJECT_ROOT / "third_party" / "diffsynth").is_dir(), (
        "third_party/diffsynth/ does not exist"
    )


def test_diffsynth_importable_via_third_party():
    """diffsynth should be importable after sys.path includes third_party/."""
    import open_wam  # noqa: F401 — triggers sys.path injection
    import diffsynth
    assert hasattr(diffsynth, "__file__")
    # Verify it resolves to third_party/
    assert "third_party" in str(Path(diffsynth.__file__).resolve())


def test_scripts_do_not_import_diffsynth_without_third_party_path():
    """scripts/ files should add third_party to sys.path before importing diffsynth."""
    scripts_dir = PROJECT_ROOT / "scripts"
    for py_file in scripts_dir.glob("*.py"):
        content = py_file.read_text()
        if "from diffsynth" in content or "import diffsynth" in content:
            assert "THIRD_PARTY" in content or "third_party" in content, (
                f"{py_file.name} imports diffsynth but doesn't reference THIRD_PARTY path"
            )
