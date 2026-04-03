"""Verify that open_wam/ uses only qualified imports for diffsynth.

All diffsynth access within open_wam/ must use the qualified path
``from third_party.diffsynth...``, never the bare ``from diffsynth...``
form which requires sys.path manipulation.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_no_bare_diffsynth_import_in_open_wam():
    """open_wam/ source files must use 'from third_party.diffsynth', not bare 'from diffsynth'."""
    open_wam_dir = PROJECT_ROOT / "open_wam"
    violations = []

    for py_file in open_wam_dir.rglob("*.py"):
        with open(py_file) as f:
            for i, line in enumerate(f, 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                # Allow qualified imports (from third_party.diffsynth...)
                if "third_party.diffsynth" in stripped:
                    continue
                if "from diffsynth" in stripped or "import diffsynth" in stripped:
                    rel = py_file.relative_to(PROJECT_ROOT)
                    violations.append(f"{rel}:{i}: {stripped}")

    assert violations == [], (
        "open_wam/ should use qualified 'from third_party.diffsynth...' imports, "
        "not bare 'from diffsynth...'.\n"
        "Violations:\n" + "\n".join(violations)
    )


def test_no_sys_path_insert_in_open_wam():
    """open_wam/ must not use sys.path.insert anywhere."""
    open_wam_dir = PROJECT_ROOT / "open_wam"
    violations = []

    for py_file in open_wam_dir.rglob("*.py"):
        with open(py_file) as f:
            for i, line in enumerate(f, 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if "sys.path.insert" in stripped or "sys.path.append" in stripped:
                    rel = py_file.relative_to(PROJECT_ROOT)
                    violations.append(f"{rel}:{i}: {stripped}")

    assert violations == [], (
        "open_wam/ must not use sys.path manipulation.\n"
        "Violations:\n" + "\n".join(violations)
    )


def test_no_legacy_imports_boundary():
    """_legacy_imports.py should no longer exist — all code is package-native."""
    assert not (PROJECT_ROOT / "open_wam" / "_legacy_imports.py").exists(), (
        "open_wam/_legacy_imports.py still exists — legacy code should be "
        "internalized into the package"
    )


def test_diffsynth_moved_to_third_party():
    """diffsynth/ should no longer exist at project root."""
    assert not (PROJECT_ROOT / "diffsynth").is_dir(), (
        "diffsynth/ still exists at project root — should be third_party/diffsynth/"
    )
    assert (PROJECT_ROOT / "third_party" / "diffsynth").is_dir(), (
        "third_party/diffsynth/ does not exist"
    )


def test_diffsynth_importable_via_qualified_path():
    """diffsynth should be importable via third_party.diffsynth."""
    from third_party.diffsynth import core  # noqa: F401
    from third_party import diffsynth
    assert hasattr(diffsynth, "__file__")
    assert "third_party" in str(Path(diffsynth.__file__).resolve())
