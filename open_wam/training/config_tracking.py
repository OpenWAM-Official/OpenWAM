"""Helpers for saving run metadata and resolved configs."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def make_run_id(now: Optional[datetime] = None) -> str:
    """Create a UTC timestamp-based run id."""
    now = now or datetime.now(timezone.utc)
    return now.strftime("%Y%m%dT%H%M%SZ")


def get_git_commit(cwd: Optional[Path] = None) -> Optional[str]:
    """Return the current git commit hash when available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def build_run_metadata(
    *,
    run_id: str,
    output_dir: str,
    hydra_output_dir: Optional[str] = None,
    argv: Optional[list[str]] = None,
    git_commit: Optional[str] = None,
) -> dict[str, Any]:
    """Build a JSON-serializable run metadata record."""
    return {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": output_dir,
        "hydra_output_dir": hydra_output_dir,
        "argv": list(sys.argv if argv is None else argv),
        "git_commit": git_commit,
    }


def write_run_artifacts(
    artifact_dir: str | Path,
    *,
    resolved_config_yaml: Optional[str] = None,
    resolved_config_dict: Optional[dict[str, Any]] = None,
    flat_args_dict: Optional[dict[str, Any]] = None,
    run_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, str]:
    """Persist resolved config and run metadata files to disk."""
    path = Path(artifact_dir)
    path.mkdir(parents=True, exist_ok=True)

    written: dict[str, str] = {}

    if resolved_config_yaml is not None:
        target = path / "resolved_config.yaml"
        target.write_text(resolved_config_yaml)
        written["resolved_config_yaml"] = str(target)

    if resolved_config_dict is not None:
        target = path / "resolved_config.json"
        target.write_text(json.dumps(resolved_config_dict, indent=2, default=str))
        written["resolved_config_json"] = str(target)

    if flat_args_dict is not None:
        target = path / "flat_args.json"
        target.write_text(json.dumps(flat_args_dict, indent=2, default=str))
        written["flat_args_json"] = str(target)

    if run_metadata is not None:
        target = path / "run_metadata.json"
        target.write_text(json.dumps(run_metadata, indent=2, default=str))
        written["run_metadata_json"] = str(target)

    return written
