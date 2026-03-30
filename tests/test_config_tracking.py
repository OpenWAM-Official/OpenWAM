"""Tests for run metadata and config snapshot helpers."""

import json
from datetime import datetime, timezone

from open_wam.training.config_tracking import (
    build_run_metadata,
    make_run_id,
    write_run_artifacts,
)


def test_make_run_id_format():
    run_id = make_run_id(datetime(2026, 3, 30, 16, 0, 0, tzinfo=timezone.utc))
    assert run_id == "20260330T160000Z"


def test_build_run_metadata_contains_expected_fields():
    metadata = build_run_metadata(
        run_id="run-123",
        output_dir="./models",
        hydra_output_dir="outputs/2026-03-30_16-00-00",
        argv=["python", "scripts/train.py"],
        git_commit="abc123",
    )
    assert metadata["run_id"] == "run-123"
    assert metadata["output_dir"] == "./models"
    assert metadata["hydra_output_dir"] == "outputs/2026-03-30_16-00-00"
    assert metadata["argv"] == ["python", "scripts/train.py"]
    assert metadata["git_commit"] == "abc123"
    assert metadata["created_at_utc"]


def test_write_run_artifacts(tmp_path):
    written = write_run_artifacts(
        tmp_path,
        resolved_config_yaml="training:\n  learning_rate: 1e-4\n",
        resolved_config_dict={"training": {"learning_rate": 1e-4}},
        flat_args_dict={"learning_rate": 1e-4},
        run_metadata={"run_id": "run-123", "output_dir": "./models"},
    )

    assert (tmp_path / "resolved_config.yaml").exists()
    assert (tmp_path / "resolved_config.json").exists()
    assert (tmp_path / "flat_args.json").exists()
    assert (tmp_path / "run_metadata.json").exists()
    assert set(written) == {
        "resolved_config_yaml",
        "resolved_config_json",
        "flat_args_json",
        "run_metadata_json",
    }

    payload = json.loads((tmp_path / "run_metadata.json").read_text())
    assert payload["run_id"] == "run-123"
