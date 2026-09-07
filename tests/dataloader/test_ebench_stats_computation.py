"""Offline EBench stats-computation tests on the synthetic v2.1 bucket.

Contracts pinned here:
  * the parquet scan produces the full six-key schema (incl. TRUE q01/q99)
    with mean/std/min/max exact against numpy ground truth over the reader's
    own raw-23 projection;
  * rot6d dims are pinned to identity (with the --no-rot6d-identity escape
    hatch), base dims keep real stats, and the reader's EBENCH_STD_FLOOR is
    applied (not Accumulator.finalize's 1.0 substitution — GenManip's
    constant commanded base deltas hit exactly that difference);
  * parity with the episodes_stats.jsonl summary merge (the offline module's
    own guardrail baseline) on every stat min-max/z-score consume;
  * the written payload is accepted verbatim by the reader's
    _load_or_build_stats (fingerprint cache-hit, no re-scan) and unlocks
    normalize_mode="quantile" end-to-end, including deploy round trip;
  * quantile off a legacy summary-built cache (no true quantiles) stays
    rejected.

Follows the family test style: compute functions called directly (never
main()/subprocess), fixture bytes replayed for exact expectations.
"""

import math

import numpy as np
import pytest

import openwam.dataloader.ebench as ebench_mod
from openwam.dataloader.ebench import (
    EBENCH_STD_FLOOR,
    EBenchDataset,
    _build_stats_from_bucket,
    _ee_pose_gripper_base_to_raw23,
    _load_or_build_stats,
    _merge_raw_stats,
)
from openwam.dataloader.utils.normalization import ROT6D_DIMS_EEF20
from openwam.dataloader.utils.stats_computation.ebench_stats_computation import (
    build_and_save_ebench_stats,
    compute_ebench_stats,
)
from tests.dataloader.test_ebench_dataset import EP_LEN, N_EPS, _make_frame_row, make_bucket

DELTA_KEYS = ebench_mod.EBENCH_ACTION_KEYS


@pytest.fixture()
def bucket(tmp_path):
    return make_bucket(tmp_path)


@pytest.fixture(autouse=True)
def _mock_video_decode(monkeypatch):
    from PIL import Image

    def fake_decode(path, frame_indices, height, width):
        return [Image.new("RGB", (width, height), (10, 20, 30)) for _ in frame_indices]

    monkeypatch.setattr(ebench_mod, "_decode_video_frames", fake_decode)


@pytest.fixture(autouse=True)
def _clear_parquet_cache():
    ebench_mod._load_parquet_table.cache_clear()
    yield
    ebench_mod._load_parquet_table.cache_clear()


def _expected_raw23(keys) -> np.ndarray:
    """Replay the fixture formula through the reader's own projection."""
    rows = [_make_frame_row(t) for t in range(EP_LEN)]
    ee = np.stack([r[keys[0]] for r in rows])
    grip = np.stack([r[keys[1]] for r in rows])
    base = np.stack([r[keys[2]] for r in rows])
    one_ep = _ee_pose_gripper_base_to_raw23(ee, grip, base)
    return np.concatenate([one_ep] * N_EPS, axis=0)  # every episode is identical


def test_schema_and_exactness(bucket):
    stats, num_timesteps, n_files = compute_ebench_stats([bucket], DELTA_KEYS)
    assert num_timesteps == N_EPS * EP_LEN and n_files == N_EPS
    assert set(stats) == {"mean", "std", "min", "max", "q01", "q99"}
    for key, vec in stats.items():
        assert vec.shape == (23,) and vec.dtype == np.float32, key

    raw = _expected_raw23(DELTA_KEYS).astype(np.float64)
    unpinned = np.ones(23, dtype=bool)
    unpinned[list(ROT6D_DIMS_EEF20)] = False
    np.testing.assert_allclose(stats["mean"][unpinned], raw.mean(axis=0)[unpinned], atol=1e-5)
    np.testing.assert_allclose(
        stats["std"][unpinned], np.maximum(raw.std(axis=0), EBENCH_STD_FLOOR)[unpinned], atol=1e-5
    )
    np.testing.assert_allclose(stats["min"][unpinned], raw.min(axis=0)[unpinned], atol=1e-6)
    np.testing.assert_allclose(stats["max"][unpinned], raw.max(axis=0)[unpinned], atol=1e-6)
    # fixture rows << reservoir cap -> q01/q99 are exact np.quantile
    np.testing.assert_allclose(stats["q01"][unpinned], np.quantile(raw, 0.01, axis=0)[unpinned], atol=1e-5)
    np.testing.assert_allclose(stats["q99"][unpinned], np.quantile(raw, 0.99, axis=0)[unpinned], atol=1e-5)


def test_rot6d_pinned_identity_with_escape_hatch(bucket):
    stats, _, _ = compute_ebench_stats([bucket], DELTA_KEYS)
    for i in ROT6D_DIMS_EEF20:
        assert stats["mean"][i] == 0.0 and stats["std"][i] == 1.0
        assert stats["min"][i] == -1.0 and stats["max"][i] == 1.0
        assert stats["q01"][i] == -1.0 and stats["q99"][i] == 1.0
    unpinned, _, _ = compute_ebench_stats([bucket], DELTA_KEYS, rot6d_identity=False)
    # the fixture's yaw ramp gives real (non-identity) rot6d marginals
    assert not np.allclose(unpinned["mean"][list(ROT6D_DIMS_EEF20)], 0.0)


def test_std_floor_matches_reader_not_accumulator(bucket):
    """The fixture's commanded base delta is constant -> raw std 0. The reader
    floors at EBENCH_STD_FLOOR; Accumulator.finalize would substitute 1.0."""
    stats, _, _ = compute_ebench_stats([bucket], DELTA_KEYS)
    np.testing.assert_allclose(stats["std"][20:23], EBENCH_STD_FLOOR, atol=1e-7)


def test_parity_with_online_summary_merge(bucket):
    """Drop-in contract: offline scan == episodes_stats.jsonl summary merge on
    every stat min-max/z-score consume."""
    offline, _, _ = compute_ebench_stats([bucket], DELTA_KEYS)
    summary = _merge_raw_stats([_build_stats_from_bucket(bucket, DELTA_KEYS)])
    for key in ("mean", "std", "min", "max"):
        np.testing.assert_allclose(offline[key], summary[key], atol=1e-4, err_msg=key)


def test_base_delta_column_statistics(bucket):
    delta, _, _ = compute_ebench_stats([bucket], DELTA_KEYS)
    np.testing.assert_allclose(delta["mean"][20:23], [0.01, -0.005, math.degrees(0.02)], atol=1e-5)


def test_finger_disagreement_fails_dataset_wide(tmp_path):
    """The scan certifies the two-finger invariant on EVERY row — including
    rows beyond the reader's sampled init check."""
    import pandas as pd

    b = make_bucket(tmp_path, "task_fingers", ep_len=100)
    p = b / "data" / "chunk-000" / "episode_000001.parquet"
    df = pd.read_parquet(p)
    bad = np.stack(df["action.gripper"].to_numpy()).copy()
    bad[90, 1] = bad[90, 0] + 0.01  # beyond the leading-64-row sample
    df["action.gripper"] = list(bad)
    df.to_parquet(p)
    with pytest.raises(ValueError, match="finger"):
        compute_ebench_stats([b], DELTA_KEYS)


def test_gripper_range_violation_fails_dataset_wide(tmp_path):
    """The scan certifies EBENCH_GRIPPER_CMD_RANGE on EVERY row the reader
    can serve — including rows beyond the sampled init check (where the
    violation would silently skew min/max stats). Excluding the episode
    restores lenience: it is accumulated, not certified."""
    import json

    import pandas as pd

    b = make_bucket(tmp_path, "task_grip_range", ep_len=100)
    p = b / "data" / "chunk-000" / "episode_000001.parquet"
    df = pd.read_parquet(p)
    bad = np.stack(df["action.gripper"].to_numpy()).copy()
    bad[90, :] = 0.06  # > 0.044 + 1e-4, beyond the leading-64-row sample
    df["action.gripper"] = list(bad)
    df.to_parquet(p)
    with pytest.raises(ValueError, match="outside"):
        compute_ebench_stats([b], DELTA_KEYS)

    (b / "meta" / "excluded_episodes.json").write_text(json.dumps({"episode_indices": [1]}))
    stats, num_timesteps, n_files = compute_ebench_stats([b], DELTA_KEYS)
    assert num_timesteps == 200 and n_files == 2
    assert stats["max"][9] == pytest.approx(0.06)  # excluded rows still pool into stats


def test_excluded_corrupt_episode_does_not_block_scan(tmp_path, capsys):
    """excluded_episodes.json marks episodes precisely BECAUSE they are bad;
    a corrupt or missing excluded episode must warn-skip, never make quantile
    permanently unreachable (the reader trains fine without it). Contract
    violations training tolerates for excluded data (finger gap) are still
    accumulated — only unreadable/non-finite data is dropped."""
    import json

    import pandas as pd

    b = make_bucket(tmp_path, "task_excl", ep_len=100)
    p = b / "data" / "chunk-000" / "episode_000001.parquet"
    (b / "meta" / "excluded_episodes.json").write_text(json.dumps({"episode_indices": [1]}))

    # finger disagreement in the excluded episode: tolerated AND accumulated
    df = pd.read_parquet(p)
    fingers = np.stack(df["action.gripper"].to_numpy()).copy()
    fingers[90, 1] = fingers[90, 0] + 0.02
    df["action.gripper"] = list(fingers)
    df.to_parquet(p)
    _, num_timesteps_all, n_files_all = compute_ebench_stats([b], DELTA_KEYS)
    assert num_timesteps_all == 200 and n_files_all == 2

    # non-finite values in the excluded episode: warn-skip
    df = pd.read_parquet(p)
    ee = np.stack(df["action.ee_pose"].to_numpy()).copy()
    ee[90, 0] = np.nan
    df["action.ee_pose"] = list(ee)
    df.to_parquet(p)
    stats, num_timesteps, n_files = compute_ebench_stats([b], DELTA_KEYS)
    assert num_timesteps == 100 and n_files == 1  # only episode 0 accumulated
    assert "skipping excluded" in capsys.readouterr().out

    # missing parquet for the excluded episode: same tolerance
    p.unlink()
    stats2, num_timesteps2, _ = compute_ebench_stats([b], DELTA_KEYS)
    assert num_timesteps2 == 100
    np.testing.assert_allclose(stats2["mean"], stats["mean"], atol=1e-6)


def test_excluded_healthy_episode_still_accumulated(tmp_path):
    """Coverage parity: the online summary merge pools excluded episodes'
    moments, so a cleanly-readable excluded episode must be accumulated too."""
    b = make_bucket(tmp_path, "task_excl_ok")
    (b / "meta" / "excluded_episodes.json").write_text("[1]")  # bare-list schema
    stats, num_timesteps, n_files = compute_ebench_stats([b], DELTA_KEYS)
    assert num_timesteps == N_EPS * EP_LEN and n_files == N_EPS
    summary = _merge_raw_stats([_build_stats_from_bucket(b, DELTA_KEYS)])
    for key in ("mean", "std", "min", "max"):
        np.testing.assert_allclose(stats[key], summary[key], atol=1e-4, err_msg=key)


def test_payload_accepted_by_reader_and_unlocks_quantile(bucket, monkeypatch):
    root = bucket.parents[1]
    out_path, stats = build_and_save_ebench_stats(str(root))
    assert out_path == root / "meta" / "ebench_normalization_stats.npy"  # the reader's fixed cache location
    payload = np.load(out_path, allow_pickle=True).item()
    assert payload["pool"] == "action" and payload["source"] == "parquet_scan"
    assert "q01" in payload["ebench"] and "q99" in payload["ebench"]

    # The reader must cache-hit the offline file (fingerprint match), never
    # re-running the scan.
    import openwam.dataloader.utils.stats_computation.ebench_stats_computation as stats_mod

    def no_rebuild(*a, **k):
        raise AssertionError("reader re-ran the offline scan despite a valid cache")

    monkeypatch.setattr(stats_mod, "build_and_save_ebench_stats", no_rebuild)
    loaded, path = _load_or_build_stats(
        [bucket], DELTA_KEYS, action_mode="ebench", dataset_dir=str(root), normalize_mode="quantile"
    )
    assert path == str(out_path)
    np.testing.assert_allclose(loaded["q99"], stats["q99"], atol=1e-6)

    ds = EBenchDataset(str(bucket), action_stats=loaded, normalize_mode="quantile", unify_action=False, num_frames=9)
    sample = ds[0]
    assert np.abs(sample["action"].numpy()).max() <= 1.0 + 1e-6


def test_summary_cache_stays_rejected_for_quantile(bucket):
    """A legacy summary-built cache (no q01/q99) at the fixed location must
    not serve quantile."""
    root = bucket.parents[1]
    summary = _merge_raw_stats([_build_stats_from_bucket(bucket, DELTA_KEYS)])
    fingerprint = ebench_mod._stats_fingerprint([bucket], DELTA_KEYS, "ebench", str(root))
    cache = root / "meta" / "ebench_normalization_stats.npy"
    ebench_mod._atomic_save_npy(cache, ebench_mod._stats_cache_payload(summary, N_EPS * EP_LEN, fingerprint, "ebench"))
    with pytest.raises(ValueError, match="q01"):
        _load_or_build_stats(
            [bucket], DELTA_KEYS, action_mode="ebench", dataset_dir=str(root), normalize_mode="quantile"
        )


def test_quantile_train_deploy_round_trip(bucket):
    """Reader quantile normalize -> deploy Normalizer(q99) unnormalize ==
    identity inside the [q01, q99] band (mirrors the min-max round trip).
    No manual pre-step: the loader auto-builds the offline cache itself."""
    from openwam.dataloader.transforms.normalize import YAML_TO_NORM_MODE, Normalizer

    root = bucket.parents[1]
    stats, _ = _load_or_build_stats(
        [bucket], DELTA_KEYS, action_mode="ebench", dataset_dir=str(root), normalize_mode="quantile"
    )
    ds = EBenchDataset(str(bucket), action_stats=stats, normalize_mode="quantile", unify_action=False, num_frames=9)
    rng = np.random.default_rng(7)
    raw = rng.uniform(stats["q01"], stats["q99"], size=(5, 23)).astype(np.float32)
    normalized = ds._normalize(raw, stats)
    deploy = Normalizer(mode=YAML_TO_NORM_MODE["quantile"], stats={"q01": stats["q01"], "q99": stats["q99"]})
    recovered = deploy.unnormalize(normalized)
    np.testing.assert_allclose(recovered, raw, atol=1e-4)
