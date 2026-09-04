"""EBenchDataset behavior tests on a synthetic LeRobot-v2.1 bucket.

Covers the audit-driven fixes: delta base rendering (yaw units,
wrap, episode-start zeros), min-max default + normalize-mode whitelist,
stats index arithmetic and fingerprint (source digest, RO fail-fast),
__getitem__ retry + wrist black-slot tolerance, prompt/bucket fail-fast,
excluded episodes, init-time data-value validation, and the ckpt loader's
explicit-weights existence check. Video decoding is mocked (empty mp4 files
on disk); no model, GPU, or real dataset needed.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import openwam.dataloader.ebench as ebench_mod
from openwam.dataloader.ebench import (
    EBENCH_ACTION_KEYS,
    EBenchDataset,
    _load_or_build_stats,
    _raw_stats_to_23,
    render_ebench_state_base,
    wrap_angle_rad,
)

CAMS = (
    "video.overlook_camera_view",
    "video.left_camera_view",
    "video.right_camera_view",
)
EP_LEN = 40
N_EPS = 2


def _unit_quat_wxyz(yaw: float) -> list:
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def _make_frame_row(t: int) -> dict:
    ee = [0.1 + 0.001 * t, 0.2, 0.3] + _unit_quat_wxyz(0.01 * t) + [-0.1, 0.2, 0.35] + _unit_quat_wxyz(-0.01 * t)
    grip_val = 0.044 if (t // 10) % 2 == 0 else 0.0
    gripper = [grip_val, grip_val, 0.022, 0.022]
    base_state = [0.01 * t, -0.005 * t, 0.02 * t]  # x_m, y_m, yaw_RAD
    base_delta = [0.01, -0.005, math.degrees(0.02)]  # commanded per-step, yaw DEG
    base_cum = [0.01 * t, -0.005 * t, math.degrees(0.02) * t]  # cumsum, yaw DEG
    return {
        "action.ee_pose": np.asarray(ee, dtype=np.float32),
        "action.gripper": np.asarray(gripper, dtype=np.float32),
        "action.base": np.asarray(base_cum, dtype=np.float32),
        "action.base_delta": np.asarray(base_delta, dtype=np.float32),
        "state.ee_pose": np.asarray(ee, dtype=np.float32),
        "state.gripper": np.asarray(gripper, dtype=np.float32),
        "state.base": np.asarray(base_state, dtype=np.float32),
        "task_index": 0,
    }


def _key_stats(rows: np.ndarray) -> dict:
    return {
        "mean": rows.mean(axis=0).tolist(),
        "std": rows.std(axis=0).tolist(),
        "min": rows.min(axis=0).tolist(),
        "max": rows.max(axis=0).tolist(),
        "count": [int(rows.shape[0])],
    }


def make_bucket(
    root: Path, name: str = "task1", *, tasks_text: bool = True, n_eps: int = N_EPS, ep_len: int = EP_LEN
) -> Path:
    bucket = root / "simple_pnp" / name
    (bucket / "meta").mkdir(parents=True)
    (bucket / "data" / "chunk-000").mkdir(parents=True)
    for cam in CAMS:
        (bucket / "videos" / "chunk-000" / cam).mkdir(parents=True)

    features = {
        "action.ee_pose": {"dtype": "float32", "shape": [14]},
        "action.gripper": {"dtype": "float32", "shape": [4]},
        "action.base": {"dtype": "float32", "shape": [3]},
        "action.base_delta": {"dtype": "float32", "shape": [3]},
        "state.ee_pose": {"dtype": "float32", "shape": [14]},
        "state.gripper": {"dtype": "float32", "shape": [4]},
        "state.base": {"dtype": "float32", "shape": [3]},
        "task_index": {"dtype": "int64", "shape": [1]},
    }
    for cam in CAMS:
        features[cam] = {"dtype": "video", "shape": [480, 640, 3]}
    info = {
        "codebase_version": "v2.1",
        "fps": 15,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
        "splits": {"train": f"0:{n_eps}"},
    }
    (bucket / "meta" / "info.json").write_text(json.dumps(info))

    episodes, stats_rows = [], []
    for ep in range(n_eps):
        rows = [_make_frame_row(t) for t in range(ep_len)]
        df = pd.DataFrame(rows)
        df.to_parquet(bucket / "data" / "chunk-000" / f"episode_{ep:06d}.parquet")
        for cam in CAMS:
            (bucket / "videos" / "chunk-000" / cam / f"episode_{ep:06d}.mp4").write_bytes(b"")
        episodes.append(
            {
                "episode_index": ep,
                "length": ep_len,
                "tasks": ["pick the apple"] if tasks_text else [],
            }
        )
        stats = {
            key: _key_stats(np.stack([r[key] for r in rows]))
            for key in (
                "action.ee_pose",
                "action.gripper",
                "action.base",
                "action.base_delta",
                "state.ee_pose",
                "state.gripper",
                "state.base",
            )
        }
        stats_rows.append({"episode_index": ep, "stats": stats})

    with (bucket / "meta" / "episodes.jsonl").open("w") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")
    with (bucket / "meta" / "episodes_stats.jsonl").open("w") as f:
        for row in stats_rows:
            f.write(json.dumps(row) + "\n")
    tasks = [{"task_index": 0, "task": "pick the apple"}] if tasks_text else []
    with (bucket / "meta" / "tasks.jsonl").open("w") as f:
        for row in tasks:
            f.write(json.dumps(row) + "\n")
    return bucket


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


def _make_ds(bucket, **kw):
    kw.setdefault("num_frames", 9)
    kw.setdefault("video_stride", 4)
    # Mirror from_config: stats only exist when normalization is on. The
    # auto-build is a full parquet scan, so corrupt-data tests that target
    # __getitem__ must construct with normalize_mode=None to get past it.
    stats = None
    if kw.get("normalize_mode", "min-max") not in (None, "none", "null"):
        stats, _ = _load_or_build_stats(
            [bucket],
            EBENCH_ACTION_KEYS,
            action_mode="ebench",
            dataset_dir=str(bucket.parents[1]),
        )
    return EBenchDataset(str(bucket), action_stats=stats, **kw)


# ---------------------------------------------------------------- base semantics


def test_wrap_angle_rad():
    assert wrap_angle_rad(np.pi) == pytest.approx(-np.pi)
    assert wrap_angle_rad(3 * np.pi / 2) == pytest.approx(-np.pi / 2)
    assert wrap_angle_rad(-0.1) == pytest.approx(-0.1)


def test_render_state_base_delta():
    cur, prev = np.array([1.0, 2.0, 0.1]), np.array([0.9, 2.1, 0.05])
    d = render_ebench_state_base(cur, prev)
    np.testing.assert_allclose(d, [0.1, -0.1, math.degrees(0.05)], atol=1e-6)
    assert render_ebench_state_base(cur, None) == pytest.approx([0, 0, 0])
    # wrap across the +/-pi seam: -3.1 -> 3.1 is a -(2pi-6.2) rotation
    w = render_ebench_state_base(np.array([0, 0, 3.1]), np.array([0, 0, -3.1]))
    assert w[2] == pytest.approx(math.degrees(6.2 - 2 * np.pi), abs=1e-4)


def test_delta_proprio_uses_measured_diff_in_degrees(bucket):
    ds = _make_ds(bucket, normalize_mode=None, unify_action=False)
    # idx>0 windows have a leading row: measured diff of the synthetic state
    # ramp is exactly [0.01, -0.005, deg(0.02)] per step.
    sample = ds[3]
    got = sample["proprio"].numpy()[0, 20:23]
    np.testing.assert_allclose(got, [0.01, -0.005, math.degrees(0.02)], atol=1e-5)
    # episode-start window: no previous frame -> zeros (GenManip convention)
    start = ds[0]["proprio"].numpy()[0, 20:23]
    np.testing.assert_allclose(start, [0.0, 0.0, 0.0], atol=1e-7)


def test_delta_proprio_matches_action_space_stats(bucket):
    # With normalization ON, delta proprio base must land in the same
    # normalized range as the action.base_delta targets (shared stats).
    ds = _make_ds(bucket, normalize_mode="min-max", unify_action=False)
    s = ds[5]
    assert np.abs(s["proprio"].numpy()).max() <= 1.0 + 1e-6
    assert np.abs(s["action"].numpy()).max() <= 1.0 + 1e-6


# ---------------------------------------------------------------- normalization


def test_default_mode_is_min_max_and_quantile_works_off_auto_build(bucket):
    """The auto-built offline-scan cache carries true q01/q99, so quantile
    works end-to-end with no manual pre-step; direct construction without
    stats and misspelled modes stay rejected at init."""
    import inspect

    assert inspect.signature(EBenchDataset.__init__).parameters["normalize_mode"].default == "min-max"
    # Direct construction without stats must be rejected AT INIT (the
    # pre-change whitelist guarantee) — not deferred to __getitem__ retries.
    with pytest.raises(ValueError, match="ebench_stats_computation"):
        EBenchDataset(str(bucket), normalize_mode="quantile", num_frames=9)
    ds = _make_ds(bucket, normalize_mode="quantile", unify_action=False)
    assert np.abs(ds[0]["action"].numpy()).max() <= 1.0 + 1e-6
    with pytest.raises(ValueError, match="normalize_mode"):
        _make_ds(bucket, normalize_mode="zscore")  # misspelling must not pass through


def test_auto_built_cache_is_offline_scan_with_true_quantiles(bucket):
    stats, path = _load_or_build_stats(
        [bucket],
        ebench_mod.EBENCH_ACTION_KEYS,
        action_mode="ebench",
        dataset_dir=str(bucket.parents[1]),
    )
    assert path == str(bucket.parents[1] / "meta" / "ebench_normalization_stats.npy")
    assert "q01" in stats and "q99" in stats
    payload = np.load(path, allow_pickle=True).item()
    assert payload["source"] == "parquet_scan"
    assert "q01" in payload["ebench"]


def test_raw_stats_to_23_index_arithmetic():
    stats_by_key = {
        "action.ee_pose": {
            "mean": np.arange(14, dtype=np.float32),
            "std": np.ones(14, np.float32),
            "min": np.zeros(14, np.float32),
            "max": np.ones(14, np.float32),
        },
        "action.gripper": {
            "mean": np.array([0.1, 0.1, 0.3, 0.3], np.float32),
            "std": np.ones(4, np.float32),
            "min": np.zeros(4, np.float32),
            "max": np.full(4, 0.044, np.float32),
        },
        "action.base_delta": {
            "mean": np.array([1.0, 2.0, 3.0], np.float32),
            "std": np.ones(3, np.float32),
            "min": -np.ones(3, np.float32),
            "max": np.ones(3, np.float32),
        },
    }
    out = _raw_stats_to_23(stats_by_key, ("action.ee_pose", "action.gripper", "action.base_delta"))
    np.testing.assert_allclose(out["mean"][0:3], [0, 1, 2])  # L xyz <- ee[0:3]
    np.testing.assert_allclose(out["mean"][10:13], [7, 8, 9])  # R xyz <- ee[7:10]
    assert out["mean"][9] == pytest.approx(0.1) and out["mean"][19] == pytest.approx(0.3)
    np.testing.assert_allclose(out["mean"][20:23], [1, 2, 3])
    # rot6d identity pinning
    np.testing.assert_allclose(out["mean"][3:9], 0.0)
    np.testing.assert_allclose(out["max"][13:19], 1.0)
    assert "q01" not in out


def test_fingerprint_invalidates_on_source_change(bucket):
    root = str(bucket.parents[1])
    keys = ebench_mod.EBENCH_ACTION_KEYS
    _load_or_build_stats([bucket], keys, action_mode="ebench", dataset_dir=root)
    # cache hit with unchanged source — the scan must not re-run
    cache = bucket.parents[1] / "meta" / "ebench_normalization_stats.npy"
    mtime = cache.stat().st_mtime_ns
    _load_or_build_stats([bucket], keys, action_mode="ebench", dataset_dir=root)
    assert cache.stat().st_mtime_ns == mtime
    # in-place dataset update -> digest change -> hard error
    stats_file = bucket / "meta" / "episodes_stats.jsonl"
    stats_file.write_text(stats_file.read_text() + "\n")
    with pytest.raises(ValueError, match="fingerprint"):
        _load_or_build_stats([bucket], keys, action_mode="ebench", dataset_dir=root)


def test_stats_write_failure_fails_fast(bucket, monkeypatch):
    """The old write-degrade path is gone: the cache location is fixed under
    the dataset dir, so an unwritable mount must fail construction (deploy
    needs the file, and other ranks would poll it forever)."""
    import openwam.dataloader.utils.stats_computation.ebench_stats_computation as stats_mod

    def refuse(path, payload):
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(stats_mod, "_atomic_save_npy", refuse)
    with pytest.raises(OSError, match="Read-only"):
        _load_or_build_stats(
            [bucket],
            ebench_mod.EBENCH_ACTION_KEYS,
            action_mode="ebench",
            dataset_dir=str(bucket.parents[1]),
        )


def test_non_finite_values_raise_not_zeroed(bucket):
    ds = _make_ds(bucket, normalize_mode="min-max", unify_action=False)
    corrupt = np.full((1, 23), 5.0, np.float32)
    corrupt[0, 7] = np.nan  # a corrupt parquet value must surface, not train as 0
    with pytest.raises(ValueError, match="non-finite"):
        ds._normalize(corrupt, ds._action_stats)


# ---------------------------------------------------------------- robustness


def test_getitem_retry_walks_to_next_window(bucket, monkeypatch):
    ds = _make_ds(bucket, normalize_mode=None, unify_action=False)
    calls = {"n": 0}
    orig = ds._getitem_impl

    def flaky(idx):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient decode failure")
        return orig(idx)

    monkeypatch.setattr(ds, "_getitem_impl", flaky)
    sample = ds[0]
    assert calls["n"] == 2 and sample["action"].shape[-1] == 23


def test_getitem_retry_exhaustion_raises(bucket, monkeypatch):
    ds = _make_ds(bucket, normalize_mode=None, unify_action=False)
    monkeypatch.setattr(ds, "_getitem_impl", lambda idx: (_ for _ in ()).throw(RuntimeError("always broken")))
    with pytest.raises(RuntimeError, match="always broken"):
        ds[0]


def test_wrist_decode_failure_black_slot_head_fatal(bucket, monkeypatch):
    ds = _make_ds(bucket, normalize_mode=None, unify_action=False)
    from PIL import Image

    def decode_selective(path, frame_indices, height, width):
        if "left_camera" in str(path):
            raise RuntimeError("corrupt wrist mp4")
        return [Image.new("RGB", (width, height), (10, 20, 30)) for _ in frame_indices]

    monkeypatch.setattr(ebench_mod, "_decode_video_frames", decode_selective)
    sample = ds[0]  # left wrist black slot, sample still valid
    assert len(sample["video"]) == ds._num_video_frames

    def decode_head_broken(path, frame_indices, height, width):
        if "overlook" in str(path):
            raise RuntimeError("corrupt head mp4")
        return [Image.new("RGB", (width, height), (10, 20, 30)) for _ in frame_indices]

    monkeypatch.setattr(ebench_mod, "_decode_video_frames", decode_head_broken)
    with pytest.raises(RuntimeError, match="corrupt head"):
        ds[0]


def test_video_indices_pure_arange(bucket):
    ds = _make_ds(bucket, normalize_mode=None, unify_action=False, num_frames=9, video_stride=4)
    np.testing.assert_array_equal(ds._video_sample_indices, [0, 4, 8])
    ds2 = _make_ds(bucket, normalize_mode=None, unify_action=False, num_frames=9, video_stride=5)
    np.testing.assert_array_equal(ds2._video_sample_indices, [0, 5])  # no last-frame append


def test_color_jitter_train_only(bucket):
    """color_jitter mirrors LeRobotV3Reader: train-split only, one factor set
    per clip. The mocked decoder returns identical constant-color frames, so
    (a) any pixel change proves the jitter ran, and (b) jittered frames staying
    identical to each other proves temporal consistency."""
    import random

    cj = {"brightness": 0.5, "contrast": 0.3, "saturation": 0.3, "hue": 0.0}
    base = np.asarray(_make_ds(bucket, normalize_mode=None, unify_action=False)[0]["video"][0])
    random.seed(123)
    jit = _make_ds(bucket, normalize_mode=None, unify_action=False, color_jitter=cj)[0]["video"]
    assert not np.array_equal(np.asarray(jit[0]), base), "train-split jitter must alter pixels"
    np.testing.assert_array_equal(np.asarray(jit[0]), np.asarray(jit[-1]))
    # val split and the disabled default keep video byte-identical
    # (the synthetic bucket only declares a train split; add val for the check)
    info_p = bucket / "meta" / "info.json"
    info = json.loads(info_p.read_text())
    info["splits"]["val"] = f"0:{N_EPS}"
    info_p.write_text(json.dumps(info))
    val = _make_ds(bucket, normalize_mode=None, unify_action=False, color_jitter=cj, split="val")
    assert val._color_jitter is None
    np.testing.assert_array_equal(np.asarray(val[0]["video"][0]), base)
    assert _make_ds(bucket, normalize_mode=None, unify_action=False)._color_jitter is None


# ---------------------------------------------------------------- fail-fast paths


def test_prompt_failfast_at_init(tmp_path):
    b = make_bucket(tmp_path, "task_noprompt", tasks_text=False)
    with pytest.raises(ValueError, match="no task text"):
        _make_ds(b, normalize_mode=None, unify_action=False)


def test_excluded_episodes_honored(bucket):
    (bucket / "meta" / "excluded_episodes.json").write_text("[0]")
    ds = _make_ds(bucket, normalize_mode=None, unify_action=False)
    assert {int(ep["episode_index"]) for ep in ds._episodes} == {1}


def test_init_rejects_bad_quaternions(tmp_path):
    b = make_bucket(tmp_path, "task_badquat")
    p = b / "data" / "chunk-000" / "episode_000000.parquet"
    df = pd.read_parquet(p)
    bad = np.stack(df["action.ee_pose"].to_numpy()).copy()
    bad[:, 3:7] *= 3.0  # non-unit quats
    df["action.ee_pose"] = list(bad)
    df.to_parquet(p)
    with pytest.raises(ValueError):
        _make_ds(b, normalize_mode=None, unify_action=False)


def test_init_rejects_finger_disagreement(tmp_path):
    b = make_bucket(tmp_path, "task_fingers")
    p = b / "data" / "chunk-000" / "episode_000000.parquet"
    df = pd.read_parquet(p)
    bad = np.stack(df["action.gripper"].to_numpy()).copy()
    bad[:, 1] = bad[:, 0] + 0.01  # fingers disagree
    df["action.gripper"] = list(bad)
    df.to_parquet(p)
    with pytest.raises(ValueError, match="finger"):
        _make_ds(b, normalize_mode=None, unify_action=False)


def test_normalization_stats_property_is_none(bucket):
    ds = _make_ds(bucket, normalize_mode="min-max", unify_action=False)
    assert ds.normalization_stats is None


# ---------------------------------------------------------------- ckpt loader


def test_explicit_safetensors_missing_fails_fast(tmp_path):
    from openwam.train.utils.ckpt_model_loader import _resolve_ckpt_source

    with pytest.raises(FileNotFoundError, match="does not exist"):
        _resolve_ckpt_source(str(tmp_path / "nope.safetensors"))
    real = tmp_path / "checkpoint_step_10.safetensors"
    real.write_bytes(b"x")
    ckpt_dir, weights = _resolve_ckpt_source(str(real))
    assert ckpt_dir == str(tmp_path) and weights == str(real)
    # directories pass through untouched
    assert _resolve_ckpt_source(str(tmp_path)) == (str(tmp_path), None)


# ---------------------------------------------------------------- gate-2 fixes


def test_data_error_not_masked_by_retry(tmp_path):
    """A deterministic NaN in the data must surface immediately (EBenchDataError),
    not be retried into a different window. The NaN sits beyond the init
    validation's leading-64-row sample, so it is only hit at __getitem__."""
    b = make_bucket(tmp_path, "task_nanlate", ep_len=100)
    p = b / "data" / "chunk-000" / "episode_000000.parquet"
    df = pd.read_parquet(p)
    bad = np.stack(df["action.ee_pose"].to_numpy()).copy()
    bad[70, 0] = np.nan
    df["action.ee_pose"] = list(bad)
    df.to_parquet(p)

    ds = _make_ds(b, normalize_mode=None, unify_action=False)
    calls = {"n": 0}
    orig = ds._getitem_impl

    def counting(idx):
        calls["n"] += 1
        return orig(idx)

    ds._getitem_impl = counting
    with pytest.raises(ebench_mod.EBenchDataError, match="non-finite"):
        ds[68]  # window offset 68 covers row 70
    assert calls["n"] == 1  # no retry churn


def test_nan_raises_even_without_normalization(bucket):
    ds = _make_ds(bucket, normalize_mode=None, unify_action=False)
    corrupt = np.full((1, 23), 1.0, np.float32)
    corrupt[0, 3] = np.inf
    with pytest.raises(ebench_mod.EBenchDataError, match="non-finite"):
        ds._normalize(corrupt, None)


def test_retry_jumps_over_long_bad_episode(bucket, monkeypatch):
    """One fully-broken episode mp4 must not consume all 64 retries: the walk
    jumps to the NEXT episode's first window and succeeds."""
    from PIL import Image

    def decode_ep0_broken(path, frame_indices, height, width):
        if "episode_000000" in str(path):
            raise RuntimeError("episode 0 mp4 truncated")
        return [Image.new("RGB", (width, height), (10, 20, 30)) for _ in frame_indices]

    monkeypatch.setattr(ebench_mod, "_decode_video_frames", decode_ep0_broken)
    ds = _make_ds(bucket, normalize_mode=None, unify_action=False)
    calls = {"n": 0}
    orig = ds._getitem_impl

    def counting(idx):
        calls["n"] += 1
        return orig(idx)

    ds._getitem_impl = counting
    sample = ds[0]  # a window of episode 0 -> retried onto episode 1
    assert calls["n"] == 2
    assert sample["action"].shape[-1] == 23


def test_excluded_episodes_scanner_schema(bucket):
    (bucket / "meta" / "excluded_episodes.json").write_text('{"episode_indices": [1]}')
    ds = _make_ds(bucket, normalize_mode=None, unify_action=False)
    assert {int(ep["episode_index"]) for ep in ds._episodes} == {0}


def test_stats_cache_survives_dataset_move(bucket, tmp_path):
    import shutil

    root_a = bucket.parents[1]
    keys = ebench_mod.EBENCH_ACTION_KEYS
    _load_or_build_stats([bucket], keys, action_mode="ebench", dataset_dir=str(root_a))
    root_b = tmp_path / "moved_root"
    shutil.copytree(root_a, root_b)
    moved_bucket = root_b / "simple_pnp" / "task1"
    cache_b = root_b / "meta" / "ebench_normalization_stats.npy"
    mtime = cache_b.stat().st_mtime_ns
    # same relative path + same bytes at a different mount -> cache hit, no rebuild
    stats, path = _load_or_build_stats([moved_bucket], keys, action_mode="ebench", dataset_dir=str(root_b))
    assert path == str(cache_b)
    assert cache_b.stat().st_mtime_ns == mtime


def test_min_max_train_deploy_round_trip(bucket):
    """Reader normalize -> deploy Normalizer(min_max) unnormalize == identity,
    including constant (min==max) dims."""
    from openwam.dataloader.transforms.normalize import Normalizer

    ds = _make_ds(bucket, normalize_mode="min-max", unify_action=False)
    stats = ds._action_stats
    rng = np.random.default_rng(7)
    raw = rng.uniform(stats["min"], stats["max"], size=(5, 23)).astype(np.float32)
    normalized = ds._normalize(raw, stats)
    deploy = Normalizer(mode="min_max", stats={"min": stats["min"], "max": stats["max"]})
    recovered = deploy.unnormalize(normalized)
    np.testing.assert_allclose(recovered, raw, atol=1e-4)


def test_validation_covers_multiple_episodes(tmp_path):
    """A defect only in the LAST episode must be caught at init."""
    b = make_bucket(tmp_path, "task_lastbad", n_eps=3)
    p = b / "data" / "chunk-000" / "episode_000002.parquet"
    df = pd.read_parquet(p)
    bad = np.stack(df["action.ee_pose"].to_numpy()).copy()
    bad[:, 3:7] *= 5.0
    df["action.ee_pose"] = list(bad)
    df.to_parquet(p)
    with pytest.raises(ValueError):
        _make_ds(b, normalize_mode=None, unify_action=False)


def test_validation_checks_base_finite(tmp_path):
    b = make_bucket(tmp_path, "task_badbase")
    p = b / "data" / "chunk-000" / "episode_000000.parquet"
    df = pd.read_parquet(p)
    bad = np.stack(df["state.base"].to_numpy()).copy()
    bad[2, 1] = np.nan
    df["state.base"] = list(bad)
    df.to_parquet(p)
    with pytest.raises(ValueError, match="non-finite"):
        _make_ds(b, normalize_mode=None, unify_action=False)


def test_deploy_normalizer_min_max_clips_and_handles_large_constants():
    """Gate-3 critical fix: deploy Normalizer(min_max).normalize must be
    bit-parallel to training's clipped apply_normalization — including
    out-of-range eval jitter (clip to ±1, not thousands of sigma) and large
    constant dims (float32 offset absorption previously gave -1.907 / 0.0)."""
    from openwam.dataloader.transforms.normalize import Normalizer
    from openwam.dataloader.utils.normalization import apply_normalization

    lo = np.array([0.0, 0.044, 12.0, 90.0, 0.1], np.float32)
    hi = np.array([0.0, 0.044, 12.0, 90.0, 0.1002], np.float32)
    deploy = Normalizer(mode="min_max", stats={"min": lo, "max": hi})
    stats = {"min": lo, "max": hi, "mean": lo, "std": np.ones(5, np.float32)}
    probes = [
        lo,
        hi,
        lo + 0.015,  # one legal base step of jitter on constant dims
        np.array([1e-4, 0.02, 11.5, 90.006, 0.2], np.float32),
    ]
    for x in probes:
        train = apply_normalization(x[None, :].astype(np.float32), stats, "min-max")[0]
        got = deploy.normalize(x.astype(np.float32))
        np.testing.assert_allclose(got, train, atol=1e-5)
        assert np.abs(got).max() <= 1.0 + 1e-6  # bounded like training
    # unnormalize still recovers the constants exactly
    rec = deploy.unnormalize(np.array([-1.0, -1.0, -1.0, -1.0, 0.3], np.float32))
    np.testing.assert_allclose(rec[:4], lo[:4], atol=1e-4)
