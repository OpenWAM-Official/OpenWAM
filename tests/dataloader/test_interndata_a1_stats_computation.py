"""Tests for the InternData-A1 stats script.

(openwam/dataloader/utils/stats_computation/interndata_a1_stats_computation.py)

Mirrors the coverage its siblings already have (test_ebench_stats_computation.py,
test_robocoin_compute_stats.py) and pins the three behaviours the script's own
comments flag but nothing enforced:

1. **Reader/stats parity.** ``_SIDES`` / ``_arm10`` / ``_eef20`` are duplicated
   from the reader ("must stay bit-identical to ``InternDataA1Dataset._eef20``"),
   so an edit to either copy would surface only as silently skewed normalization
   — the stats would describe a distribution the reader never emits.
2. **Identity pinning at generation time.** ``materialize_eef_stats`` only
   *warns* on an unpinned file, so dropping ``pin_rot6d_identity`` would distort
   every rotation with a green suite. Same for the single-arm right-half pin
   that keeps franka's padding at exactly 0 instead of the -1 boundary.
3. **Merge determinism.** q01/q99 come from a seeded reservoir whose surviving
   rows depend on insertion order, so the parent must merge in a fixed order.

The bucket builder is shared with the reader tests (same precedent as
test_ebench_stats_computation.py importing from test_ebench_dataset).
"""

from __future__ import annotations

import json
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

import openwam.dataloader.utils.stats_computation.interndata_a1_stats_computation as a1s
from openwam.dataloader.interndata_a1 import (
    _BIMANUAL_SIDES,
    _SINGLE_ARM_SIDES,
    InternDataA1Dataset,
    resolve_gripper_scale,
)
from openwam.dataloader.utils.lerobotv3 import DataContractError
from openwam.dataloader.utils.normalization import ROT6D_DIMS_EEF20
from openwam.dataloader.utils.stats_computation.robocoin_stats_computation import Accumulator
from tests.dataloader.test_interndata_a1 import _make_bucket

BIMANUAL = ("bimanual", "AgileX Split Aloha", "split_aloha")
SINGLE_ARM = ("single_arm", "Franka", "franka")


def _group(dirs, layout: str, robot_type: str) -> dict:
    return {"robot_type": robot_type, "arm_layout": layout, "dirs": list(dirs)}


class _InlinePool:
    """ProcessPoolExecutor stub that runs each task inline at submit time.

    Keeps ``_scan_bucket`` in-process (covered, and no process-spawn cost) and
    leaves every future already-done by the time the merge loop runs — which is
    exactly the state in which ``as_completed``'s yield order stops tracking
    submission order.
    """

    def __init__(self, max_workers=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, arg):
        fut: Future = Future()
        try:
            fut.set_result(fn(arg))
        except BaseException as e:  # noqa: BLE001 - mirror the real pool's contract
            fut.set_exception(e)
        return fut


@pytest.fixture
def inline_pool(monkeypatch):
    monkeypatch.setattr(a1s, "ProcessPoolExecutor", _InlinePool)


# ---------------------------------------------------------------------------
# 1. Reader / stats parity
# ---------------------------------------------------------------------------


class TestReaderParity:
    """The stats script and the reader must assemble byte-identical vectors."""

    @pytest.mark.parametrize("layout,robot_type,_emb", [BIMANUAL, SINGLE_ARM])
    @pytest.mark.parametrize("kind", ["action", "state"])
    def test_eef20_is_bit_identical_to_the_reader(self, tmp_path, layout, robot_type, _emb, kind):
        d = _make_bucket(tmp_path, "cat/emb/task", layout=layout, robot_type=robot_type)
        ds = InternDataA1Dataset(str(d), normalize_mode=None, num_frames=9, video_stride=4)
        win = ds._load_data_table(0, 0).to_pandas()
        reader_out = ds._eef20(win, kind, len(win))

        sides = a1s._SIDES[layout]
        table = pq.read_table(d / "data" / "chunk-000" / "file-000.parquet")
        grip_scales = tuple(
            resolve_gripper_scale(d, _emb, spec[1]) if spec is not None else 1.0 for spec in sides["state"]
        )
        stats_out = a1s._eef20(table, sides, kind, grip_scales)

        assert stats_out.shape == reader_out.shape
        np.testing.assert_array_equal(stats_out, reader_out)

    def test_gripper_rescale_is_shared_not_reimplemented(self, tmp_path):
        """A franka Robotiq bucket: both sides must pick the same 1.0 stroke, or
        the stats describe a distribution 12.5x off what the reader emits."""
        d = _make_bucket(tmp_path, "cat/franka/task", layout="single_arm", robot_type="Franka")
        (d / "meta" / "stats.json").write_text(
            json.dumps({"states.gripper.position": {"min": [0.0], "max": [1.0], "mean": [0.45]}})
        )
        ds = InternDataA1Dataset(str(d), normalize_mode=None, num_frames=9, video_stride=4)
        win = ds._load_data_table(0, 0).to_pandas()
        sides = a1s._SIDES["single_arm"]
        grip_scales = tuple(
            resolve_gripper_scale(d, "franka", spec[1]) if spec is not None else 1.0 for spec in sides["state"]
        )
        assert grip_scales[0] == pytest.approx(ds._grip_scale[0]) == 1.0
        table = pq.read_table(d / "data" / "chunk-000" / "file-000.parquet")
        np.testing.assert_array_equal(
            a1s._eef20(table, sides, "action", grip_scales), ds._eef20(win, "action", len(win))
        )

    def test_each_side_gets_its_own_scale_in_both_implementations(self, tmp_path, monkeypatch):
        """Every bimanual embodiment currently declares one stroke, so the two
        sides always resolve to the SAME divisor — which means a left/right
        scale mix-up in either `_eef20` is invisible to the parity test above.
        Force distinct per-side scales and assert each arm's ABSOLUTE value, so a
        mix-up applied to both copies at once — which the parity assert cannot
        see, since it only compares the two against each other — still fails."""
        d = _make_bucket(tmp_path, "cat/emb/task")
        scales = (0.1, 0.4)
        ds = InternDataA1Dataset(str(d), normalize_mode=None, num_frames=9, video_stride=4)
        monkeypatch.setattr(ds, "_grip_scale", scales)
        win = ds._load_data_table(0, 0).to_pandas()
        table = pq.read_table(d / "data" / "chunk-000" / "file-000.parquet")

        reader_out = ds._eef20(win, "action", len(win))
        stats_out = a1s._eef20(table, a1s._SIDES["bimanual"], "action", scales)
        np.testing.assert_array_equal(stats_out, reader_out)

        # Each arm's gripper must be its OWN raw column over its OWN divisor.
        # (Asserting only that cols 9 and 19 differ is unfalsifiable here:
        # _make_bucket draws each side's gripper independently, so they differ
        # whatever the routing does.)
        raw = {
            side: np.stack(win[f"actions.{side}_gripper.position"].values).astype(np.float32).ravel()
            for side in ("left", "right")
        }
        for col, side, scale in ((9, "left", scales[0]), (19, "right", scales[1])):
            np.testing.assert_allclose(reader_out[:, col], raw[side] / scale, rtol=0, atol=1e-6)
        # Sanity: the two divisors are distinct, so the assertions above are not
        # accidentally identical.
        assert scales[0] != scales[1]

    def test_sides_tables_are_the_readers_own(self):
        """The column tables are no longer restated in the stats script — it
        imports the reader's. `is` rather than `==`: a re-introduced local copy
        would compare equal on the day it was written and drift later, which is
        exactly the failure the script's old "must stay in sync" comment named."""
        assert a1s._SIDES["bimanual"] is _BIMANUAL_SIDES
        assert a1s._SIDES["single_arm"] is _SINGLE_ARM_SIDES


# ---------------------------------------------------------------------------
# 2. Identity pinning at generation time
# ---------------------------------------------------------------------------


class TestIdentityPinning:
    def test_rot6d_dims_are_pinned(self, tmp_path, inline_pool):
        d = _make_bucket(tmp_path, "cat/emb/task")
        out = a1s.compute_stats_for_embodiment("split_aloha", _group([d], "bimanual", "AgileX Split Aloha"))
        eef = out["eef"]
        assert out["rot6d_identity"] is True
        for i in ROT6D_DIMS_EEF20:
            assert (eef["min"][i], eef["max"][i]) == (-1.0, 1.0)
            assert (eef["q01"][i], eef["q99"][i]) == (-1.0, 1.0)
            assert (eef["mean"][i], eef["std"][i]) == (0.0, 1.0)

    def test_no_rot6d_identity_leaves_the_measured_values(self, tmp_path, inline_pool):
        d = _make_bucket(tmp_path, "cat/emb/task")
        out = a1s.compute_stats_for_embodiment(
            "split_aloha", _group([d], "bimanual", "AgileX Split Aloha"), rot6d_identity=False
        )
        eef = out["eef"]
        assert out["rot6d_identity"] is False
        # Random unit quaternions never produce exactly +-1 on all 12 rot6d dims.
        assert not all(eef["q01"][i] == -1.0 and eef["q99"][i] == 1.0 for i in ROT6D_DIMS_EEF20)

    def test_single_arm_right_half_is_pinned_so_padding_stays_zero(self, tmp_path, inline_pool):
        """franka fills [0:10) only. Without the right-half pin the zero padding
        would normalize onto the -1 boundary instead of staying at 0."""
        d = _make_bucket(tmp_path, "cat/franka/task", layout="single_arm", robot_type="Franka")
        out = a1s.compute_stats_for_embodiment("franka", _group([d], "single_arm", "Franka"))
        eef = out["eef"]
        assert out["arm_layout"] == "single_arm"
        for i in range(10, 20):  # xyz, rot6d AND gripper of the padded right arm
            assert (eef["q01"][i], eef["q99"][i]) == (-1.0, 1.0)
            assert (eef["mean"][i], eef["std"][i]) == (0.0, 1.0)
        # quantile-normalizing a 0 with q01/q99 = -/+1 leaves it at 0.
        assert (0.0 - eef["q01"][10]) / (eef["q99"][10] - eef["q01"][10]) * 2 - 1 == 0.0

    def test_bimanual_right_half_is_data_not_padding(self, tmp_path, inline_pool):
        """The right-half pin must fire for single_arm ONLY — pinning a bimanual
        bucket's real right arm would disable its normalization."""
        d = _make_bucket(tmp_path, "cat/emb/task")
        eef = a1s.compute_stats_for_embodiment("split_aloha", _group([d], "bimanual", "AgileX Split Aloha"))["eef"]
        assert (eef["q01"][10], eef["q99"][10]) != (-1.0, 1.0)  # right xyz
        assert (eef["q01"][19], eef["q99"][19]) != (-1.0, 1.0)  # right gripper

    def test_written_file_round_trips_into_the_reader(self, tmp_path, inline_pool, monkeypatch):
        """End-to-end: generated stats must satisfy the reader's own width check
        and load under the default quantile mode."""
        d = _make_bucket(tmp_path, "cat/split_aloha/task")
        monkeypatch.setattr("sys.argv", ["prog", "--dataset_dir", str(tmp_path)])
        a1s.main()
        ds = InternDataA1Dataset(
            str(d), a1_stats_root=str(tmp_path), normalize_mode="quantile", num_frames=9, video_stride=4
        )
        assert ds._normalization_stats is not None


# ---------------------------------------------------------------------------
# 3. Merge determinism
# ---------------------------------------------------------------------------


class TestMergeDeterminism:
    """q01/q99 come from a seeded Algorithm-R reservoir: which rows survive
    depends on INSERTION order, so consuming as_completed made the file drift run
    to run on identical data (mean/std/min/max are order-invariant and stayed
    put, which is what let it go unnoticed)."""

    @staticmethod
    def _small_acc_factory(cap: int):
        class _SmallAccumulator(Accumulator):
            def __init__(self, dim: int = 20, **kw):
                super().__init__(dim=dim, reservoir_cap=cap, seed=0)

        return _SmallAccumulator

    def _buckets(self, tmp_path):
        # Distinct row counts so the merge order is identifiable from batch sizes.
        return [
            _make_bucket(tmp_path, f"cat/emb/task{i}", n_eps=1, ep_len=ep, seed=i)
            for i, ep in enumerate((10, 12, 14, 16, 18, 20))
        ]

    def test_merges_in_submission_order_not_completion_order(self, tmp_path, inline_pool, monkeypatch):
        dirs = self._buckets(tmp_path)
        seen: list[int] = []
        base = self._small_acc_factory(64)

        class _Recording(base):  # type: ignore[valid-type,misc]
            def update_batch(self, batch):
                seen.append(len(batch))
                return super().update_batch(batch)

        monkeypatch.setattr(a1s, "Accumulator", _Recording)
        a1s.compute_stats_for_embodiment("split_aloha", _group(dirs, "bimanual", "AgileX Split Aloha"))
        # _scan_bucket returns action rows ++ state rows -> 2 * ep_len per bucket,
        # and dirs are submitted in the order given.
        assert seen == [2 * ep for ep in (10, 12, 14, 16, 18, 20)]

    def test_quantiles_match_a_sequential_merge_of_the_same_buckets(self, tmp_path, inline_pool, monkeypatch):
        """The reservoir is deliberately shrunk below the row count here — at the
        1M production cap a small fixture never evicts, so order could not bite."""
        dirs = self._buckets(tmp_path)
        small = self._small_acc_factory(64)
        monkeypatch.setattr(a1s, "Accumulator", small)
        got = a1s.compute_stats_for_embodiment(
            "split_aloha", _group(dirs, "bimanual", "AgileX Split Aloha"), rot6d_identity=False
        )["eef"]

        ref = small(dim=a1s.EEF20_DIM)
        for d in dirs:  # same fixed order the script must use
            _, rows, _, _ = a1s._scan_bucket((str(d), "bimanual", "split_aloha"))
            ref.update_batch(rows)
        expected = ref.finalize()
        for key in ("mean", "std", "min", "max", "q01", "q99"):
            np.testing.assert_allclose(got[key], expected[key], atol=0, rtol=0, err_msg=key)

    def test_regenerating_reproduces_the_file_byte_for_byte(self, tmp_path, inline_pool, monkeypatch):
        _make_bucket(tmp_path, "cat/split_aloha/task")
        monkeypatch.setattr("sys.argv", ["prog", "--dataset_dir", str(tmp_path)])
        out = tmp_path / "meta" / "stats_split_aloha.json"
        a1s.main()
        first = out.read_bytes()
        a1s.main()
        assert out.read_bytes() == first


# ---------------------------------------------------------------------------
# 4. Output location (read-only dataset mounts)
# ---------------------------------------------------------------------------


class TestStatsRoot:
    def test_defaults_to_the_dataset_dir(self, tmp_path, inline_pool, monkeypatch):
        _make_bucket(tmp_path, "cat/split_aloha/task")
        monkeypatch.setattr("sys.argv", ["prog", "--dataset_dir", str(tmp_path)])
        a1s.main()
        assert (tmp_path / "meta" / "stats_split_aloha.json").is_file()

    def test_stats_root_redirects_the_write_off_the_dataset_mount(self, tmp_path, inline_pool, monkeypatch):
        """interndata_a1.yaml advertises stats_root for read-only mounts and the
        reader wires a1_stats_root for it; without this flag the shipped tool
        could not produce the files in the one scenario stats_root exists for."""
        root = tmp_path / "readonly_mount"
        elsewhere = tmp_path / "scratch"
        d = _make_bucket(root, "cat/split_aloha/task")
        monkeypatch.setattr("sys.argv", ["prog", "--dataset_dir", str(root), "--stats_root", str(elsewhere)])
        a1s.main()
        assert (elsewhere / "meta" / "stats_split_aloha.json").is_file()
        assert not (root / "meta").exists()  # dataset mount untouched
        # And the reader resolves the same path from dataloader.stats_root.
        ds = InternDataA1Dataset(
            str(d), a1_stats_root=str(elsewhere), normalize_mode="quantile", num_frames=9, video_stride=4
        )
        assert ds._normalization_stats is not None


# ---------------------------------------------------------------------------
# 5. Bucket classification
# ---------------------------------------------------------------------------


class TestClassifyBuckets:
    def test_groups_by_embodiment(self, tmp_path):
        _make_bucket(tmp_path, "cat/split_aloha/a")
        _make_bucket(tmp_path, "cat/split_aloha/b")
        _make_bucket(tmp_path, "cat/franka/c", layout="single_arm", robot_type="Franka")
        _make_bucket(tmp_path, "cat/lift2/d", robot_type="ARX Lift-2")
        groups = a1s.classify_buckets(a1s.discover_a1_buckets(tmp_path))
        assert {k: len(v["dirs"]) for k, v in groups.items()} == {"split_aloha": 2, "franka": 1, "lift2": 1}
        assert groups["franka"]["arm_layout"] == "single_arm"
        assert groups["split_aloha"]["arm_layout"] == "bimanual"

    def test_one_unreadable_bucket_aborts_classification(self, tmp_path):
        _make_bucket(tmp_path, "cat/split_aloha/good")
        bad = tmp_path / "cat" / "split_aloha" / "bad"
        (bad / "meta").mkdir(parents=True)
        (bad / "meta" / "info.json").write_text("{ not json")
        with pytest.raises(RuntimeError, match="Refusing to compute partial.*unreadable meta/info.json"):
            a1s.classify_buckets(a1s.discover_a1_buckets(tmp_path))

    def test_nonempty_manifest_with_zero_physical_rows_fails_closed(self, tmp_path, inline_pool):
        """A corrupt non-empty bucket must not become a degenerate stats file."""
        d = _make_bucket(tmp_path, "cat/emb/task", n_eps=1, ep_len=1)
        pq.write_table(
            pq.read_table(d / "data" / "chunk-000" / "file-000.parquet").slice(0, 0),
            d / "data" / "chunk-000" / "file-000.parquet",
        )
        with pytest.raises(RuntimeError, match="Refusing to write partial.*non-empty bucket"):
            a1s.compute_stats_for_embodiment("split_aloha", _group([d], "bimanual", "AgileX Split Aloha"))


def test_scan_bucket_reads_action_and_state_rows(tmp_path):
    """Both streams are pooled — actions[t] == states[t+1], the same signal
    offset by one row — so the row count is 2x the parquet length."""
    d = _make_bucket(tmp_path, "cat/emb/task", n_eps=2, ep_len=20)
    name, rows, _, _ = a1s._scan_bucket((str(d), "bimanual", "split_aloha"))
    assert name == str(d)
    assert rows.shape == (2 * 40, a1s.EEF20_DIM)
    assert rows.dtype == np.float32


def test_scan_bucket_of_a_single_arm_leaves_the_right_half_zero(tmp_path):
    d = _make_bucket(tmp_path, "cat/franka/task", layout="single_arm", robot_type="Franka")
    _, rows, _, _ = a1s._scan_bucket((str(d), "single_arm", "franka"))
    np.testing.assert_array_equal(rows[:, 10:], 0.0)
    assert np.abs(rows[:, :10]).sum() > 0


def test_module_exports_stay_importable():
    for name in a1s.__all__:
        assert hasattr(a1s, name), name
    assert Path(a1s.__file__).name == "interndata_a1_stats_computation.py"


# ---------------------------------------------------------------------------
# Cleaned-view filtering: the stats must describe exactly what the reader emits
# ---------------------------------------------------------------------------


def _trim_file(path: Path, rows: str) -> str:
    path.write_text("dataset,episode_index,total_frames,trim_head_to,trim_tail_from\n" + rows)
    return str(path)


class TestCleanedViewFiltering:
    """A cleaned view symlinks ``data/``, so deleted episodes are still
    physically in the parquet. The scanner has to honour
    ``meta/excluded_episodes.json`` or the normalizer pools rows the reader
    never emits."""

    def test_excluded_episodes_are_dropped_from_the_scan(self, tmp_path):
        d = _make_bucket(tmp_path, "cat/emb/task", n_eps=2, ep_len=20)
        (d / "meta" / "excluded_episodes.json").write_text(json.dumps({"episode_indices": [0]}))
        _, rows, _, _ = a1s._scan_bucket((str(d), "bimanual", "split_aloha", "cat/emb/task", None))
        assert rows.shape[0] == 2 * 20  # only episode 1, both streams

    def test_an_empty_kept_set_excludes_everything(self, tmp_path):
        """``set()`` means "manifest read, nothing survives" — distinct from
        ``None`` ("unreadable, cannot filter"). Collapsing the two would pool a
        fully-deleted bucket's rows back into the statistics."""
        d = _make_bucket(tmp_path, "cat/emb/task", n_eps=2, ep_len=20)
        (d / "meta" / "excluded_episodes.json").write_text(json.dumps({"episode_indices": [0, 1]}))
        assert a1s._kept_episodes(d) == set()
        _, rows, _, _ = a1s._scan_bucket((str(d), "bimanual", "split_aloha", "cat/emb/task", None))
        assert rows.shape[0] == 0

    def test_kept_episodes_is_none_only_when_the_manifest_is_unreadable(self, tmp_path):
        d = _make_bucket(tmp_path, "cat/emb/task", n_eps=2, ep_len=20)
        assert a1s._kept_episodes(d) == {0, 1}
        for f in (d / "meta" / "episodes").rglob("*.parquet"):
            f.unlink()
        assert a1s._kept_episodes(d) is None


class TestStatsTrimMatchesReader:
    """Both paths call :func:`resolve_trim_bounds`, so a trim that the reader
    declines to apply must not be applied here either."""

    def test_a_too_short_trim_is_left_whole_like_the_reader_does(self, tmp_path):
        d = _make_bucket(tmp_path, "cat/emb/task", n_eps=1, ep_len=4)
        trim = _trim_file(tmp_path / "trim.csv", "cat/emb/task,0,4,3,\n")  # leaves 1 < min_len 2
        _, rows, _, _ = a1s._scan_bucket((str(d), "bimanual", "split_aloha", "cat/emb/task", trim, 2))
        assert rows.shape[0] == 2 * 4, "stats trimmed an episode the reader keeps whole"

    def test_a_valid_trim_is_applied(self, tmp_path):
        d = _make_bucket(tmp_path, "cat/emb/task", n_eps=1, ep_len=20)
        trim = _trim_file(tmp_path / "trim.csv", "cat/emb/task,0,20,4,18\n")
        _, rows, _, _ = a1s._scan_bucket((str(d), "bimanual", "split_aloha", "cat/emb/task", trim, 2))
        assert rows.shape[0] == 2 * 14  # 18 - 4

    def test_a_stale_total_frames_disables_the_entry(self, tmp_path):
        d = _make_bucket(tmp_path, "cat/emb/task", n_eps=1, ep_len=20)
        trim = _trim_file(tmp_path / "trim.csv", "cat/emb/task,0,999,4,18\n")
        _, rows, _, _ = a1s._scan_bucket((str(d), "bimanual", "split_aloha", "cat/emb/task", trim, 2))
        assert rows.shape[0] == 2 * 20


class TestDirectBucketParity:
    """`--dataset_dir` pointed straight at a bucket is a supported mode, and it
    is the one where the generator and the reader can silently disagree.

    There `relative_to(root)` is `'.'` — a key no trim CSV holds and one the
    reader (whose id is then the bare directory name) cannot suffix-match. Left
    unnormalized the generator skips the trim while the reader applies it, and
    writes `exclusions: {".": ...}` the reader then rejects. These go through the
    real `main()` rather than the helpers, because that is the seam the unit
    tests were blind to.
    """

    def _setup(self, tmp_path):
        d = _make_bucket(tmp_path, "cat/emb/task", n_eps=2, ep_len=40)
        (d / "meta" / "excluded_episodes.json").write_text(json.dumps({"episode_indices": [0]}))
        trim = tmp_path / "trim.csv"
        trim.write_text(
            "dataset,episode_index,total_frames,trim_head_to,trim_tail_from\n"
            "task,1,40,10,30\n"  # keyed by the bare bucket name, as the reader resolves it
        )
        return d, trim

    def _run(self, monkeypatch, d, trim, out):
        import sys

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                "--dataset_dir",
                str(d),
                "--stats_root",
                str(out),
                "--trim_csv",
                str(trim),
                "--workers",
                "1",
                "--min_keep",
                "2",
            ],
        )
        a1s.main()
        return json.load(open(out / "meta" / "stats_split_aloha.json"))

    def test_generator_applies_the_trim_and_the_exclusion(self, tmp_path, monkeypatch):
        d, trim = self._setup(tmp_path)
        out = tmp_path / "stats"
        res = self._run(monkeypatch, d, trim, out)
        # ep0 excluded; ep1 trimmed to [10, 30) = 20 frames, over action+state.
        assert res["num_rows"] == 40, "generator ignored the trim or the exclusion"

    def test_the_reader_accepts_what_the_generator_produced(self, tmp_path, monkeypatch):
        d, trim = self._setup(tmp_path)
        out = tmp_path / "stats"
        self._run(monkeypatch, d, trim, out)
        ds = InternDataA1Dataset(  # no dataset_id: id is the bare bucket name
            str(d),
            a1_stats_root=str(out),
            trim_csv=str(trim),
            normalize_mode="quantile",
            num_frames=9,
            video_stride=4,
        )
        assert ds._normalization_stats is not None
        assert ds._eps_df["episode_index"].tolist() == [1]
        assert int(ds._eps_df["length"].iloc[0]) == 20

    def test_the_emitted_key_is_not_a_dot(self, tmp_path, monkeypatch):
        d, trim = self._setup(tmp_path)
        out = tmp_path / "stats"
        res = self._run(monkeypatch, d, trim, out)
        assert "." not in res["scanned_buckets"], "direct-bucket key leaked as '.'"
        assert res["scanned_buckets"] == ["task"]

    def test_reader_rejects_stats_after_trim_csv_content_changes(self, tmp_path, monkeypatch):
        d, trim = self._setup(tmp_path)
        out = tmp_path / "stats"
        self._run(monkeypatch, d, trim, out)

        # Keep the same path and bucket key, but change the effective span after
        # the stats were generated.  A path-only or trim-enabled boolean check
        # would accept the stale normalizer.
        trim.write_text("dataset,episode_index,total_frames,trim_head_to,trim_tail_from\ntask,1,40,5,30\n")
        with pytest.raises(DataContractError, match="trim provenance"):
            InternDataA1Dataset(
                str(d),
                a1_stats_root=str(out),
                trim_csv=str(trim),
                normalize_mode="quantile",
                num_frames=9,
                video_stride=4,
            )

    def test_reader_rejects_stats_after_exclusions_change(self, tmp_path, monkeypatch):
        d, trim = self._setup(tmp_path)
        out = tmp_path / "stats"
        self._run(monkeypatch, d, trim, out)

        # The old stats exclude episode 0.  Removing that exclusion changes the
        # population without changing the shared stats path or trim artifact.
        (d / "meta" / "excluded_episodes.json").write_text(json.dumps({"episode_indices": []}))
        with pytest.raises(DataContractError, match="excluded_episodes.json changed"):
            InternDataA1Dataset(
                str(d),
                a1_stats_root=str(out),
                trim_csv=str(trim),
                normalize_mode="quantile",
                num_frames=9,
                video_stride=4,
            )

    def test_capped_reader_still_checks_the_full_train_population(self, tmp_path, monkeypatch):
        d = _make_bucket(tmp_path, "cat/emb/task", n_eps=3, ep_len=40)
        trim = tmp_path / "trim.csv"
        trim.write_text("dataset,episode_index,total_frames,trim_head_to,trim_tail_from\ntask,1,40,5,30\n")
        out = tmp_path / "stats"
        self._run(monkeypatch, d, trim, out)

        # Narrow the train split after stats generation. max_hours is applied
        # only after A1 captures its full pre-subsample train certificate, so it
        # must not exempt this reader from the population check.
        info_path = d / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        info["splits"] = {"train": "0:2", "val": "2:3"}
        info_path.write_text(json.dumps(info))
        with pytest.raises(DataContractError, match="effective.*population changed"):
            InternDataA1Dataset(
                str(d),
                a1_stats_root=str(out),
                trim_csv=str(trim),
                normalize_mode="quantile",
                num_frames=9,
                video_stride=4,
                max_hours=1.0,
            )


class TestFailClosedCoverage:
    """A real bucket failure aborts the shared stats instead of shrinking it."""

    def test_a_failed_nonempty_bucket_aborts_the_embodiment(self, tmp_path):
        good = _make_bucket(tmp_path, "cat/emb/good", n_eps=2, ep_len=8)
        bad = _make_bucket(tmp_path, "cat/emb/bad", n_eps=2, ep_len=8)
        # Strip episode_index so the scan of `bad` raises (trim/exclusions cannot
        # select rows without it).
        (bad / "meta" / "excluded_episodes.json").write_text(json.dumps({"episode_indices": [0]}))
        p = bad / "data" / "chunk-000" / "file-000.parquet"
        t = pq.read_table(p)
        pq.write_table(t.drop(["episode_index"]), p)

        with pytest.raises(RuntimeError, match="Refusing to write partial.*cat/emb/bad"):
            a1s.compute_stats_for_embodiment(
                "split_aloha",
                _group([good, bad], "bimanual", "AgileX Split Aloha"),
                workers=1,
                root=tmp_path,
            )

    def test_cli_publishes_no_embodiment_when_a_later_one_fails(self, tmp_path, monkeypatch, inline_pool):
        """The CLI is transactional across its requested embodiment set."""
        _make_bucket(
            tmp_path,
            "cat/franka/good",
            n_eps=2,
            ep_len=8,
            layout="single_arm",
            robot_type="Franka",
        )
        bad = _make_bucket(tmp_path, "cat/split_aloha/bad", n_eps=2, ep_len=8)
        p = bad / "data" / "chunk-000" / "file-000.parquet"
        pq.write_table(pq.read_table(p).slice(0, 15), p)

        out = tmp_path / "published"
        meta = out / "meta"
        meta.mkdir(parents=True)
        old = {
            meta / "stats_franka.json": "old-franka\n",
            meta / "stats_split_aloha.json": "old-split\n",
        }
        for path, content in old.items():
            path.write_text(content)
        monkeypatch.setattr(
            "sys.argv",
            [
                "prog",
                "--dataset_dir",
                str(tmp_path),
                "--stats_root",
                str(out),
                "--workers",
                "1",
            ],
        )

        with pytest.raises(RuntimeError, match="Refusing to write partial.*cat/split_aloha/bad"):
            a1s.main()
        assert {path: path.read_text() for path in old} == old
        assert not list(meta.glob("*.json.tmp"))


class TestSplitFailuresDoNotFailOpen:
    """ "Cannot determine the population" must never degrade into "use every row".

    The reader raises on the same metadata and refuses the bucket, so pooling it
    here would put rows into the normalizer that training can never load.
    """

    def test_a_malformed_split_spec_raises(self, tmp_path):
        d = _make_bucket(tmp_path, "cat/emb/task", n_eps=2, ep_len=8)
        info = json.loads((d / "meta" / "info.json").read_text())
        info["splits"] = {"train": "not-a-range"}
        (d / "meta" / "info.json").write_text(json.dumps(info))
        with pytest.raises(ValueError, match="unusable splits"):
            a1s._split_episodes(d, "train")

    def test_an_unreadable_manifest_raises(self, tmp_path):
        d = _make_bucket(tmp_path, "cat/emb/task", n_eps=2, ep_len=8)
        for f in (d / "meta" / "episodes").rglob("*.parquet"):
            f.unlink()
        with pytest.raises(ValueError, match="population is unknown"):
            a1s._split_episodes(d, "train")

    def test_absent_splits_are_empty_for_val_not_everything(self, tmp_path):
        d = _make_bucket(tmp_path, "cat/emb/task", n_eps=2, ep_len=8)
        info = json.loads((d / "meta" / "info.json").read_text())
        info.pop("splits", None)
        (d / "meta" / "info.json").write_text(json.dumps(info))
        assert a1s._split_episodes(d, "train") is None  # unrestricted
        assert a1s._split_episodes(d, "val") == set()  # not "everything"


class TestScannerRefusesWhatTheReaderRefuses:
    """`len(rows) > 0` proves rows were read, not that a reader can consume them.

    Each case below yielded a healthy row count here while
    ``InternDataA1Dataset`` raised on the same bucket — statistics describing a
    population that never reaches training, with nothing downstream able to tell.
    Both sides now enumerate shards and validate the manifest through the same
    functions, so these are parity assertions rather than a second rule set.
    """

    def _scan(self, d, **kw):
        return (
            a1s._scan_bucket((str(d), "bimanual", "split_aloha", kw.pop("dsid", "task")), **kw)
            if kw
            else a1s._scan_bucket((str(d), "bimanual", "split_aloha"))
        )

    def test_a_truncated_shard_is_refused_like_the_reader_refuses_it(self, tmp_path):
        d = _make_bucket(tmp_path, "cat/emb/task", n_eps=2, ep_len=4)
        p = d / "data" / "chunk-000" / "file-000.parquet"
        t = pq.read_table(p)
        pq.write_table(t.slice(0, t.num_rows - 1), p)  # manifest says 8, shard holds 7

        with pytest.raises(ValueError, match="manifest ends at 8"):
            InternDataA1Dataset(str(d), normalize_mode=None, num_frames=2, video_stride=1)
        with pytest.raises(ValueError, match="manifest ends at 8"):
            a1s._scan_bucket((str(d), "bimanual", "split_aloha"))

    def test_a_stray_backup_parquet_does_not_double_the_population(self, tmp_path):
        """`file-000.backup.parquet` matched the old glob; the reader never read it."""
        import shutil

        d = _make_bucket(tmp_path, "cat/emb/task", n_eps=2, ep_len=4)
        src = d / "data" / "chunk-000" / "file-000.parquet"
        _, rows_before, _, _ = a1s._scan_bucket((str(d), "bimanual", "split_aloha"))
        shutil.copy(src, src.with_name("file-000.backup.parquet"))
        _, rows_after, _, _ = a1s._scan_bucket((str(d), "bimanual", "split_aloha"))
        assert len(rows_after) == len(rows_before), "a non-shard file entered the statistics"

    def test_overlapping_manifest_ranges_are_refused(self, tmp_path):
        d = _make_bucket(tmp_path, "cat/emb/task", n_eps=3, ep_len=4)
        man = d / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        m = pq.read_table(man).to_pydict()
        m["dataset_from_index"], m["dataset_to_index"] = [0, 2, 8], [4, 6, 12]
        m["length"] = [4, 4, 4]
        import pyarrow as pa

        pq.write_table(pa.Table.from_pydict(m), man)
        with pytest.raises(ValueError, match="Overlapping manifest ranges"):
            a1s._scan_bucket((str(d), "bimanual", "split_aloha"))

    def test_quoted_exclusion_indices_are_refused_on_both_sides(self, tmp_path):
        """`["0"]` excluded episode 0 here and nothing in the reader."""
        d = _make_bucket(tmp_path, "cat/emb/task", n_eps=2, ep_len=4)
        (d / "meta" / "excluded_episodes.json").write_text(json.dumps({"episode_indices": ["0"]}))
        with pytest.raises(ValueError, match="must be JSON integers"):
            a1s._scan_bucket((str(d), "bimanual", "split_aloha"))


class TestPopulationContractEndToEnd:
    """Through the real `main()` and the real reader — no hand-written stats.

    Every provenance defect so far survived a green suite because the tests
    wrote the stats file themselves and so could not observe what the generator
    actually emits. These run the CLI and hand its output to a reader.
    """

    def _run(self, monkeypatch, d, out, *extra):
        import sys

        monkeypatch.setattr(
            sys, "argv", ["prog", "--dataset_dir", str(d), "--stats_root", str(out), "--workers", "1", *extra]
        )
        a1s.main()
        return json.load(open(out / "meta" / "stats_split_aloha.json"))

    def test_the_generator_records_what_it_scanned(self, tmp_path, monkeypatch):
        d = _make_bucket(tmp_path, "cat/emb/task", n_eps=2, ep_len=8)
        res = self._run(monkeypatch, d, tmp_path / "stats")
        pop = res["population"]
        assert {key: pop[key] for key in ("split", "trim_active", "min_keep", "buckets", "empty_buckets")} == {
            "split": "train",
            "trim_active": False,
            "min_keep": 2,
            "buckets": ["task"],
            "empty_buckets": [],
        }
        assert pop["schema_version"] == 2
        assert pop["trim_provenance"] is None
        assert pop["bucket_provenance"]["task"]["excluded_episode_indices"] == []
        assert pop["bucket_provenance"]["task"]["effective_population"]["num_episodes"] == 2
        assert pop["bucket_provenance"]["task"]["effective_population"]["num_rows"] == 16

    def test_val_stats_are_refused_by_a_train_reader(self, tmp_path, monkeypatch):
        d = _make_bucket(tmp_path, "cat/emb/task", n_eps=2, ep_len=8)
        out = tmp_path / "stats"
        # info.json ships splits {"train": "0:N"}, so give it a val range to scan.
        info = json.loads((d / "meta" / "info.json").read_text())
        info["splits"] = {"train": "0:1", "val": "1:2"}
        (d / "meta" / "info.json").write_text(json.dumps(info))
        res = self._run(monkeypatch, d, out, "--split", "val")
        assert res["population"]["split"] == "val"
        with pytest.raises(DataContractError, match="computed over the 'val' split"):
            InternDataA1Dataset(str(d), a1_stats_root=str(out), normalize_mode="quantile", num_frames=2, video_stride=1)

    def test_untrimmed_stats_are_refused_by_a_trimming_reader(self, tmp_path, monkeypatch):
        d = _make_bucket(tmp_path, "cat/emb/task", n_eps=2, ep_len=8)
        out = tmp_path / "stats"
        self._run(monkeypatch, d, out)  # generated WITHOUT --trim_csv
        trim = tmp_path / "trim.csv"
        trim.write_text("dataset,episode_index,total_frames,trim_head_to,trim_tail_from\ntask,0,8,2,6\n")
        with pytest.raises(DataContractError, match="without a trim list but this reader is"):
            InternDataA1Dataset(
                str(d),
                a1_stats_root=str(out),
                trim_csv=str(trim),
                normalize_mode="quantile",
                num_frames=2,
                video_stride=1,
            )

    def test_cli_refuses_to_publish_when_one_bucket_scan_fails(self, tmp_path, monkeypatch):
        """One good bucket must not turn a failed embodiment into partial stats."""
        _make_bucket(tmp_path, "cat/emb/good", n_eps=2, ep_len=8)
        bad = _make_bucket(tmp_path, "cat/emb/bad", n_eps=2, ep_len=8)
        p = bad / "data" / "chunk-000" / "file-000.parquet"
        pq.write_table(pq.read_table(p).slice(0, 15), p)  # truncated: scan raises

        out = tmp_path / "stats"
        with pytest.raises(RuntimeError, match="Refusing to write partial.*cat/emb/bad"):
            self._run(monkeypatch, tmp_path, out)
        assert not (out / "meta" / "stats_split_aloha.json").exists()


class TestValOnlyBucketReusesTrainStats:
    """A val-only bucket must borrow the train distribution, not be refused.

    Establishing "stats always come from train" while also demanding the bucket
    appear among the TRAIN contributors is contradictory: a bucket whose
    `info.json` declares `train=0:0, val=0:2` cannot be a train contributor and
    is still a perfectly valid bucket to evaluate. Refusing it rejected the
    configuration the train-derived rule exists to support.
    """

    def _two_buckets(self, tmp_path):
        a = _make_bucket(tmp_path, "cat/emb/a", n_eps=2, ep_len=8)
        b = _make_bucket(tmp_path, "cat/emb/b", n_eps=2, ep_len=8)
        for bucket, splits in ((a, {"train": "0:2"}), (b, {"train": "0:0", "val": "0:2"})):
            info = json.loads((bucket / "meta" / "info.json").read_text())
            info["splits"] = splits
            (bucket / "meta" / "info.json").write_text(json.dumps(info))
        return a, b

    def _generate(self, monkeypatch, root, out):
        import sys

        monkeypatch.setattr(
            sys, "argv", ["prog", "--dataset_dir", str(root), "--stats_root", str(out), "--workers", "1"]
        )
        a1s.main()
        return json.load(open(out / "meta" / "stats_split_aloha.json"))

    def test_the_val_only_bucket_is_recorded_as_empty_not_dropped(self, tmp_path, monkeypatch):
        self._two_buckets(tmp_path)
        pop = self._generate(monkeypatch, tmp_path, tmp_path / "stats")["population"]
        assert pop["buckets"] == ["cat/emb/a"]
        assert pop["empty_buckets"] == ["cat/emb/b"]

    def test_the_val_only_bucket_has_windows_without_normalization(self, tmp_path):
        """Pins the premise: B is a real bucket, not an empty one."""
        _, b = self._two_buckets(tmp_path)
        r = InternDataA1Dataset(
            str(b), dataset_id="cat/emb/b", normalize_mode=None, num_frames=2, video_stride=1, split="val"
        )
        assert len(r) > 0

    def test_the_val_only_bucket_loads_the_train_stats(self, tmp_path, monkeypatch):
        _, b = self._two_buckets(tmp_path)
        out = tmp_path / "stats"
        self._generate(monkeypatch, tmp_path, out)
        r = InternDataA1Dataset(
            str(b),
            dataset_id="cat/emb/b",
            a1_stats_root=str(out),
            normalize_mode="quantile",
            num_frames=2,
            video_stride=1,
            split="val",
        )
        assert r._normalization_stats is not None
        assert len(r) > 0

    def test_a_corrupt_val_only_bucket_still_aborts_generation(self, tmp_path, monkeypatch):
        """Empty-by-split is allowed only when the bucket itself validates."""
        _, b = self._two_buckets(tmp_path)
        p = b / "data" / "chunk-000" / "file-000.parquet"
        pq.write_table(pq.read_table(p).slice(0, 15), p)  # truncated -> scan raises
        out = tmp_path / "stats"
        with pytest.raises(RuntimeError, match="Refusing to write partial.*cat/emb/b"):
            self._generate(monkeypatch, tmp_path, out)
        assert not (out / "meta" / "stats_split_aloha.json").exists()


class TestZeroRowsIsNotProofOfAnEmptyPopulation:
    """A substituted physical slice survives every count-and-envelope check.

    Ranges, grand total and the shard's `episode_index` min/max are all
    preserved when one episode's rows are overwritten with another's, so the
    only symptom is that the mask for the expected episode selects nothing.
    Read as "empty population" that bucket gets certified as covered, and its
    reader then loads another bucket's statistics while reading another
    episode's rows.
    """

    def _substituted(self, tmp_path):
        """3 episodes x 4 rows, split train=1:2, rows [4:8] replaced by ep2's."""
        import pyarrow as pa

        d = _make_bucket(tmp_path, "cat/emb/bad", n_eps=3, ep_len=4)
        info = json.loads((d / "meta" / "info.json").read_text())
        info["splits"] = {"train": "1:2"}
        (d / "meta" / "info.json").write_text(json.dumps(info))
        p = d / "data" / "chunk-000" / "file-000.parquet"
        t = pq.read_table(p).to_pydict()
        for k, v in t.items():
            t[k] = v[:4] + v[8:12] + v[8:12]  # ep0, ep2, ep2 — total and envelope intact
        pq.write_table(pa.Table.from_pydict(t), p)
        return d

    def test_the_premise_holds_total_and_envelope_are_unchanged(self, tmp_path):
        d = self._substituted(tmp_path)
        t = pq.read_table(d / "data" / "chunk-000" / "file-000.parquet")
        eps = t.column("episode_index").to_pylist()
        assert t.num_rows == 12, "grand total must still match the manifest"
        assert (min(eps), max(eps)) == (0, 2), "envelope must still contain episode 1"
        assert 1 not in eps, "episode 1's rows are the ones that were replaced"

    def test_the_scanner_refuses_it_instead_of_calling_it_empty(self, tmp_path):
        d = self._substituted(tmp_path)
        with pytest.raises(ValueError, match="no shard row carries them"):
            a1s._scan_bucket((str(d), "bimanual", "split_aloha", "cat/emb/bad"))

    def test_substituted_rows_abort_generation(self, tmp_path, monkeypatch):
        import sys

        _make_bucket(tmp_path, "cat/emb/good", n_eps=2, ep_len=8)
        self._substituted(tmp_path)
        out = tmp_path / "stats"
        monkeypatch.setattr(
            sys, "argv", ["prog", "--dataset_dir", str(tmp_path), "--stats_root", str(out), "--workers", "1"]
        )
        with pytest.raises(RuntimeError, match="Refusing to write partial.*cat/emb/bad"):
            a1s.main()
        assert not (out / "meta" / "stats_split_aloha.json").exists()

    def test_a_genuinely_empty_train_split_is_still_reported_empty(self, tmp_path):
        """The distinction must not collapse the other way."""
        d = _make_bucket(tmp_path, "cat/emb/valonly", n_eps=2, ep_len=8)
        info = json.loads((d / "meta" / "info.json").read_text())
        info["splits"] = {"train": "0:0", "val": "0:2"}
        (d / "meta" / "info.json").write_text(json.dumps(info))
        _, rows, population_empty, _ = a1s._scan_bucket((str(d), "bimanual", "split_aloha", "cat/emb/valonly"))
        assert len(rows) == 0 and population_empty is True
