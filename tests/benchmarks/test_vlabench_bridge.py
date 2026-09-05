"""VLABench <-> OpenWAM bridge conversion tests.

Pure numpy: no simulator, no server, no VLABench install required. What these
pin down is the handful of conventions that would silently destroy an eval run
if they drifted — coordinate frame, rotation representation, gripper polarity —
each checked against the dataloader side that produced the training data.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.utils.action_conversion import (
    VLABENCH_EEF10_DIM,
    VLABENCH_GRIPPER_OPEN_WIDTH,
    VLABENCH_ROBOT_BASE_DEFAULT,
    eef10_to_vlabench_ee,
    rot6d_to_euler_xyz,
    vlabench_obs_to_eef10,
)
from openwam.dataloader.utils.eef import euler_xyz_to_rot6d
from openwam.dataloader.utils.oxe_schema import euler7_action_to_arm10


def _euler_to_quat_wxyz(euler: np.ndarray) -> np.ndarray:
    """Extrinsic XYZ euler -> [w, x, y, z], matching R = Rz @ Ry @ Rx."""
    cx, cy, cz = np.cos(np.asarray(euler, np.float64) * 0.5)
    sx, sy, sz = np.sin(np.asarray(euler, np.float64) * 0.5)
    return np.array(
        [
            cx * cy * cz + sx * sy * sz,
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
        ]
    )


class TestRot6dToEulerXyz:
    """rot6d -> euler must invert the dataloader's euler -> rot6d exactly."""

    def test_round_trip_random(self):
        rng = np.random.default_rng(0)
        for _ in range(2000):
            euler = np.array(
                [
                    rng.uniform(-np.pi, np.pi),
                    rng.uniform(-np.pi / 2 * 0.99, np.pi / 2 * 0.99),
                    rng.uniform(-np.pi, np.pi),
                ]
            )
            r6d = euler_xyz_to_rot6d(euler[None])[0]
            # Euler triples are not unique; compare through the rotation they encode.
            np.testing.assert_allclose(euler_xyz_to_rot6d(rot6d_to_euler_xyz(r6d)[None])[0], r6d, atol=1e-6)

    def test_identity_rotation(self):
        r6d = euler_xyz_to_rot6d(np.zeros((1, 3)))[0]
        np.testing.assert_allclose(rot6d_to_euler_xyz(r6d), np.zeros(3), atol=1e-9)

    @pytest.mark.parametrize("pitch", [np.pi / 2, -np.pi / 2])
    def test_gimbal_lock_stays_finite_and_consistent(self, pitch):
        """At cos(pitch) = 0 roll/yaw are degenerate; the encoded rotation must still match."""
        euler = np.array([0.3, pitch, -0.7])
        r6d = euler_xyz_to_rot6d(euler[None])[0]
        out = rot6d_to_euler_xyz(r6d)
        assert np.all(np.isfinite(out))
        np.testing.assert_allclose(euler_xyz_to_rot6d(out[None])[0], r6d, atol=1e-6)


class TestObsToEef10:
    def test_matches_dataloader_on_same_physical_state(self):
        """The client's proprio must equal what the reader built from the recorded state.

        The dataset stores ``[pos - base, euler, gripper]`` and the reader runs
        it through ``euler7_action_to_arm10``; the client sees the same pose as
        a world-frame quaternion. Both must land on identical EEF10.
        """
        base = np.array([0.0, -0.4, 0.78])
        world_pos = np.array([0.11, -0.05, 1.02])
        euler = np.array([0.4, -0.21, 1.7])
        gripper = 1.0

        client_eef10 = vlabench_obs_to_eef10(np.concatenate([world_pos, _euler_to_quat_wxyz(euler), [gripper]]), base)
        dataset_euler7 = np.concatenate([world_pos - base, euler, [gripper]])[None]
        reader_eef10 = euler7_action_to_arm10(dataset_euler7.astype(np.float32))[0]

        np.testing.assert_allclose(client_eef10, reader_eef10, atol=1e-5)

    def test_subtracts_robot_base(self):
        base = np.array([0.0, -0.4, 0.78])
        world_pos = np.array([0.2, 0.1, 1.0])
        eef10 = vlabench_obs_to_eef10(np.concatenate([world_pos, [1.0, 0.0, 0.0, 0.0], [0.0]]), base)
        np.testing.assert_allclose(eef10[0:3], world_pos - base, atol=1e-6)

    def test_gripper_forwarded_verbatim(self):
        """State gripper is passed through, upstream polarity inversion included."""
        for raw in (0.0, 1.0):
            eef10 = vlabench_obs_to_eef10(
                np.concatenate([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [raw]]),
                np.zeros(3),
            )
            assert eef10[9] == pytest.approx(raw)

    def test_rejects_short_state(self):
        with pytest.raises(ValueError, match="at least 8-D"):
            vlabench_obs_to_eef10(np.zeros(7), np.zeros(3))

    def test_rejects_bad_base(self):
        with pytest.raises(ValueError, match="robot_base must be 3-D"):
            vlabench_obs_to_eef10(np.zeros(8), np.zeros(4))


class TestEef10ToVlabenchEe:
    def test_adds_robot_base_back(self):
        base = np.array([0.0, -0.4, 0.78])
        eef10 = np.concatenate([[0.1, 0.2, 0.3], euler_xyz_to_rot6d(np.zeros((1, 3)))[0], [1.0]])
        pos, _euler, _grip = eef10_to_vlabench_ee(eef10, base)
        np.testing.assert_allclose(pos, np.array([0.1, 0.2, 0.3]) + base, atol=1e-6)

    def test_frame_round_trip_is_identity(self):
        """obs -> EEF10 -> target must return the original world pose."""
        base = np.array([0.0, -0.4, 0.78])
        world_pos = np.array([0.13, 0.02, 1.11])
        euler = np.array([-0.2, 0.35, 2.1])
        ee_state = np.concatenate([world_pos, _euler_to_quat_wxyz(euler), [1.0]])

        eef10 = vlabench_obs_to_eef10(ee_state, base)
        pos, out_euler, _grip = eef10_to_vlabench_ee(eef10, base)

        np.testing.assert_allclose(pos, world_pos, atol=1e-5)
        np.testing.assert_allclose(
            euler_xyz_to_rot6d(out_euler[None])[0], euler_xyz_to_rot6d(euler[None])[0], atol=1e-5
        )

    @pytest.mark.parametrize(
        "grip,expect_open",
        [(1.0, True), (0.9, True), (0.5, True), (0.49, False), (0.0, False), (-1.0, False)],
    )
    def test_gripper_polarity_one_is_open(self, grip, expect_open):
        """Dataset action convention: 1 = OPEN (finger width > 0.03 of a 0.04 m span).

        Inverting this is the single most damaging silent bug available here —
        the arm would grasp air and release on contact.
        """
        eef10 = np.concatenate([np.zeros(3), euler_xyz_to_rot6d(np.zeros((1, 3)))[0], [grip]])
        _pos, _euler, gripper_state = eef10_to_vlabench_ee(eef10, np.zeros(3))
        expected = np.full(2, VLABENCH_GRIPPER_OPEN_WIDTH) if expect_open else np.zeros(2)
        np.testing.assert_allclose(gripper_state, expected)
        assert gripper_state.shape == (2,)

    def test_custom_gripper_threshold_and_width(self):
        eef10 = np.concatenate([np.zeros(3), euler_xyz_to_rot6d(np.zeros((1, 3)))[0], [0.2]])
        _p, _e, grip = eef10_to_vlabench_ee(eef10, np.zeros(3), gripper_open_threshold=0.1, gripper_open_width=0.07)
        np.testing.assert_allclose(grip, np.full(2, 0.07))

    def test_rejects_wrong_action_dim(self):
        with pytest.raises(ValueError, match=f"{VLABENCH_EEF10_DIM}-D EEF action"):
            eef10_to_vlabench_ee(np.zeros(20), np.zeros(3))

    def test_default_base_matches_converter_fallback(self):
        """VLABench's converter falls back to [0, -0.4, 0.78] when an episode
        config carries no robot.position; the client must use the same value."""
        assert VLABENCH_ROBOT_BASE_DEFAULT == (0.0, -0.4, 0.78)


def _load_single_eval():
    """Import benchmarks/vlabench/single_eval.py by path.

    It is a script, not a package module, and it sits next to
    openwam2vlabench_interface.py which it imports by bare name — so its own
    directory has to be on sys.path.
    """
    import importlib.util
    import sys
    from pathlib import Path

    script_dir = Path(__file__).resolve().parents[2] / "benchmarks" / "vlabench"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("_single_eval", script_dir / "single_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestIntentionScorePatch:
    """The upstream metric bug that silently destroys completed episodes.

    ``Evaluator.evaluate_single_episode`` calls ``env.get_intention_score()``
    after the rollout but before recording ``info["success"]``. On the
    positionally-targeted tasks it raises KeyError, the per-episode ``except``
    in ``Evaluator.evaluate`` swallows the whole record, and a finished rollout
    is lost. It took out all 50 episodes of ``select_poker_spatial``.
    """

    @staticmethod
    def _fake_env_cls():
        class FakeEnv:
            def __init__(self, mode="ok"):
                self.mode = mode

            def get_intention_score(self, threshold=0.5, discrete=True):
                if self.mode == "keyerror":
                    raise KeyError("10_of_spades")
                if self.mode == "other":
                    raise ValueError("unrelated failure")
                return 0.75

        return FakeEnv

    def test_keyerror_becomes_nan(self):
        cls = self._fake_env_cls()
        _load_single_eval().patch_intention_score_keyerror(cls)
        assert np.isnan(cls("keyerror").get_intention_score())

    def test_normal_path_untouched(self):
        cls = self._fake_env_cls()
        _load_single_eval().patch_intention_score_keyerror(cls)
        assert cls("ok").get_intention_score(threshold=0.1) == 0.75

    def test_other_exceptions_still_propagate(self):
        """A blanket except would hide real breakage behind a plausible metric."""
        cls = self._fake_env_cls()
        _load_single_eval().patch_intention_score_keyerror(cls)
        with pytest.raises(ValueError, match="unrelated failure"):
            cls("other").get_intention_score()

    def test_idempotent(self):
        """single_eval patches once per process; double-wrapping must be a no-op."""
        cls = self._fake_env_cls()
        mod = _load_single_eval()
        assert mod.patch_intention_score_keyerror(cls) is True
        assert mod.patch_intention_score_keyerror(cls) is False

    def test_nan_confines_damage_to_intention_score(self):
        """Why NaN and not 0.0: compute_metric averages each metric separately,
        so success_rate and progress_score survive while intention_score reads
        as unavailable rather than as a silently deflated number."""
        infos = [
            {"success": True, "intention_score": float("nan"), "progress_score": 0.8},
            {"success": False, "intention_score": float("nan"), "progress_score": 0.2},
        ]
        assert np.mean([i["success"] for i in infos]) == 0.5
        assert np.mean([i["progress_score"] for i in infos]) == 0.5
        assert np.isnan(np.mean([i["intention_score"] for i in infos]))
