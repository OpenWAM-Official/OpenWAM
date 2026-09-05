"""GR1 joint / end-effector kinematics for training and benchmark deployment.

The NVIDIA tabletop data stores a padded 44-D vector while the executable
``GR1ArmsAndWaistFourierHands`` environment exposes 29 active dimensions.  This
module is the single source of truth for projecting either representation to
OpenWAM's 33-D EEF+dex-hand layout and for solving EEF targets back to the
environment's absolute arm-joint commands.

MuJoCo / RoboCasa imports are intentionally avoided at module import time.  A
live robosuite environment is injected through :meth:`GR1Kinematics.from_env`,
which keeps dataset-only unit tests lightweight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

JOINT44_DIM = 44
EEF33_DIM = 33
JOINT29_DIM = 29

JOINT44_SLICES = {
    "left_arm": slice(0, 7),
    "left_hand": slice(7, 13),
    "left_leg": slice(13, 19),
    "neck": slice(19, 22),
    "right_arm": slice(22, 29),
    "right_hand": slice(29, 35),
    "right_leg": slice(35, 41),
    "waist": slice(41, 44),
}

EEF33_SLICES = {
    "left_pose": slice(0, 9),
    "left_hand": slice(9, 15),
    "right_pose": slice(15, 24),
    "right_hand": slice(24, 30),
    "waist": slice(30, 33),
}

# Raw EEF33 -> canonical unified-80 destinations.
EEF33_UNIFY_DST = np.asarray(
    [
        *range(0, 9),
        *range(10, 16),
        *range(34, 43),
        *range(44, 50),
        *range(68, 71),
    ],
    dtype=np.int64,
)

ROT6D_DIMS_EEF33 = tuple(range(3, 9)) + tuple(range(18, 24))
HAND_DIMS_EEF33 = tuple(range(9, 15)) + tuple(range(24, 30))

STATE_KEYS = (
    "state.left_arm",
    "state.left_hand",
    "state.right_arm",
    "state.right_hand",
    "state.waist",
)
ACTION_KEYS = (
    "action.left_arm",
    "action.left_hand",
    "action.right_arm",
    "action.right_hand",
    "action.waist",
)


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    """Return the first two columns of a 3x3 rotation matrix."""
    mat = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    return np.concatenate([mat[:, 0], mat[:, 1]]).astype(np.float32)


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Gram-Schmidt a rot6d vector into a proper 3x3 rotation matrix."""
    vec = np.asarray(rot6d, dtype=np.float64).reshape(6)
    first = vec[:3]
    first_norm = float(np.linalg.norm(first))
    if first_norm < 1e-6:
        raise ValueError("rot6d first basis vector is degenerate")
    first /= first_norm
    second = vec[3:] - np.dot(first, vec[3:]) * first
    second_norm = float(np.linalg.norm(second))
    if second_norm < 1e-6:
        raise ValueError("rot6d second basis vector is degenerate or collinear")
    second /= second_norm
    return np.stack([first, second, np.cross(first, second)], axis=1)


def matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to the shortest axis-angle rotation vector."""
    mat = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    angle = float(np.arccos(np.clip((np.trace(mat) - 1.0) * 0.5, -1.0, 1.0)))
    if angle < 1e-9:
        return np.zeros(3, dtype=np.float64)
    if np.pi - angle < 1e-6:
        # Stable eigenvector extraction near pi.
        vals, vecs = np.linalg.eigh((mat + np.eye(3)) * 0.5)
        axis = vecs[:, int(np.argmax(vals))]
        return axis * angle
    axis = np.array(
        [mat[2, 1] - mat[1, 2], mat[0, 2] - mat[2, 0], mat[1, 0] - mat[0, 1]],
        dtype=np.float64,
    )
    axis /= 2.0 * np.sin(angle)
    return axis * angle


def _unwrap_env(env):
    wrapper = getattr(env, "unwrapped", env)
    sim_env = getattr(wrapper, "env", wrapper)
    if not hasattr(sim_env, "sim") or not getattr(sim_env, "robots", None):
        raise TypeError("GR1Kinematics requires a live RoboCasa/robosuite environment")
    return sim_env


@dataclass
class IKSolution:
    left_arm: np.ndarray
    right_arm: np.ndarray
    position_error: float
    rotation_error: float
    converged: bool


class GR1Kinematics:
    """FK/IK adapter bound to one live GR1 robosuite simulation."""

    def __init__(self, sim_env):
        self.env = sim_env
        self.sim = sim_env.sim
        self.robot = sim_env.robots[0]
        self.model = self.sim.model._model
        self.data = self.sim.data._data

        controllers = self.robot.composite_controller.part_controllers
        self.qpos_index = {
            "left_arm": np.asarray(controllers["left"].joint_index, dtype=np.int64),
            "right_arm": np.asarray(controllers["right"].joint_index, dtype=np.int64),
            "waist": np.asarray(controllers["torso"].joint_index, dtype=np.int64),
        }
        self.joint_names = {
            "left_arm": tuple(controllers["left"].joint_names),
            "right_arm": tuple(controllers["right"].joint_names),
            "waist": tuple(controllers["torso"].joint_names),
        }
        self.dof_index = {
            part: np.asarray(
                [self.model.jnt_dofadr[self.model.joint(name).id] for name in names],
                dtype=np.int64,
            )
            for part, names in self.joint_names.items()
        }
        self.site_id = {
            "left": int(self.robot.eef_site_id["left"]),
            "right": int(self.robot.eef_site_id["right"]),
        }
        self.base_body_id = int(self.model.body(f"{self.robot.robot_model.naming_prefix}base").id)

        for part, expected in (("left_arm", 7), ("right_arm", 7), ("waist", 3)):
            if self.qpos_index[part].shape != (expected,):
                raise ValueError(f"GR1 {part} qpos width must be {expected}, got {self.qpos_index[part]}")

    @classmethod
    def from_env(cls, env) -> "GR1Kinematics":
        return cls(_unwrap_env(env))

    def _set_active_qpos(self, left_arm, right_arm, waist) -> None:
        self.data.qpos[self.qpos_index["left_arm"]] = np.asarray(left_arm, dtype=np.float64)
        self.data.qpos[self.qpos_index["right_arm"]] = np.asarray(right_arm, dtype=np.float64)
        self.data.qpos[self.qpos_index["waist"]] = np.asarray(waist, dtype=np.float64)
        self.sim.forward()

    def _site_pose(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        site = self.site_id[side]
        world_pos = np.asarray(self.sim.data.site_xpos[site], dtype=np.float64)
        world_rot = np.asarray(self.sim.data.site_xmat[site], dtype=np.float64).reshape(3, 3)
        base_pos = np.asarray(self.data.xpos[self.base_body_id], dtype=np.float64)
        base_rot = np.asarray(self.data.xmat[self.base_body_id], dtype=np.float64).reshape(3, 3)
        base_from_world = base_rot.T
        pos = base_from_world @ (world_pos - base_pos)
        rot = base_from_world @ world_rot
        return pos.astype(np.float32), matrix_to_rot6d(rot)

    def active_to_eef33(
        self,
        left_arm: np.ndarray,
        left_hand: np.ndarray,
        right_arm: np.ndarray,
        right_hand: np.ndarray,
        waist: np.ndarray,
        *,
        update_sim: bool = True,
    ) -> np.ndarray:
        """Convert active joint values to robot-base-frame EEF33."""
        if update_sim:
            self._set_active_qpos(left_arm, right_arm, waist)
        l_pos, l_rot = self._site_pose("left")
        r_pos, r_rot = self._site_pose("right")
        out = np.concatenate(
            [
                l_pos,
                l_rot,
                np.asarray(left_hand, dtype=np.float32).reshape(6),
                r_pos,
                r_rot,
                np.asarray(right_hand, dtype=np.float32).reshape(6),
                np.asarray(waist, dtype=np.float32).reshape(3),
            ]
        ).astype(np.float32)
        if out.shape != (EEF33_DIM,) or not np.isfinite(out).all():
            raise ValueError(f"invalid GR1 EEF33 result: shape={out.shape}, finite={np.isfinite(out).all()}")
        return out

    def observation_to_eef33(self, obs: Mapping) -> np.ndarray:
        """Build proprio from a live GR00T observation without mutating the sim."""
        missing = [key for key in STATE_KEYS if key not in obs]
        if missing:
            raise KeyError(f"GR1 observation missing state keys: {missing}")
        return self.active_to_eef33(
            obs["state.left_arm"],
            obs["state.left_hand"],
            obs["state.right_arm"],
            obs["state.right_hand"],
            obs["state.waist"],
            update_sim=False,
        )

    def _clip_arm_qpos(self, part: str, qpos: np.ndarray) -> np.ndarray:
        clipped = np.asarray(qpos, dtype=np.float64).copy()
        for i, name in enumerate(self.joint_names[part]):
            joint = self.model.joint(name)
            if bool(self.model.jnt_limited[joint.id]):
                clipped[i] = np.clip(clipped[i], *self.model.jnt_range[joint.id])
        return clipped

    def solve_eef33(
        self,
        target: np.ndarray,
        *,
        max_iterations: int = 80,
        damping: float = 0.03,
        max_step: float = 0.12,
        position_tolerance: float = 2e-3,
        rotation_tolerance: float = 2e-2,
    ) -> IKSolution:
        """Solve absolute dual-arm EEF33 targets with waist fixed by the model.

        The solve operates on the live simulation data but restores the original
        qpos before returning. Callers therefore receive joint targets without
        perturbing the environment between ``obs`` and ``env.step``.
        """
        import mujoco

        tgt = np.asarray(target, dtype=np.float64).reshape(-1)
        if tgt.shape != (EEF33_DIM,) or not np.isfinite(tgt).all():
            raise ValueError(f"target must be finite EEF33, got shape={tgt.shape}")
        base_pos = np.asarray(self.data.xpos[self.base_body_id], dtype=np.float64)
        base_rot = np.asarray(self.data.xmat[self.base_body_id], dtype=np.float64).reshape(3, 3)
        target_pos = [base_pos + base_rot @ tgt[0:3], base_pos + base_rot @ tgt[15:18]]
        target_rot = [base_rot @ rot6d_to_matrix(tgt[3:9]), base_rot @ rot6d_to_matrix(tgt[18:24])]
        sites = [self.site_id["left"], self.site_id["right"]]
        arm_parts = ("left_arm", "right_arm")
        qpos_ids = np.concatenate([self.qpos_index[p] for p in arm_parts])
        dof_ids = np.concatenate([self.dof_index[p] for p in arm_parts])
        saved_qpos = self.data.qpos.copy()

        try:
            self.data.qpos[self.qpos_index["waist"]] = tgt[30:33]
            self.sim.forward()
            converged = False
            pos_error = rot_error = float("inf")
            for _ in range(max_iterations):
                error_parts = []
                pos_errors = []
                rot_errors = []
                jacobians = []
                for site, wanted_pos, wanted_rot in zip(sites, target_pos, target_rot):
                    current_pos = np.asarray(self.data.site_xpos[site], dtype=np.float64)
                    current_rot = np.asarray(self.data.site_xmat[site], dtype=np.float64).reshape(3, 3)
                    p_err = wanted_pos - current_pos
                    r_err = matrix_to_rotvec(wanted_rot @ current_rot.T)
                    pos_errors.append(float(np.linalg.norm(p_err)))
                    rot_errors.append(float(np.linalg.norm(r_err)))
                    error_parts.extend([p_err, r_err])
                    jacp = np.zeros((3, self.model.nv), dtype=np.float64)
                    jacr = np.zeros((3, self.model.nv), dtype=np.float64)
                    mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site)
                    jacobians.append(np.vstack([jacp[:, dof_ids], jacr[:, dof_ids]]))
                pos_error = max(pos_errors)
                rot_error = max(rot_errors)
                if pos_error <= position_tolerance and rot_error <= rotation_tolerance:
                    converged = True
                    break
                error = np.concatenate(error_parts)
                jac = np.vstack(jacobians)
                dq = jac.T @ np.linalg.solve(
                    jac @ jac.T + (float(damping) ** 2) * np.eye(jac.shape[0]),
                    error,
                )
                peak = float(np.max(np.abs(dq)))
                if peak > max_step:
                    dq *= max_step / peak
                self.data.qpos[qpos_ids] += dq
                for part in arm_parts:
                    self.data.qpos[self.qpos_index[part]] = self._clip_arm_qpos(
                        part,
                        self.data.qpos[self.qpos_index[part]],
                    )
                self.sim.forward()
            left = self.data.qpos[self.qpos_index["left_arm"]].copy().astype(np.float32)
            right = self.data.qpos[self.qpos_index["right_arm"]].copy().astype(np.float32)
            return IKSolution(left, right, pos_error, rot_error, converged)
        finally:
            self.data.qpos[:] = saved_qpos
            self.sim.forward()

    def eef33_to_action_dict(
        self,
        target: np.ndarray,
        *,
        hold_on_failure: bool = True,
        **ik_kwargs,
    ) -> tuple[dict[str, np.ndarray], IKSolution]:
        """Convert EEF33 to the GR00T 29-D action dictionary."""
        tgt = np.asarray(target, dtype=np.float32).reshape(EEF33_DIM)
        solution = self.solve_eef33(tgt, **ik_kwargs)
        if not solution.converged and not hold_on_failure:
            raise RuntimeError(
                f"GR1 IK failed: position_error={solution.position_error:.6f}, "
                f"rotation_error={solution.rotation_error:.6f}"
            )
        left = solution.left_arm
        right = solution.right_arm
        if not solution.converged:
            left = self.data.qpos[self.qpos_index["left_arm"]].copy().astype(np.float32)
            right = self.data.qpos[self.qpos_index["right_arm"]].copy().astype(np.float32)
        action = {
            "action.left_arm": left,
            "action.left_hand": tgt[9:15].copy(),
            "action.right_arm": right,
            "action.right_hand": tgt[24:30].copy(),
            "action.waist": tgt[30:33].copy(),
        }
        return action, solution


__all__ = [
    "ACTION_KEYS",
    "EEF33_DIM",
    "EEF33_SLICES",
    "EEF33_UNIFY_DST",
    "GR1Kinematics",
    "IKSolution",
    "JOINT29_DIM",
    "JOINT44_DIM",
    "JOINT44_SLICES",
    "ROT6D_DIMS_EEF33",
    "STATE_KEYS",
    "matrix_to_rot6d",
    "matrix_to_rotvec",
    "rot6d_to_matrix",
]
