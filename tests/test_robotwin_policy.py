"""Tests for the package-native RoboTwin policy compatibility layer."""

import numpy as np
from PIL import Image

from open_wam.evaluation.robotwin_policy import RoboTwinVAMPolicy, eval_one_step, reset_model


class MockActionDiT:
    action_dim = 7
    bridge_type = "cross_attn_detach"
    bridge_layers_set = set()
    action_mean = np.zeros(7, dtype=np.float32)
    action_std = np.ones(7, dtype=np.float32)


class MockPipe:
    pass


def test_robotwin_policy_chunk_reuse(monkeypatch):
    policy = RoboTwinVAMPolicy(
        pipe=MockPipe(),
        action_dit=MockActionDiT(),
        task_name="pick bottle",
        num_frames=4,
        height=32,
        width=32,
        num_denoise_steps=2,
        target_camera="head_camera",
        multiview=False,
    )

    monkeypatch.setattr(
        "open_wam.evaluation.robotwin_policy.generate_video_and_actions",
        lambda **kwargs: (
            [],
            np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1],
                    [2, 2, 2, 2, 2, 2, 2],
                    [3, 3, 3, 3, 3, 3, 3],
                    [4, 4, 4, 4, 4, 4, 4],
                ],
                dtype=np.float32,
            ),
        ),
    )

    obs = {"head_camera": Image.new("RGB", (32, 32))}
    a0 = policy.predict_action(obs)
    a1 = policy.predict_action(obs)

    assert np.allclose(a0, 1.0)
    assert np.allclose(a1, 2.0)


def test_robotwin_eval_one_step_and_reset(monkeypatch):
    policy = RoboTwinVAMPolicy(
        pipe=MockPipe(),
        action_dit=MockActionDiT(),
        num_frames=2,
        height=16,
        width=16,
    )
    monkeypatch.setattr(
        "open_wam.evaluation.robotwin_policy.generate_video_and_actions",
        lambda **kwargs: ([], np.ones((2, 7), dtype=np.float32)),
    )

    obs = {"head_camera": Image.new("RGB", (16, 16))}
    action = eval_one_step(policy, obs)
    assert action.shape == (7,)
    reset_model(policy)
    assert len(policy.obs_history) == 0
