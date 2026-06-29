"""Tests for video/action padding masks in RoboTwin dataset and loss.

Verifies that:
  1. action_mask and video_mask have correct lengths and values
  2. video_mask accounts for video_stride subsampling
  3. VAE latent temporal downsampling (FastWAM-aligned: separate frame 0, .all())
  4. Loss correctly masks out padded positions
  5. First-frame exclusion: mask built without frame 0, loss trims pred/target only
"""

import torch

# ============================================================================
# Part 1: Mask generation logic (mirrors RoboTwinDataset.__getitem__)
# ============================================================================


def _build_masks(num_frames: int, video_stride: int, valid_len: int):
    """Reproduce the mask logic from RoboTwinDataset.__getitem__."""
    if video_stride > 1:
        video_indices = list(range(0, num_frames, video_stride))
    else:
        video_indices = list(range(num_frames))

    action_mask = torch.ones(num_frames, dtype=torch.bool)
    action_mask[valid_len:] = False

    video_mask = torch.tensor([i < valid_len for i in video_indices], dtype=torch.bool)
    return action_mask, video_mask, video_indices


def _downsample(video_is_pad):
    from openwam.model.architectures.utils.common import downsample_video_mask_to_latent

    return downsample_video_mask_to_latent(video_is_pad)


def _make_arch():
    """Build a minimal architecture with a mock video backbone for loss tests."""
    from tests.test_openwam_trainer import _make_tiny_arch

    return _make_tiny_arch()


class _MockScheduler:
    num_train_timesteps = 1000
    linear_timesteps_weights = torch.ones(1000)

    def add_noise(self, original, noise, sigma):
        return (1 - sigma) * original + sigma * noise

    def training_target(self, original, noise):
        return noise - original

    def training_weight(self, timestep_ids):
        return self.linear_timesteps_weights[timestep_ids]

    def flow_step(self, pred, sigma, sigma_next, sample):
        return sample + pred * (sigma_next - sigma)


# ============================================================================
# Part 2: Mask generation tests
# ============================================================================


class TestMaskGeneration:
    def test_no_padding(self):
        action_mask, video_mask, indices = _build_masks(33, 4, 33)
        assert action_mask.shape == (33,) and action_mask.all()
        assert video_mask.shape == (9,) and video_mask.all()
        assert indices == [0, 4, 8, 12, 16, 20, 24, 28, 32]

    def test_partial_padding_aligned(self):
        action_mask, video_mask, _ = _build_masks(33, 4, 20)
        assert action_mask[:20].all() and not action_mask[20:].any()
        assert (video_mask == torch.tensor([True] * 5 + [False] * 4)).all()

    def test_partial_padding_unaligned(self):
        _, video_mask, _ = _build_masks(33, 4, 21)
        assert (video_mask == torch.tensor([True] * 6 + [False] * 3)).all()

    def test_extreme_one_valid(self):
        action_mask, video_mask, _ = _build_masks(33, 4, 1)
        assert action_mask[0].item() and not action_mask[1:].any()
        assert video_mask[0].item() and not video_mask[1:].any()

    def test_no_stride(self):
        action_mask, video_mask, _ = _build_masks(33, 1, 25)
        assert (action_mask == video_mask).all()

    def test_stride2(self):
        _, video_mask, indices = _build_masks(10, 2, 7)
        assert indices == [0, 2, 4, 6, 8]
        assert (video_mask == torch.tensor([True] * 4 + [False])).all()


# ============================================================================
# Part 3: VAE latent temporal downsample tests (FastWAM-aligned)
# ============================================================================


class TestLatentMaskDownsample:
    """FastWAM approach: separate frame 0 (conditioning), group tail by 4, .all().

    With 9 video frames:
      frame 0 separated → tail 8 frames → groups [4][4] → 2 tail latent steps
    """

    def test_9_frames_all_valid(self):
        result = _downsample(torch.zeros(9, dtype=torch.bool))
        assert result.shape == (2,)
        assert not result.any()

    def test_9_frames_5_valid(self):
        result = _downsample(torch.tensor([False] * 5 + [True] * 4))
        assert result.shape == (2,)
        assert (result == torch.tensor([False, True])).all()

    def test_mixed_group_not_all_padded(self):
        result = _downsample(torch.tensor([False] * 6 + [True] * 3))
        assert result.shape == (2,)
        assert not result.any(), "Mixed group should NOT be padded"

    def test_only_frame0_valid(self):
        result = _downsample(torch.tensor([False] + [True] * 8))
        assert result.shape == (2,)
        assert result.all()

    def test_single_frame(self):
        result = _downsample(torch.tensor([False]))
        assert result.shape == (0,)

    def test_end_to_end_33_stride4_valid20(self):
        _, video_mask, _ = _build_masks(33, 4, 20)
        result = _downsample(~video_mask)
        assert result.shape == (2,)
        assert (result == torch.tensor([False, True])).all()

    def test_end_to_end_33_stride4_valid33(self):
        _, video_mask, _ = _build_masks(33, 4, 33)
        result = _downsample(~video_mask)
        assert result.shape == (2,)
        assert not result.any()

    def test_end_to_end_33_stride4_valid28(self):
        _, video_mask, _ = _build_masks(33, 4, 28)
        result = _downsample(~video_mask)
        assert result.shape == (2,)
        assert not result.any()


# ============================================================================
# Part 4: Loss masking correctness (via architecture._compute_*_loss)
# ============================================================================


class TestLossMasking:
    def test_video_loss_ignores_padded(self):
        arch = _make_arch()
        target = torch.zeros(1, 2, 4, 2, 2)
        pred = torch.zeros(1, 2, 4, 2, 2)
        pred[:, :, :2] = 1.0
        pred[:, :, 2:] = 1000.0

        mask = torch.tensor([[False, False, True, True]])
        inputs_m = {"video_is_pad": mask}

        loss_m = arch._compute_video_loss(pred, target, torch.tensor([0]), inputs_m, device="cpu")
        loss_u = arch._compute_video_loss(pred, target, torch.tensor([0]), {}, device="cpu")

        assert abs(loss_m.item() - 1.0) < 1e-5
        assert loss_u.item() > 100.0

    def test_video_loss_first_frame_exclusion_with_tail_mask(self):
        arch = _make_arch()
        target = torch.zeros(1, 1, 3, 1, 1)
        pred = torch.zeros(1, 1, 3, 1, 1)
        pred[0, 0, 0, 0, 0] = 999.0
        pred[0, 0, 1, 0, 0] = 2.0
        pred[0, 0, 2, 0, 0] = 888.0

        inputs = {
            "first_frame_latents": torch.zeros(1, 1, 1, 1, 1),
            "video_is_pad": torch.tensor([[False, True]]),
        }

        loss = arch._compute_video_loss(pred, target, torch.tensor([0]), inputs, device="cpu")
        assert abs(loss.item() - 4.0) < 1e-5, f"got {loss.item()}"

    def test_video_loss_i2v_conditioning_latent_without_prefix(self):
        """I2V has no ref-prefix metadata, but its tail mask still excludes the
        leading first-frame conditioning latent."""
        arch = _make_arch()
        target = torch.zeros(1, 1, 3, 1, 1)
        pred = torch.zeros(1, 1, 3, 1, 1)
        pred[0, 0, 0, 0, 0] = 999.0
        pred[0, 0, 1, 0, 0] = 2.0
        pred[0, 0, 2, 0, 0] = 888.0

        inputs = {
            "num_clean_prefix_frames": 0,
            "first_frame_latents": None,
            "clip_feature": torch.zeros(1, 257, 1280),
            "y": torch.zeros(1, 20, 3, 1, 1),
            "video_is_pad": torch.tensor([[False, True]]),
        }

        loss = arch._compute_video_loss(pred, target, torch.tensor([0]), inputs, device="cpu")
        assert abs(loss.item() - 4.0) < 1e-5, f"got {loss.item()}"

    def test_video_loss_raises_on_mask_overlong(self):
        """Sanity guard: mask longer than noise_pred T must raise ValueError
        at the loss boundary, not silently RuntimeError inside per_frame *
        valid_mask. Covers the fallback gap reported in PR #48 review."""
        import pytest

        arch = _make_arch()
        pred = torch.zeros(1, 1, 3, 1, 1)
        target = torch.zeros(1, 1, 3, 1, 1)
        # mask length 4 > noise_pred T=3; no n_skip path fixes this case.
        inputs = {"video_is_pad": torch.tensor([[False, False, False, True]])}
        with pytest.raises(ValueError, match="video_is_pad length"):
            arch._compute_video_loss(pred, target, torch.tensor([0]), inputs, device="cpu")

    def test_action_loss_ignores_padded(self):
        arch = _make_arch()
        target = torch.zeros(1, 6, 2)
        pred = torch.zeros(1, 6, 2)
        pred[:, :4] = 0.5
        pred[:, 4:] = 1000.0

        mask = torch.tensor([[False] * 4 + [True] * 2])
        inputs = {"action_is_pad": mask}
        loss = arch._compute_action_loss(pred, target, torch.tensor([0]), _MockScheduler(), inputs=inputs, device="cpu")
        assert abs(loss.item() - 0.25) < 1e-5

    def test_no_padding_same_result(self):
        arch = _make_arch()
        pred = torch.randn(2, 4, 8, 3, 3)
        target = torch.randn(2, 4, 8, 3, 3)
        ids = torch.tensor([10, 20])
        all_valid = torch.zeros(2, 8, dtype=torch.bool)

        inputs_m = {"video_is_pad": all_valid}
        loss_m = arch._compute_video_loss(pred, target, ids, inputs_m, device="cpu")
        loss_u = arch._compute_video_loss(pred, target, ids, {}, device="cpu")
        assert abs(loss_m.item() - loss_u.item()) < 1e-5

    def test_batch_mixed_padding(self):
        arch = _make_arch()
        target = torch.zeros(2, 4, 2)
        pred = torch.ones(2, 4, 2)
        pred[1, 2:] = 1000.0

        mask = torch.tensor([[False] * 4, [False, False, True, True]])
        inputs = {"action_is_pad": mask}
        loss = arch._compute_action_loss(
            pred, target, torch.tensor([0, 0]), _MockScheduler(), inputs=inputs, device="cpu"
        )
        assert abs(loss.item() - 1.0) < 1e-5

    def test_end_to_end_mask_dimensions(self):
        arch = _make_arch()
        _, video_mask, _ = _build_masks(33, 4, 20)
        latent_mask = _downsample(~video_mask)

        assert latent_mask.shape == (2,)
        assert (latent_mask == torch.tensor([False, True])).all()

        target = torch.zeros(1, 2, 3, 2, 2)
        pred = torch.zeros(1, 2, 3, 2, 2)
        pred[:, :, 0] = 999.0
        pred[:, :, 1] = 1.0
        pred[:, :, 2] = 888.0

        inputs = {
            "first_frame_latents": torch.zeros(1, 2, 1, 2, 2),
            "video_is_pad": latent_mask.unsqueeze(0),
        }

        loss = arch._compute_video_loss(pred, target, torch.tensor([0]), inputs, device="cpu")
        assert abs(loss.item() - 1.0) < 1e-5, f"got {loss.item()}"


# ============================================================================
# Part 5: prepare_inputs honors `video_backbone.needs_first_frame_skip` (Fix #4)
#
# Reviewer @wayrise flagged that `prepare_inputs` previously decided
# ``skip_first`` purely from ``inputs.get("first_frame_latents")``. That
# matched Wan TI2V / VACE / cosmos_predict25 TI2V (those set the field) and
# cosmos_predict25 T2V (no FFL, no skip). But Wan I2V wires conditioning through
# the ``y`` channel *without* setting ``first_frame_latents``, and a
# future Wan T2V configuration would (silently) also lose its skip.
#
# Fix: ``skip_first = first_frame_latents is not None or
# vb.needs_first_frame_skip``. The tests below pin both arms.
# ============================================================================


def _make_arch_with_backbone(vb):
    """Wire a custom mock VideoBackbone onto the tiny architecture used in tests."""
    arch = _make_arch()
    arch.video_backbone = vb
    return arch


def _build_one_sample(num_frames=33, video_stride=4, valid_len=33):
    """Construct a single-sample batch that prepare_inputs accepts."""
    import numpy as np

    _, video_mask, _ = _build_masks(num_frames, video_stride, valid_len)
    # `video` content doesn't matter: the mock backbone's preprocess_input_for_train
    # ignores raw frames and emits its own latents/context. We use a small
    # list so ``FirstFrameConditioningTransform`` still picks frame 0.
    return {
        "video": [np.zeros((1, 1, 3), dtype=np.uint8) for _ in range(num_frames)],
        "prompt": "ignored",
        "video_mask": video_mask,
        "action": np.zeros((num_frames, 7), dtype=np.float32),
        "action_mask": torch.ones(num_frames, dtype=torch.bool),
    }


class TestPrepareInputsSkipFirst:
    """`prepare_inputs` uses (first_frame_latents OR needs_first_frame_skip)
    to decide whether ``video_is_pad`` is sized to T_lat-1 or T_lat."""

    def _make_backbone(self, *, emit_first_frame_latents, needs_skip):
        """Build a tiny mock that controls both signals independently."""
        from tests.test_openwam_trainer import _MockVideoBackbone

        class _ConfigurableBackbone(_MockVideoBackbone):
            @property
            def needs_first_frame_skip(self) -> bool:
                return needs_skip

            def preprocess_input_for_train(self, *, frames=None, text=None, **kw):
                out = {
                    "input_latents": torch.randn(1, 16, 3, 8, 8),
                    "context": torch.randn(1, 4, self._dim),
                    "context_mask": torch.ones(1, 4, dtype=torch.bool),
                    "seq_lens": torch.ones(1, dtype=torch.long),
                }
                if emit_first_frame_latents:
                    out["first_frame_latents"] = torch.zeros(1, 16, 1, 8, 8)
                return out

        return _ConfigurableBackbone(dim=64, num_layers=2)

    def test_wan_i2v_path_skips_via_property_when_no_first_frame_latents(self):
        """Wan I2V signature: `needs_first_frame_skip=True` but no
        ``first_frame_latents`` (image rides on ``y``). Mask must still be
        sized to T_lat-1 so the loss-side shape-fallback trims latent[0]."""
        vb = self._make_backbone(emit_first_frame_latents=False, needs_skip=True)
        arch = _make_arch_with_backbone(vb)
        arch.set_training_runtime()

        inputs = arch.prepare_inputs([_build_one_sample(num_frames=33, video_stride=4, valid_len=33)])

        # 33 frames @ stride 4 → 9 latent slots; skip_first=True → 8 tail latents.
        assert "video_is_pad" in inputs
        assert inputs["video_is_pad"].shape == (1, 2), (
            f"expected (1, 2) tail mask after skip_first, got {tuple(inputs['video_is_pad'].shape)}"
        )

    def test_wan_t2v_path_keeps_frame_zero_when_no_signal(self):
        """Wan T2V signature: no ``first_frame_latents`` AND
        ``needs_first_frame_skip=False``. ``video_is_pad`` must cover the
        full T_lat so latent[0] enters the loss as a predicted frame."""
        vb = self._make_backbone(emit_first_frame_latents=False, needs_skip=False)
        arch = _make_arch_with_backbone(vb)
        arch.set_training_runtime()

        inputs = arch.prepare_inputs([_build_one_sample(num_frames=33, video_stride=4, valid_len=33)])

        assert "video_is_pad" in inputs
        # 33 frames @ stride 4 → 9 latent slots; skip_first=False → full 9.
        # ``downsample_video_mask_to_latent`` without skip_first returns a
        # mask of length ``T_lat`` (3 here for the tiny mock_latent shape
        # case)... but our mock emits 3 latents directly; the dataloader
        # mask length is still derived from the raw video mask. Assert the
        # resulting mask matches the raw downsample (skip_first=False).
        from openwam.model.architectures.utils.common import downsample_video_mask_to_latent

        _, video_mask, _ = _build_masks(33, 4, 33)
        expected = downsample_video_mask_to_latent(~video_mask, skip_first=False)
        assert inputs["video_is_pad"].shape == (1, expected.shape[0])
        assert torch.equal(inputs["video_is_pad"][0].cpu(), expected)

    def test_per_batch_signal_still_wins_when_property_is_false(self):
        """cosmos_predict25 TI2V case: ``needs_first_frame_skip=False`` (TI2V is
        data-driven on cosmos), but the batch carries ``first_frame_latents``.
        The OR clause must still trip skip_first."""
        vb = self._make_backbone(emit_first_frame_latents=True, needs_skip=False)
        arch = _make_arch_with_backbone(vb)
        arch.set_training_runtime()

        inputs = arch.prepare_inputs([_build_one_sample(num_frames=33, video_stride=4, valid_len=33)])

        assert inputs["video_is_pad"].shape == (1, 2)
