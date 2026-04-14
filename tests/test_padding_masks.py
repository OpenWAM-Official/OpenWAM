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
    """Import and call the actual trainer helper."""
    from openwam.train.native_trainer import _downsample_video_mask_to_latent

    return _downsample_video_mask_to_latent(video_is_pad)


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
        """9 frames all valid.
        tail: [F]*8, groups: [F,F,F,F][F,F,F,F] → [F, F]
        """
        result = _downsample(torch.zeros(9, dtype=torch.bool))
        assert result.shape == (2,)
        assert not result.any()

    def test_9_frames_5_valid(self):
        """[F,F,F,F,F, T,T,T,T]. Separate frame 0 (F).
        tail: [F,F,F,F, T,T,T,T]
        groups: [F,F,F,F] [T,T,T,T] → .all() → [F, T]
        """
        result = _downsample(torch.tensor([False] * 5 + [True] * 4))
        assert result.shape == (2,)
        assert (result == torch.tensor([False, True])).all()

    def test_mixed_group_not_all_padded(self):
        """[F,F,F,F,F,F, T,T,T]. Separate frame 0 (F).
        tail: [F,F,F,F,F, T,T,T]
        groups: [F,F,F,F] [F,T,T,T] → .all() → [F, F]
        Mixed group: NOT all padded → valid.
        """
        result = _downsample(torch.tensor([False] * 6 + [True] * 3))
        assert result.shape == (2,)
        assert not result.any(), "Mixed group should NOT be padded"

    def test_only_frame0_valid(self):
        """[F, T,T,T,T, T,T,T,T]. Separate frame 0 (F).
        tail: [T]*8
        groups: [T,T,T,T] [T,T,T,T] → [T, T]
        """
        result = _downsample(torch.tensor([False] + [True] * 8))
        assert result.shape == (2,)
        assert result.all()

    def test_single_frame(self):
        """Only 1 frame → no tail → empty mask."""
        result = _downsample(torch.tensor([False]))
        assert result.shape == (0,)

    def test_end_to_end_33_stride4_valid20(self):
        """num_frames=33, stride=4, valid=20.

        video_is_pad: [F,F,F,F,F, T,T,T,T]
        tail: [F,F,F,F, T,T,T,T] → [F, T]
        """
        _, video_mask, _ = _build_masks(33, 4, 20)
        result = _downsample(~video_mask)
        assert result.shape == (2,)
        assert (result == torch.tensor([False, True])).all()

    def test_end_to_end_33_stride4_valid33(self):
        """No padding. tail: [F]*8 → [F, F]"""
        _, video_mask, _ = _build_masks(33, 4, 33)
        result = _downsample(~video_mask)
        assert result.shape == (2,)
        assert not result.any()

    def test_end_to_end_33_stride4_valid28(self):
        """valid=28.
        video_mask: [T]*7 + [F]*2
        video_is_pad: [F]*7 + [T]*2
        tail: [F,F,F,F,F,F, T,T]
        groups: [F,F,F,F] [F,F,T,T] → .all() → [F, F]
        """
        _, video_mask, _ = _build_masks(33, 4, 28)
        result = _downsample(~video_mask)
        assert result.shape == (2,)
        assert not result.any()


# ============================================================================
# Part 4: Loss masking correctness
# ============================================================================


class TestLossMasking:
    def _mock_scheduler(self):
        class S:
            linear_timesteps_weights = torch.ones(1000)

        return S()

    def _mock_pipe(self):
        class P:
            torch_dtype = torch.float32
            device = "cpu"
            scheduler = self._mock_scheduler()

        return P()

    def test_video_loss_ignores_padded(self):
        """B=1, C=2, T_latent=4, H=W=2. Steps 0,1 valid; 2,3 padded."""
        from openwam.train.loss.flow_match_loss import FlowMatchVideoActionLoss

        loss_fn = FlowMatchVideoActionLoss()

        target = torch.zeros(1, 2, 4, 2, 2)
        pred = torch.zeros(1, 2, 4, 2, 2)
        pred[:, :, :2] = 1.0
        pred[:, :, 2:] = 1000.0

        mask = torch.tensor([[False, False, True, True]])

        loss_m = loss_fn._compute_video_loss(
            pred, target, torch.tensor([0]), self._mock_pipe(), {}, 1, video_is_pad=mask
        )
        loss_u = loss_fn._compute_video_loss(
            pred, target, torch.tensor([0]), self._mock_pipe(), {}, 1, video_is_pad=None
        )

        assert abs(loss_m.item() - 1.0) < 1e-5
        assert loss_u.item() > 100.0

    def test_video_loss_first_frame_exclusion_with_tail_mask(self):
        """first_frame_latents triggers [:,:,1:] on pred/target only, NOT on mask.

        B=1, C=1, T=3 latent steps. mask is tail-only (2,) = [F, T].
        After [:,:,1:] on pred/target → (1,1,2,1,1) matches mask (1,2) ✓
        """
        from openwam.train.loss.flow_match_loss import FlowMatchVideoActionLoss

        loss_fn = FlowMatchVideoActionLoss()

        target = torch.zeros(1, 1, 3, 1, 1)
        pred = torch.zeros(1, 1, 3, 1, 1)
        pred[0, 0, 0, 0, 0] = 999.0  # frame 0 (cond) — excluded by [:,:,1:]
        pred[0, 0, 1, 0, 0] = 2.0  # tail step 0: valid, MSE=4
        pred[0, 0, 2, 0, 0] = 888.0  # tail step 1: padded — masked

        mask = torch.tensor([[False, True]])  # tail-only: [valid, padded]
        inputs = {"first_frame_latents": torch.zeros(1, 1, 1, 1, 1)}

        loss = loss_fn._compute_video_loss(
            pred, target, torch.tensor([0]), self._mock_pipe(), inputs, 1, video_is_pad=mask
        )

        assert abs(loss.item() - 4.0) < 1e-5, f"got {loss.item()}"

    def test_action_loss_ignores_padded(self):
        from openwam.train.loss.flow_match_loss import FlowMatchVideoActionLoss

        loss_fn = FlowMatchVideoActionLoss()

        target = torch.zeros(1, 6, 2)
        pred = torch.zeros(1, 6, 2)
        pred[:, :4] = 0.5
        pred[:, 4:] = 1000.0

        mask = torch.tensor([[False] * 4 + [True] * 2])
        loss = loss_fn._compute_action_loss(
            pred, target, torch.tensor([0]), self._mock_scheduler(), self._mock_pipe(), 1, action_is_pad=mask
        )
        assert abs(loss.item() - 0.25) < 1e-5

    def test_no_padding_same_result(self):
        from openwam.train.loss.flow_match_loss import FlowMatchVideoActionLoss

        loss_fn = FlowMatchVideoActionLoss()

        pred = torch.randn(2, 4, 8, 3, 3)
        target = torch.randn(2, 4, 8, 3, 3)
        ids = torch.tensor([10, 20])
        all_valid = torch.zeros(2, 8, dtype=torch.bool)

        loss_m = loss_fn._compute_video_loss(pred, target, ids, self._mock_pipe(), {}, 2, video_is_pad=all_valid)
        loss_u = loss_fn._compute_video_loss(pred, target, ids, self._mock_pipe(), {}, 2, video_is_pad=None)
        assert abs(loss_m.item() - loss_u.item()) < 1e-5

    def test_batch_mixed_padding(self):
        from openwam.train.loss.flow_match_loss import FlowMatchVideoActionLoss

        loss_fn = FlowMatchVideoActionLoss()

        target = torch.zeros(2, 4, 2)
        pred = torch.ones(2, 4, 2)
        pred[1, 2:] = 1000.0

        mask = torch.tensor([[False] * 4, [False, False, True, True]])
        loss = loss_fn._compute_action_loss(
            pred, target, torch.tensor([0, 0]), self._mock_scheduler(), self._mock_pipe(), 2, action_is_pad=mask
        )
        assert abs(loss.item() - 1.0) < 1e-5

    def test_end_to_end_mask_dimensions(self):
        """Full pipeline: dataset → trainer downsample → loss.

        num_frames=33, stride=4, valid=20.
        Downsample → tail mask (2,) = [F, T].
        Loss: pred/target [:,:,1:] → (1,C,2,H,W), mask (1,2) — match ✓
        """
        from openwam.train.loss.flow_match_loss import FlowMatchVideoActionLoss

        loss_fn = FlowMatchVideoActionLoss()

        _, video_mask, _ = _build_masks(33, 4, 20)
        latent_mask = _downsample(~video_mask)

        assert latent_mask.shape == (2,)
        assert (latent_mask == torch.tensor([False, True])).all()

        target = torch.zeros(1, 2, 3, 2, 2)
        pred = torch.zeros(1, 2, 3, 2, 2)
        pred[:, :, 0] = 999.0  # frame 0 (cond) → excluded
        pred[:, :, 1] = 1.0  # tail step 0: valid
        pred[:, :, 2] = 888.0  # tail step 1: padded → masked

        inputs = {"first_frame_latents": torch.zeros(1, 2, 1, 2, 2)}

        loss = loss_fn._compute_video_loss(
            pred,
            target,
            torch.tensor([0]),
            self._mock_pipe(),
            inputs,
            1,
            video_is_pad=latent_mask.unsqueeze(0),
        )

        assert abs(loss.item() - 1.0) < 1e-5, f"got {loss.item()}"
