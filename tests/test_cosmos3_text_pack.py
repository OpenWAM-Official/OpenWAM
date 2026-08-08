"""text_pack: prompt templates, und padding, and mRoPE positions.

Position assertions are pinned to facts observed in the native-pipeline golden
dump (cosmos3_golden/golden_step0.pt): und_len 68 → vision T starts at
68 + 15000 = 15068.0, advances 1.0 per latent frame at 24 fps / 4× temporal
compression; H/W restart at 0 with W the fastest axis; positions are float32
when fps modulation is on.
"""

import torch

from openwam.model.video_backbone.cosmos3 import text_pack


def test_prompt_templates_match_upstream_strings():
    text = text_pack.apply_prompt_templates("A robot.", num_frames=29, height=480, width=832, fps=24.0)
    assert text == ("A robot. The video is 1.2 seconds long and is of 24 FPS. This video is of 480x832 resolution.")
    neg = text_pack.apply_prompt_templates("", num_frames=29, height=480, width=832, fps=24.0, negative=True)
    assert neg == ("The video is not 1.2 seconds long and is not of 24 FPS. This video is not of 480x832 resolution.")
    img = text_pack.apply_prompt_templates("X", num_frames=1, height=480, width=832, fps=24.0)
    assert img == "X. This image is of 480x832 resolution."


def test_pad_und_batch_right_pads_and_masks():
    ids, mask, lens = text_pack.pad_und_batch([[1, 2, 3], [4, 5]], pad_token_id=0)
    assert ids.tolist() == [[1, 2, 3], [4, 5, 0]]
    assert mask.tolist() == [[True, True, True], [True, True, False]]
    assert lens.tolist() == [3, 2]


def test_text_positions_shared_axes():
    pos = text_pack.text_mrope_positions(5, float_positions=True)
    assert pos.shape == (3, 5)
    assert pos.dtype == torch.float32
    assert torch.equal(pos[0], pos[1]) and torch.equal(pos[0], pos[2])
    assert pos[0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_vision_positions_match_golden_facts():
    und_len, margin = 68, 15000
    grid = (8, 15, 26)
    text_pos, vis_pos = text_pack.build_joint_positions(
        und_len, grid, modality_margin=margin, fps=24.0, temporal_compression_factor=4
    )
    assert text_pos.shape == (3, 68) and vis_pos.shape == (3, 8 * 15 * 26)
    assert vis_pos.dtype == torch.float32
    tokens_per_frame = 15 * 26
    # T axis: starts at 15068.0, +1.0 per latent frame (24 fps / tc4 vs base 24/4).
    assert vis_pos[0, 0].item() == 15068.0
    assert vis_pos[0, tokens_per_frame].item() == 15069.0
    assert vis_pos[0, -1].item() == 15068.0 + 7
    # H axis: resets to 0, advances every W tokens.
    assert vis_pos[1, 0].item() == 0.0 and vis_pos[1, 26].item() == 1.0
    assert vis_pos[1, tokens_per_frame - 1].item() == 14.0
    # W axis: fastest, wraps every 26.
    assert vis_pos[2, :3].tolist() == [0.0, 1.0, 2.0]
    assert vis_pos[2, 25].item() == 25.0 and vis_pos[2, 26].item() == 0.0


def test_vision_positions_fps_scaling():
    # 12 fps at tc=4 → 3 tokens/sec vs base 6 → T advances 2.0 per latent frame.
    _, vis_pos = text_pack.build_joint_positions(
        10, (3, 1, 1), modality_margin=15000, fps=12.0, temporal_compression_factor=4
    )
    assert vis_pos[0].tolist() == [15010.0, 15012.0, 15014.0]


def test_patch_grid_ceil():
    assert text_pack.patch_grid(8, 30, 52, 2) == (8, 15, 26)
    assert text_pack.patch_grid(3, 5, 7, 2) == (3, 3, 4)
