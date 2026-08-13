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


def test_prompt_templates_can_drop_the_duration_sentence():
    """The duration sentence asserts a clip fps nothing in the dataloader stack
    reports; it must be droppable without losing the (measured) resolution one."""
    text = text_pack.apply_prompt_templates(
        "A robot.", num_frames=9, height=384, width=320, fps=24.0, add_duration_template=False
    )
    assert text == "A robot. This video is of 384x320 resolution."
    assert "seconds long" not in text


class _StubTokenizer:
    """Chat template = ``[BOS] <body> <|im_end|> [ASSISTANT_HDR]``, one id per char."""

    eos_token_id = 90
    unk_token_id = 99
    pad_token_id = 0
    _END = 79  # the user turn's terminator, e.g. <|im_end|>
    _HDR = [80, 81]

    def apply_chat_template(self, conversations, *, tokenize, add_generation_prompt, return_dict):
        body = [1] + [ord(c) % 60 + 2 for c in conversations[-1]["content"]] + [self._END]
        return body + (self._HDR if add_generation_prompt else [])

    def convert_tokens_to_ids(self, token):
        return 91 if token == text_pack.START_OF_GENERATION_TOKEN else self.unk_token_id


def test_tokenize_truncation_preserves_the_whole_constant_tail():
    """A right-cut deletes the turn terminator AND the assistant header.

    Both are constant tail, independent of the prompt, and the model never saw
    a sequence that jumps from mid-sentence straight to ``[eos, vision_start]``.
    """
    tok = _StubTokenizer()
    tail = [_StubTokenizer._END, *_StubTokenizer._HDR]

    untruncated = text_pack.tokenize_prompt(tok, "hi")
    assert untruncated[-5:] == [*tail, tok.eos_token_id, 91]

    ids = text_pack.tokenize_prompt(tok, "x" * 200, max_length=32)
    assert len(ids) == 32
    assert ids[-5:] == [*tail, tok.eos_token_id, 91]
    # The kept body is a genuine prefix of the untruncated body: a cut, not a shift.
    full = tok.apply_chat_template(
        [{"role": "user", "content": "x" * 200}], tokenize=True, add_generation_prompt=True, return_dict=False
    )
    assert ids[: 32 - 5] == full[: 32 - 5]


def test_tokenize_truncation_never_exceeds_max_length():
    """Clamp the assembled result too: a tail longer than the budget would
    otherwise push the output back over the cap."""
    tok = _StubTokenizer()
    for max_length in (8, 9, 12, 40):
        ids = text_pack.tokenize_prompt(tok, "y" * 200, max_length=max_length)
        assert len(ids) <= max_length, f"max_length={max_length} produced {len(ids)} ids"


def test_tokenize_falls_back_when_the_tail_is_not_a_suffix():
    """Templates whose constant part is not a pure suffix take the plain cut."""

    class _Interleaving(_StubTokenizer):
        def apply_chat_template(self, conversations, *, tokenize, add_generation_prompt, return_dict):
            body = [1] + [ord(c) % 60 + 2 for c in conversations[-1]["content"]]
            return ([7] + body) if add_generation_prompt else body

    ids = text_pack.tokenize_prompt(_Interleaving(), "y" * 200, max_length=20)
    assert len(ids) == 20 and ids[0] == 7
