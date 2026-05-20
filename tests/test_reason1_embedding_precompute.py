"""Unit tests for ``openwam.dataloader.reason1_embedding_computation``.

Mostly hits the pure-Python helpers and the crossattn_proj loader against
a stand-in checkpoint dict — does NOT need the real Reason1-7B weights or
a real Cosmos DiT body. The GPU-gated test at the bottom exercises the
end-to-end path on hardware that has both.
"""

from __future__ import annotations

import json
import os

import pytest
import torch

from openwam.dataloader import reason1_embedding_computation as ript


def test_constants_match_upstream_geometry():
    aux = ript._aux_for_tests()
    assert aux["NUM_EMBEDDING_PADDING_TOKENS"] == 512
    assert aux["REASON1_FULL_CONCAT_DIM"] == 28 * 3584
    assert aux["COSMOS_POSTPROJ_DIM"] == 1024
    # Upstream system prompt — copied verbatim from text_encoder.py:145-150.
    assert "image generator" in aux["SYSTEM_PROMPT"]


def test_sha256_text_is_utf8_stable():
    sha = ript._sha256_text("pick up the block")
    assert (
        sha
        == "98fffccbfa71a725c80a9f6370854ba0345582b4e1c6b0459dc306ea63b93f3b"
    )
    # Unicode / emoji content must hash too (utf-8 byte stream).
    sha_unicode = ript._sha256_text("拿起方块")
    assert len(sha_unicode) == 64


def test_mean_normalize_zero_mean_unit_std():
    x = torch.randn(2, 4, 8)
    y = ript._mean_normalize_along_last(x)
    assert torch.allclose(y.mean(dim=-1), torch.zeros_like(y.mean(dim=-1)), atol=1e-5)
    # std along last dim should be ~1 (up to eps).
    assert torch.allclose(y.std(dim=-1), torch.ones_like(y.std(dim=-1)), atol=1e-3)


def test_mean_normalize_handles_constant_input():
    # All-zero input: mean=0 exactly, std=0 exactly, so numerator is 0 and
    # the eps in the denominator keeps the result finite (=0). Guards
    # against accidentally producing NaN/Inf on degenerate hidden states.
    x = torch.zeros((1, 1, 8))
    y = ript._mean_normalize_along_last(x)
    assert torch.isfinite(y).all()
    assert torch.equal(y, torch.zeros_like(y))


def test_build_crossattn_proj_loads_correct_subset(tmp_path):
    """Synthesize a fake Cosmos `*_ema_bf16.pt` containing only the two
    crossattn_proj keys (plus some unrelated keys to verify they're ignored)
    and confirm the loader returns a Sequential(Linear, GELU) with copied
    weights."""
    ckpt = tmp_path / "fake_ema_bf16.pt"
    w = torch.randn(1024, 100352, dtype=torch.bfloat16)
    b = torch.randn(1024, dtype=torch.bfloat16)
    fake_sd = {
        "net.crossattn_proj.0.weight": w,
        "net.crossattn_proj.0.bias": b,
        # Junk keys the loader must ignore.
        "net.blocks.0.self_attn.q_proj.weight": torch.zeros(2048, 2048),
        "optimizer.step": 0,
    }
    torch.save(fake_sd, ckpt)

    proj = ript._build_crossattn_proj(ckpt, device="cpu", dtype=torch.float32)
    assert isinstance(proj, torch.nn.Sequential)
    assert isinstance(proj[0], torch.nn.Linear)
    assert isinstance(proj[1], torch.nn.GELU)
    assert proj[0].weight.shape == (1024, 100352)
    assert proj[0].bias.shape == (1024,)
    # Weights cast to fp32 but values must match (bf16 → fp32 is lossless).
    assert torch.equal(proj[0].weight, w.to(torch.float32))
    assert torch.equal(proj[0].bias, b.to(torch.float32))


def test_build_crossattn_proj_rejects_wrong_shape(tmp_path):
    ckpt = tmp_path / "bad_ema_bf16.pt"
    torch.save(
        {
            "net.crossattn_proj.0.weight": torch.randn(1024, 64),  # wrong in-dim
            "net.crossattn_proj.0.bias": torch.randn(1024),
        },
        ckpt,
    )
    with pytest.raises(ValueError, match="unexpected shape"):
        ript._build_crossattn_proj(ckpt, device="cpu", dtype=torch.float32)


def test_build_crossattn_proj_missing_keys_raise(tmp_path):
    ckpt = tmp_path / "no_proj.pt"
    torch.save({"net.blocks.0.x": torch.zeros(2)}, ckpt)
    with pytest.raises(KeyError, match="crossattn_proj"):
        ript._build_crossattn_proj(ckpt, device="cpu", dtype=torch.float32)


def test_build_crossattn_proj_applies_gelu_after_linear(tmp_path):
    """Linear+GELU is the upstream wrapper; verify ordering by feeding a
    deliberately constructed input through and matching against a manual
    composition."""
    ckpt = tmp_path / "fake.pt"
    w = torch.randn(1024, 100352, dtype=torch.float32) * 0.01
    b = torch.randn(1024, dtype=torch.float32) * 0.01
    torch.save({"net.crossattn_proj.0.weight": w, "net.crossattn_proj.0.bias": b}, ckpt)

    proj = ript._build_crossattn_proj(ckpt, device="cpu", dtype=torch.float32)
    x = torch.randn(1, 4, 100352)
    expected = torch.nn.functional.gelu(torch.nn.functional.linear(x, w, b))
    got = proj(x)
    assert torch.allclose(got, expected, atol=1e-5)


def test_project_to_postproj_returns_bf16_cpu(tmp_path):
    ckpt = tmp_path / "fake.pt"
    w = torch.randn(1024, 100352, dtype=torch.bfloat16)
    b = torch.randn(1024, dtype=torch.bfloat16)
    torch.save({"net.crossattn_proj.0.weight": w, "net.crossattn_proj.0.bias": b}, ckpt)
    proj = ript._build_crossattn_proj(ckpt, device="cpu", dtype=torch.bfloat16)
    fake_reason1 = torch.randn(1, 512, 100352, dtype=torch.bfloat16)
    out = ript._project_to_postproj(fake_reason1, proj)
    assert out.shape == (512, 1024)
    assert out.dtype == torch.bfloat16
    assert out.device.type == "cpu"


def _write_episode_instructions(root: str, episode_idx: int, payload):
    instr_dir = os.path.join(root, "instructions")
    os.makedirs(instr_dir, exist_ok=True)
    with open(os.path.join(instr_dir, f"episode{episode_idx}.json"), "w") as f:
        json.dump(payload, f)


def test_enumerate_prompts_walks_seen_unseen_and_dedups(tmp_path, monkeypatch):
    """Synthesize a fake RoboTwin layout and confirm that every entry in
    `seen` and `unseen` lists is reflected in the enumerated set, with
    dedup applied. Also verifies the task-name fallback is included."""

    # Build a fake task root: <dataset_dir>/<robot>_<variant>/<task>/{data, instructions}
    dataset_dir = tmp_path / "robotwin"
    task_root = dataset_dir / "aloha-agilex_clean_50" / "fold_towel"
    data_root = task_root / "data"
    data_root.mkdir(parents=True)
    _write_episode_instructions(
        str(task_root),
        0,
        {"seen": ["fold the towel.", "neatly fold the towel."], "unseen": ["towel: fold."]},
    )
    _write_episode_instructions(
        str(task_root),
        1,
        {"seen": ["fold the towel."], "unseen": []},  # duplicate "fold the towel."
    )

    # Stub the dataset discovery so we don't need real RoboTwin layout heuristics.
    def fake_discover(_dataset_dir, _robot, _variant, tasks):
        return [("fold_towel", str(data_root))]

    monkeypatch.setattr(
        "openwam.dataloader.robotwin_dataset.discover_robotwin_roots", fake_discover
    )

    cfg = {
        "dataset_dir": str(dataset_dir),
        "robot": "aloha-agilex",
        "variant": "clean_50",
        "task_name": "fold_towel",
    }
    prompts = ript._enumerate_prompts(cfg)

    # All three distinct base prompts (seen + unseen) survive, formatted.
    from openwam.dataloader.transforms.multiview import format_prompt_for_inference

    expected_subset = {
        format_prompt_for_inference("fold the towel."),
        format_prompt_for_inference("neatly fold the towel."),
        format_prompt_for_inference("towel: fold."),
        # Task-name fallback is always included.
        format_prompt_for_inference("The bimanual robot is performing a fold_towel task."),
    }
    assert expected_subset.issubset(set(prompts))


def test_enumerate_prompts_includes_task_fallback_even_without_instructions(tmp_path, monkeypatch):
    """If a task root has no instructions/ folder, the enumerator still
    emits the task-name fallback. This guards against silently missing
    prompts for tasks without RoboTwin instruction files."""
    dataset_dir = tmp_path / "robotwin"
    data_root = dataset_dir / "fake_robot_clean_50" / "open_laptop" / "data"
    data_root.mkdir(parents=True)
    # NOTE: no instructions/ dir created.

    def fake_discover(_dataset_dir, _robot, _variant, tasks):
        return [("open_laptop", str(data_root))]

    monkeypatch.setattr(
        "openwam.dataloader.robotwin_dataset.discover_robotwin_roots", fake_discover
    )

    cfg = {
        "dataset_dir": str(dataset_dir),
        "robot": "fake_robot",
        "variant": "clean_50",
        "task_name": "open_laptop",
    }
    prompts = ript._enumerate_prompts(cfg)

    from openwam.dataloader.transforms.multiview import format_prompt_for_inference

    assert format_prompt_for_inference(
        "The bimanual robot is performing a open_laptop task."
    ) in prompts


def test_tokenize_with_chat_template_pads_to_512():
    """Without real HF tokenizer, mock the apply_chat_template + tokenize
    behavior and verify the pad-or-truncate-to-512 semantics."""

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def __init__(self, token_count: int):
            self._token_count = token_count

        def apply_chat_template(self, conversations, **kw):
            assert kw.get("tokenize") is False
            assert kw.get("add_generation_prompt") is False
            # The exact wrapped string doesn't matter — we don't pass it back through.
            user = conversations[1]["content"][0]["text"]
            return f"<sys>{conversations[0]['content'][0]['text']}<user>{user}<eos>"

        def __call__(self, text, **kw):
            return {"input_ids": torch.full((1, self._token_count), 7, dtype=torch.long)}

    # Short prompt → padding.
    short = FakeTokenizer(token_count=10)
    ids = ript._tokenize_with_chat_template(short, "x")
    assert len(ids) == 512
    assert ids[:10] == [7] * 10
    assert ids[10:] == [short.pad_token_id] * (512 - 10)

    # Long prompt → truncation.
    long = FakeTokenizer(token_count=1000)
    ids = ript._tokenize_with_chat_template(long, "long input")
    assert len(ids) == 512
    assert all(t == 7 for t in ids)


# ----------------------------------------------------------------------
# Reason1 real-load smoke — only runs when Reason1 weights AND CUDA exist.
# ----------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(
    not (
        os.path.isdir("/path/to/assets/Cosmos-Reason1-7B")
        and torch.cuda.is_available()
    ),
    reason="needs Cosmos-Reason1-7B weights + CUDA",
)
def test_real_reason1_encode_smoke():
    """Run one prompt through real Reason1-7B and assert the geometry."""
    model, tokenizer = ript._build_reason1(
        reason1_ckpt=ript.Path("/path/to/assets/Cosmos-Reason1-7B"),
        device="cuda:0",
        dtype=torch.bfloat16,
    )
    out = ript._encode_reason1(model, tokenizer, "pick up the block", device="cuda:0")
    assert out.shape == (1, 512, 100352)
    assert torch.isfinite(out).all()
