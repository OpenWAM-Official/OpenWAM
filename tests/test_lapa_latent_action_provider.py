import os
from pathlib import Path

import pytest
import torch

from openwam.model.action_backbone.latent_encoder.lapa_dinov3 import _validate_lapa_paths, is_lfs_pointer_file


def test_lfs_pointer_detection(tmp_path: Path):
    pointer = tmp_path / "laq_dinov3.pt"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 3244897688\n",
        encoding="utf-8",
    )
    real = tmp_path / "real.pt"
    real.write_bytes(b"PK\x03\x04not-a-pointer")

    assert is_lfs_pointer_file(pointer)
    assert not is_lfs_pointer_file(real)
    assert not is_lfs_pointer_file(tmp_path / "missing.pt")


def test_lapa_path_validation_rejects_wrong_dinov3_hidden_size(tmp_path: Path):
    lapa_dir = tmp_path / "lapa"
    lapa_dir.mkdir()
    (lapa_dir / "laq_dinov3.pt").write_bytes(b"PK\x03\x04real-ish")

    dinov3_dir = tmp_path / "dinov3-vitb16"
    dinov3_dir.mkdir()
    (dinov3_dir / "config.json").write_text('{"hidden_size": 768}', encoding="utf-8")

    with pytest.raises(ValueError, match="hidden_size=1024"):
        _validate_lapa_paths(
            {"lapa_model_dir": str(lapa_dir), "dinov3_model_dir": str(dinov3_dir)},
            expected_dim=1024,
        )


def test_lapa_path_validation_rejects_dinov3_lfs_pointer(tmp_path: Path):
    lapa_dir = tmp_path / "lapa"
    lapa_dir.mkdir()
    (lapa_dir / "laq_dinov3.pt").write_bytes(b"PK\x03\x04real-ish")

    dinov3_dir = tmp_path / "dinov3-vitl16"
    dinov3_dir.mkdir()
    (dinov3_dir / "config.json").write_text('{"hidden_size": 1024}', encoding="utf-8")
    (dinov3_dir / "model.safetensors").write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="DINOv3 weight file is still a Git LFS pointer"):
        _validate_lapa_paths(
            {"lapa_model_dir": str(lapa_dir), "dinov3_model_dir": str(dinov3_dir)},
            expected_dim=1024,
        )


def test_lapa_provider_builds_flattened_pair_targets_without_real_model(monkeypatch):
    import openwam.model.action_backbone.latent_encoder.lapa_dinov3 as lapa_mod

    class _FakeModel(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs
            self.return_only_codebook_ids = None

        def load_state_dict(self, state, strict=True):  # noqa: ARG002
            return None

        def forward(self, video, return_only_codebook_ids=False):
            self.return_only_codebook_ids = return_only_codebook_ids
            n = video.shape[0]
            tokens = torch.arange(n * 16 * 1024, dtype=torch.float32, device=video.device).reshape(n, 16, 1024)
            ids = torch.zeros(n, 16, dtype=torch.long, device=video.device)
            return tokens, ids

    monkeypatch.setattr(lapa_mod, "_validate_lapa_paths", lambda _paths, expected_dim=None: (Path("."), Path(".")))
    monkeypatch.setattr(lapa_mod, "LatentActionQuantizationDinov3Feature", _FakeModel)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {"model": {}})

    cfg = {
        "name": "lapa_dinov3",
        "paths": {},
        "input": {"image_size": 224},
        "model": {"code_seq_len": 16},
        "output": {"tokens_per_pair": 16, "token_dim": 1024, "flatten_pairs": True, "action_dim": 1024},
    }
    provider = lapa_mod.LAPADinov3TargetProvider(cfg, device=torch.device("cpu"), dtype=torch.float32)
    videos = [torch.rand(3, 384, 320, 3), torch.rand(3, 384, 320, 3)]

    out = provider(videos)

    assert out.shape == (2, 32, 1024)
    assert torch.equal(out[0], torch.arange(32 * 1024, dtype=torch.float32).reshape(32, 1024))
    assert provider.model.return_only_codebook_ids is True


def test_lapa_provider_rejects_short_video(monkeypatch):
    import openwam.model.action_backbone.latent_encoder.lapa_dinov3 as lapa_mod

    class _BareProvider(lapa_mod.LAPADinov3TargetProvider):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.device = torch.device("cpu")
            self.image_size = 224

    provider = _BareProvider()

    with pytest.raises(ValueError, match="at least 2"):
        provider._build_pairs([torch.rand(1, 384, 320, 3)])


def test_lapa_provider_rejects_ambiguous_channel_first_t3_video():
    import openwam.model.action_backbone.latent_encoder.lapa_dinov3 as lapa_mod

    with pytest.raises(ValueError, match="Ambiguous video tensor layout"):
        lapa_mod._video_to_tensor(torch.rand(3, 3, 8, 8))

    class _BareProvider(lapa_mod.LAPADinov3TargetProvider):
        def __init__(self):
            torch.nn.Module.__init__(self)

    provider = _BareProvider()
    with pytest.raises(ValueError, match="Ambiguous batch video tensor layout"):
        provider._build_pairs(torch.rand(1, 3, 3, 8, 8))


def test_lapa_provider_rejects_action_dim_token_dim_mismatch(monkeypatch):
    import openwam.model.action_backbone.latent_encoder.lapa_dinov3 as lapa_mod

    class _FakeModel(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

    monkeypatch.setattr(lapa_mod, "_validate_lapa_paths", lambda _paths, expected_dim=None: (Path("."), Path(".")))
    monkeypatch.setattr(lapa_mod, "LatentActionQuantizationDinov3Feature", _FakeModel)

    cfg = {
        "name": "lapa_dinov3",
        "paths": {},
        "input": {"image_size": 224},
        "model": {"code_seq_len": 16},
        "output": {"tokens_per_pair": 16, "token_dim": 512, "flatten_pairs": True, "action_dim": 1024},
    }
    with pytest.raises(ValueError, match="must equal"):
        lapa_mod.LAPADinov3TargetProvider(cfg, device=torch.device("cpu"), dtype=torch.float32)


def test_lapa_load_state_dict_rejects_unmatched_keys(monkeypatch):
    from openwam.model.action_backbone.latent_encoder import lapa_dinov3_model as model_mod

    class _TinyTokenizer(torch.nn.Module):
        def forward(self, x):
            return type("Out", (), {"last_hidden_state": torch.zeros(x.shape[0], 201, 1024)})()

    monkeypatch.setattr(model_mod, "load_dinov3_tokenizer", lambda *args, **kwargs: _TinyTokenizer())
    model = model_mod.LatentActionQuantizationDinov3Feature(
        dim=1024,
        quant_dim=32,
        codebook_size=8,
        image_size=224,
        patch_size=16,
        spatial_depth=1,
        temporal_depth=1,
        dim_head=64,
        heads=16,
        code_seq_len=16,
        dinov3_model_dir=".",
        device=torch.device("cpu"),
    )

    with pytest.raises(RuntimeError, match="did not match any model parameters"):
        model.load_state_dict({"not_a_real_weight": torch.zeros(1)})


@pytest.mark.skipif(
    not bool(os.environ.get("OPENWAM_RUN_LAPA_REAL_CKPT_TEST")), reason="opt-in test loads a large LAPA checkpoint"
)
def test_lapa_real_checkpoint_loads_with_expected_key_coverage():
    from unittest.mock import patch

    from openwam.model.action_backbone.latent_encoder.lapa_dinov3_model import LatentActionQuantizationDinov3Feature

    class _TinyTokenizer(torch.nn.Module):
        def forward(self, x):  # noqa: D401
            return type("Out", (), {"last_hidden_state": torch.zeros(x.shape[0], 201, 1024)})()

    with patch(
        "openwam.model.action_backbone.latent_encoder.lapa_dinov3_model.load_dinov3_tokenizer",
        return_value=_TinyTokenizer(),
    ):
        model = LatentActionQuantizationDinov3Feature(
            dim=1024,
            quant_dim=32,
            codebook_size=8,
            image_size=224,
            patch_size=16,
            spatial_depth=4,
            temporal_depth=4,
            dim_head=64,
            heads=16,
            code_seq_len=16,
            dinov3_model_dir="models/dinov3-vitl16-pretrain-lvd1689m",
            device=torch.device("cpu"),
        )
        state = torch.load("models/LAPA-DINOv3/LAPA-DINOv3/laq_dinov3.pt", map_location="cpu", weights_only=False)
        incompatible = model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
        assert all(k.startswith("dino_tokenizer.") for k in incompatible.missing_keys)
        assert not incompatible.unexpected_keys


@pytest.mark.skipif(
    not bool(os.environ.get("OPENWAM_RUN_LARYBENCH_ALIGNMENT_TEST")),
    reason="opt-in test loads the reference LARYBench stack and large local weights",
)
def test_lapa_matches_larybench_reference_for_fixed_pair(monkeypatch):
    import sys

    from openwam.model.action_backbone.latent_encoder.lapa_dinov3_model import LatentActionQuantizationDinov3Feature

    repo = Path.cwd()
    ref_root = repo / "ref_code" / "LARYBench"
    monkeypatch.setenv("DINO_V3_PATH", str(repo / "models" / "dinov3-vitl16-pretrain-lvd1689m"))
    sys.path.insert(0, str(ref_root))
    try:
        from get_latent_action.models.laq_model.latent_action_quantization_dinov3_feature import (
            LatentActionQuantizationDinov3Feature as RefModel,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        kwargs = dict(
            dim=1024,
            quant_dim=32,
            codebook_size=8,
            image_size=224,
            patch_size=16,
            spatial_depth=4,
            temporal_depth=4,
            dim_head=64,
            heads=16,
            code_seq_len=16,
        )
        ours = LatentActionQuantizationDinov3Feature(
            **kwargs,
            dinov3_model_dir=repo / "models" / "dinov3-vitl16-pretrain-lvd1689m",
            device=device,
        ).to(device)
        ref = RefModel(**kwargs).to(device)
        state = torch.load(
            repo / "models" / "LAPA-DINOv3" / "LAPA-DINOv3" / "laq_dinov3.pt", map_location="cpu", weights_only=False
        )
        state = state["model"] if isinstance(state, dict) and "model" in state else state
        ours.load_state_dict(state)
        ref.load_state_dict(state)
        ours.eval()
        ref.eval()

        torch.manual_seed(7)
        video = torch.rand(2, 3, 2, 224, 224, device=device)
        with torch.inference_mode():
            ours_tokens, ours_ids = ours(video, return_only_codebook_ids=True)
            ref_tokens, ref_ids = ref(video, return_only_codebook_ids=True)
        assert torch.equal(ours_ids.cpu(), ref_ids.cpu())
        assert torch.allclose(ours_tokens.cpu(), ref_tokens.cpu(), atol=1e-5, rtol=1e-5)
    finally:
        try:
            sys.path.remove(str(ref_root))
        except ValueError:
            pass
