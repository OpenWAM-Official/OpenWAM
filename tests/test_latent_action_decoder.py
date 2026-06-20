import pytest
import torch
import torch.nn.functional as F

from openwam.model.action_backbone.latent_decoder import (
    LatentQueryDecoder,
    build_latent_action_decoder,
)


def _small_decoder(num_query=32, real_action_dim=20, latent_dim=1024):
    return LatentQueryDecoder(
        latent_dim=latent_dim,
        real_action_dim=real_action_dim,
        num_query=num_query,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        attn_head_dim=16,
        ffn_dim=128,
    )


def _small_decoder_proprio(num_query=32, real_action_dim=20, latent_dim=1024, proprio_dim=20):
    return LatentQueryDecoder(
        latent_dim=latent_dim,
        real_action_dim=real_action_dim,
        num_query=num_query,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        attn_head_dim=16,
        ffn_dim=128,
        use_proprioception=True,
        proprio_dim=proprio_dim,
    )


def test_decoder_proprio_shape():
    """Proprio-conditioned decoder: (latent, proprio) -> (B, num_query, real_action_dim)."""
    dec = _small_decoder_proprio(num_query=8, real_action_dim=7, proprio_dim=20)
    latent = torch.randn(3, 6, 1024)
    proprio = torch.randn(3, 1, 20)
    out = dec(latent, proprio)
    assert out.shape == (3, 8, 7)


def test_decoder_proprio_affects_output():
    """Changing proprio must change the decoded action (proprio truly enters KV)."""
    torch.manual_seed(0)
    dec = _small_decoder_proprio(num_query=8, real_action_dim=7, proprio_dim=20).eval()
    latent = torch.randn(2, 6, 1024)
    p_a = torch.randn(2, 1, 20, generator=torch.Generator().manual_seed(11))
    p_b = torch.randn(2, 1, 20, generator=torch.Generator().manual_seed(22))
    with torch.no_grad():
        out_a = dec(latent, p_a)
        out_b = dec(latent, p_b)
    assert not torch.allclose(out_a, out_b, atol=1e-5)


def test_decoder_proprio_required_when_enabled():
    """use_proprioception=True but no proprio -> fail fast."""
    dec = _small_decoder_proprio(proprio_dim=20)
    with pytest.raises(ValueError, match="requires a proprio"):
        dec(torch.randn(2, 6, 1024))


def test_decoder_proprio_dim_required():
    """use_proprioception=True with proprio_dim<=0 -> fail fast at construction."""
    with pytest.raises(ValueError, match="proprio_dim > 0"):
        LatentQueryDecoder(latent_dim=1024, real_action_dim=20, num_query=8, use_proprioception=True, proprio_dim=0)


def test_decoder_proprio_shape_normalization():
    """Decoder accepts (D,) and (B, D) proprio, broadcasting to (B, 1, D)."""
    dec = _small_decoder_proprio(num_query=8, real_action_dim=7, proprio_dim=20).eval()
    latent = torch.randn(2, 6, 1024)
    with torch.no_grad():
        out_1d = dec(latent, torch.randn(20))  # (D,)
        out_2d = dec(latent, torch.randn(2, 20))  # (B, D)
    assert out_1d.shape == (2, 8, 7)
    assert out_2d.shape == (2, 8, 7)


def test_build_decoder_proprio_from_cfg():
    """build_latent_action_decoder wires use_proprioception + proprio_dim."""
    cfg = {
        "name": "cross_attn_query",
        "num_query": 8,
        "real_action_dim": 7,
        "hidden_dim": 64,
        "num_layers": 2,
        "num_heads": 4,
        "attn_head_dim": 16,
        "ffn_dim": 128,
        "use_proprioception": True,
        "proprio_dim": 20,
    }
    dec = build_latent_action_decoder(cfg, latent_dim=1024)
    assert dec.use_proprioception is True
    assert dec.proprio_proj is not None
    out = dec(torch.randn(2, 6, 1024), torch.randn(2, 1, 20))
    assert out.shape == (2, 8, 7)


def test_decoder_shape():
    dec = _small_decoder(num_query=32, real_action_dim=20)
    out = dec(torch.randn(2, 128, 1024))
    assert out.shape == (2, 32, 20)


@pytest.mark.parametrize("num_query,real_action_dim,t_lat", [(32, 20, 128), (16, 14, 64), (8, 7, 16)])
def test_decoder_shape_parametrized(num_query, real_action_dim, t_lat):
    dec = _small_decoder(num_query=num_query, real_action_dim=real_action_dim)
    out = dec(torch.randn(3, t_lat, 1024))
    assert out.shape == (3, num_query, real_action_dim)


def test_build_from_cfg():
    cfg = {
        "name": "cross_attn_query",
        "hidden_dim": 64,
        "num_layers": 2,
        "num_heads": 4,
        "attn_head_dim": 16,
        "ffn_dim": 128,
        "num_query": 32,
        "real_action_dim": 20,
    }
    dec = build_latent_action_decoder(cfg, latent_dim=1024)
    assert dec(torch.randn(1, 128, 1024)).shape == (1, 32, 20)


def test_build_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unsupported latent action decoder"):
        build_latent_action_decoder({"name": "mlp", "num_query": 32, "real_action_dim": 20}, latent_dim=1024)


def test_build_requires_dims():
    with pytest.raises(ValueError, match="real_action_dim and num_query"):
        build_latent_action_decoder({"name": "cross_attn_query"}, latent_dim=1024)


def test_decoder_gradient_flows_to_input():
    """End-to-end: decoder backward must reach its input latent (the reconstructed
    x0 carries ActionDiT's gradient), and update decoder params."""
    dec = _small_decoder()
    latent = torch.randn(2, 128, 1024, requires_grad=True)  # stands in for reconstructed x0
    target = torch.randn(2, 32, 20)
    loss = F.mse_loss(dec(latent), target)
    loss.backward()
    assert latent.grad is not None and latent.grad.abs().sum() > 0  # gradient flows back to ActionDiT
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in dec.parameters())


def test_latent_reconstruction_formula():
    """x0 = noisy - sigma*velocity inverts add_noise/training_target exactly."""
    from openwam.model.action_backbone.scheduler import ActionScheduler

    sched = ActionScheduler()
    x0 = torch.randn(2, 32, 20)
    noise = torch.randn(2, 32, 20)
    sigma = torch.rand(2, 1, 1)
    noisy = sched.add_noise(x0, noise, sigma)
    velocity = sched.training_target(x0, noise)
    x0_rec = noisy - sigma * velocity
    assert torch.allclose(x0_rec, x0, atol=1e-5)


def test_masked_mse_full_pad_is_zero():
    from openwam.model.architectures.architecture_base import BaseWAMArchitecture

    pred = torch.randn(2, 32, 20, requires_grad=True)
    target = torch.randn(2, 32, 20)
    is_pad = torch.ones(2, 32, 20, dtype=torch.bool)  # fully padded
    loss = BaseWAMArchitecture._masked_mse(pred, target, is_pad)
    assert loss.item() == 0.0
    loss.backward()
    assert pred.grad is not None and pred.grad.abs().sum().item() == 0.0  # no gradient when all-pad


def test_masked_mse_shapes():
    from openwam.model.architectures.architecture_base import BaseWAMArchitecture

    pred = torch.zeros(1, 4, 3)
    target = torch.ones(1, 4, 3)
    # no mask -> plain mean (=1.0 since (0-1)^2)
    assert BaseWAMArchitecture._masked_mse(pred, target, None).item() == pytest.approx(1.0)
    # (B, T) mask broadcast across action dim
    is_pad_bt = torch.tensor([[False, False, True, True]])
    assert BaseWAMArchitecture._masked_mse(pred, target, is_pad_bt).item() == pytest.approx(1.0)
    # (B, T, D) per-cell mask
    is_pad_btd = torch.ones(1, 4, 3, dtype=torch.bool)
    is_pad_btd[0, 0, 0] = False
    assert BaseWAMArchitecture._masked_mse(pred, target, is_pad_btd).item() == pytest.approx(1.0)


def _build_action_dit(action_type, action_dim, with_decoder=True):
    from openwam.model.action_backbone.action_dit import ActionDiT

    dec_cfg = {
        "name": "cross_attn_query",
        "hidden_dim": 64,
        "num_layers": 2,
        "num_heads": 4,
        "attn_head_dim": 16,
        "ffn_dim": 128,
        "num_query": 32,
        "real_action_dim": 20,
    }
    return ActionDiT(
        action_dim=action_dim,
        dim=64,
        ffn_dim=128,
        num_heads=4,
        num_layers=1,
        video_dim=64,
        bridge_layers=(0,),
        variant="joint_cross_attn",
        attn_head_dim=16,
        action_type=action_type,
        latent_decoder=dec_cfg if with_decoder else None,
    )


def test_actiondit_owns_decoder_in_latent_mode():
    """Decoder ownership lives in ActionDiT; architecture is decoder-agnostic."""
    # latent mode + decoder cfg -> ActionDiT builds it, action_dim=1024 = latent_dim
    latent = _build_action_dit("latent", action_dim=1024, with_decoder=True)
    assert latent.has_latent_decoder is True
    out = latent.decode_latent_to_action(torch.randn(2, 128, 1024))
    assert out.shape == (2, 32, 20)

    # explicit mode -> no decoder even if cfg present
    explicit = _build_action_dit("explicit", action_dim=20, with_decoder=True)
    assert explicit.has_latent_decoder is False
    assert explicit.decode_latent_to_action(torch.randn(2, 128, 1024)) is None

    # latent mode without decoder cfg -> no decoder
    nodec = _build_action_dit("latent", action_dim=1024, with_decoder=False)
    assert nodec.has_latent_decoder is False


def test_actiondit_decoder_on_device_via_set_dtype():
    """set_dtype_device (self.to) moves the decoder submodule — no device mismatch.

    This is the regression guard for the bug remote-verify caught when the decoder
    was an architecture-level module the action backbone's set_dtype_device missed."""
    ad = _build_action_dit("latent", action_dim=1024, with_decoder=True)
    ad.set_dtype_device(torch.float32, torch.device("cpu"))
    dev = next(ad.latent_action_decoder.parameters()).device
    assert dev.type == "cpu"  # decoder followed set_dtype_device


# --- Architecture-level end-to-end (CPU mock backbone) ---
# Exercises the full latent encoder->ActionDiT->decoder->loss path through the
# real dual_system compute_loss, mirroring the remote GPU verify but with a mock
# video backbone so structure + gradient flow are covered without Wan weights.

_LATENT_DIM = 32  # ActionDiT hidden = mock video_dim; doubles as latent token_dim here
_REAL_ACTION_DIM = 7
_NUM_QUERY = 5
_T_LATENT = 6  # latent sequence length (e.g. pairs*tokens_per_pair)


def _make_latent_dual(lambda_decoder_cfg=True):
    """Build a dual_system (cross_attn) in latent mode with a latent decoder,
    on a mock video backbone — no Wan weights."""
    from openwam.model.architectures.dual_system.joint_cross_attn import DualSystemCrossAttnArchitecture
    from tests.test_openwam_trainer import _MockVideoBackbone

    dec_cfg = {
        "name": "cross_attn_query",
        "hidden_dim": 32,
        "num_layers": 2,
        "num_heads": 4,
        "attn_head_dim": 8,
        "ffn_dim": 64,
        "num_query": _NUM_QUERY,
        "real_action_dim": _REAL_ACTION_DIM,
    }
    cfg = {
        "framework": "dual_system",
        "variant": "joint_cross_attn",
        "detach_bridge": False,
        "action_dim": _LATENT_DIM,  # latent mode: action_dim == latent token_dim
        "dim": _LATENT_DIM,
        "ffn_dim": 64,
        "num_heads": 4,
        "video_dim": _LATENT_DIM,
        "text_dim": 16,
        "bridge_layers": (0, 1),
        "type": "latent",
        "latent_decoder": dec_cfg if lambda_decoder_cfg else None,
    }
    arch = DualSystemCrossAttnArchitecture(cfg=cfg)
    arch.video_backbone = _MockVideoBackbone(dim=_LATENT_DIM, num_layers=2, num_heads=4)
    arch._device = torch.device("cpu")
    arch._dtype = torch.float32
    return arch


def _latent_loss_inputs(B=1, full_mask=False):
    from tests.test_openwam_trainer import _make_fake_loss_inputs

    inputs = _make_fake_loss_inputs(B=B, action_dim=_LATENT_DIM, video_dim=_LATENT_DIM)
    # ActionDiT target = latent (B, T_latent, latent_dim); decoder target = real action.
    inputs["actions"] = torch.randn(B, _T_LATENT, _LATENT_DIM)
    inputs["decoder_target"] = torch.randn(B, _NUM_QUERY, _REAL_ACTION_DIM)
    is_pad = torch.ones if full_mask else torch.zeros
    inputs["decoder_action_is_pad"] = is_pad(B, _NUM_QUERY, _REAL_ACTION_DIM, dtype=torch.bool)
    return inputs


def test_arch_latent_decoder_end_to_end_gradient():
    """Full compute_loss: decoder loss present and backprops into ActionDiT."""
    from tests.test_openwam_trainer import _MockScheduler

    arch = _make_latent_dual()
    arch.action_backbone.scheduler = _MockScheduler()
    assert arch.action_backbone.has_latent_decoder is True

    inputs = _latent_loss_inputs(B=1)
    result = arch.compute_loss(**inputs, lambda_video=1.0, lambda_action=1.0, lambda_decoder=1.0, current_step=0)

    assert "loss_decoder" in result and result["loss_decoder"].item() > 0
    result["loss"].backward()
    g_action = sum(p.grad.abs().sum().item() for p in arch.action_backbone.parameters() if p.grad is not None)
    assert g_action > 0  # decoder MSE reached ActionDiT (end-to-end)


def test_arch_latent_decoder_full_mask_zero():
    """Full action mask -> decoder loss 0, no decoder gradient."""
    from tests.test_openwam_trainer import _MockScheduler

    arch = _make_latent_dual()
    arch.action_backbone.scheduler = _MockScheduler()

    inputs = _latent_loss_inputs(B=1, full_mask=True)
    result = arch.compute_loss(**inputs, lambda_video=0.0, lambda_action=1.0, lambda_decoder=1.0, current_step=0)

    ld = result.get("loss_decoder")
    assert (ld.item() if ld is not None else 0.0) == 0.0
    result["loss"].backward()
    g_dec = sum(
        p.grad.abs().sum().item() for p in arch.action_backbone.latent_action_decoder.parameters() if p.grad is not None
    )
    assert g_dec == 0.0


def test_arch_latent_decoder_fail_fast_without_action():
    """lambda_decoder>0 requires lambda_action>0 (reconstruction needs velocity)."""
    from tests.test_openwam_trainer import _MockScheduler

    arch = _make_latent_dual()
    arch.action_backbone.scheduler = _MockScheduler()
    inputs = _latent_loss_inputs(B=1)
    with pytest.raises(ValueError, match="requires lambda_action>0"):
        arch.compute_loss(**inputs, lambda_video=1.0, lambda_action=0.0, lambda_decoder=1.0, current_step=0)


def test_arch_explicit_no_decoder():
    """Explicit dual_system has no decoder; decoder branch is inert."""
    from openwam.model.architectures.dual_system.joint_cross_attn import DualSystemCrossAttnArchitecture
    from tests.test_openwam_trainer import _make_fake_loss_inputs, _MockScheduler, _MockVideoBackbone

    cfg = {
        "framework": "dual_system",
        "variant": "joint_cross_attn",
        "detach_bridge": False,
        "action_dim": _REAL_ACTION_DIM,
        "dim": 32,
        "ffn_dim": 64,
        "num_heads": 4,
        "video_dim": 32,
        "text_dim": 16,
        "bridge_layers": (0, 1),
        # no type/latent_decoder -> explicit
    }
    arch = DualSystemCrossAttnArchitecture(cfg=cfg)
    arch.video_backbone = _MockVideoBackbone(dim=32, num_layers=2, num_heads=4)
    arch._device, arch._dtype = torch.device("cpu"), torch.float32
    arch.action_backbone.scheduler = _MockScheduler()
    assert arch.action_backbone.has_latent_decoder is False

    inputs = _make_fake_loss_inputs(B=1, action_dim=_REAL_ACTION_DIM)
    result = arch.compute_loss(
        **inputs,
        actions=torch.randn(1, 5, _REAL_ACTION_DIM),
        lambda_video=1.0,
        lambda_action=1.0,
        lambda_decoder=1.0,
        current_step=0,
    )
    assert "loss_decoder" not in result  # decoder branch inert in explicit mode
