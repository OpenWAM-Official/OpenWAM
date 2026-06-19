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
    from openwam.model.action_backbone.joint_action_dit import ActionDiT

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
