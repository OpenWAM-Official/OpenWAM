"""Unit tests for the S-VAE feature reducer core module."""

import pytest
import torch

from openwam.model.video_backbone.encoder.svae import (
    SVAE,
    DiagonalGaussian,
    build_svae,
    load_svae,
    svae_loss,
)


def _mk(**kw) -> SVAE:
    cfg = dict(input_dim=16, latent_dim=4, num_heads=2, num_layers=1, intermediate_size=16, dropout=0.0)
    cfg.update(kw)
    return SVAE(**cfg)


def _inp(B=2, C=16, T=3, H=2, W=2) -> torch.Tensor:
    return torch.randn(B, C, T, H, W)


def test_encode_eval_shape():
    mu = _mk().eval().encode(_inp())
    assert mu.shape == (2, 4, 3, 2, 2)


def test_encode_train_returns_tuple():
    z, mu, logvar = _mk().train().encode(_inp())
    assert z.shape == mu.shape == logvar.shape == (2, 4, 3, 2, 2)


def test_decode_shape():
    rec = _mk().eval().decode(torch.randn(2, 4, 3, 2, 2))
    assert rec.shape == (2, 16, 3, 2, 2)


def test_forward_keys_and_shapes():
    out = _mk().train()(_inp())
    assert {"recon", "target", "mu", "logvar"} <= set(out)
    assert out["recon"].shape == out["target"].shape == (2, 16, 3, 2, 2)
    assert out["mu"].shape == out["logvar"].shape == (2, 4, 3, 2, 2)


def test_loss_is_finite_and_nonneg():
    out = _mk().train()(_inp())
    loss, stats = svae_loss(out, beta=1e-4, free_bits_per_dim=0.05)
    assert torch.isfinite(loss) and loss.item() >= 0.0
    for k in ("mse", "cos", "kl", "loss"):
        assert torch.isfinite(stats[k])


def test_free_bits_floor():
    # With a collapsed posterior (mu≈0, logvar≈0), per-dim KL≈0; free bits
    # raise the reported KL to the floor.
    out = _mk().train()(_inp())
    out = {**out, "mu": torch.zeros_like(out["mu"]), "logvar": torch.zeros_like(out["logvar"])}
    _, stats = svae_loss(out, beta=1.0, free_bits_per_dim=0.05)
    assert stats["kl"].item() == pytest.approx(0.05, abs=1e-6)


def test_eval_is_deterministic():
    m = _mk().eval()
    z = _inp()
    assert torch.allclose(m.encode(z), m.encode(z))


def test_train_is_stochastic():
    torch.manual_seed(0)
    m = _mk().train()
    z = _inp()
    assert not torch.allclose(m.encode(z)[0], m.encode(z)[0])


def test_set_input_stats():
    m = _mk().eval()
    mean = torch.arange(16).float()
    std = torch.full((16,), 2.0)
    m.set_input_stats(mean, std)
    assert torch.allclose(m.input_mean, mean)
    assert torch.allclose(m.input_std, std)


def test_set_input_stats_clamps_zero_std():
    m = _mk().eval()
    m.set_input_stats(torch.zeros(16), torch.zeros(16))
    assert (m.input_std > 0).all()


def test_standardisation_stats_stay_fp32_across_dtype_cast():
    # The deploy host casts the encoder to bf16; the standardisation
    # buffers must stay fp32 (they are dataset stats, not compute tensors) so
    # the standardisation matches standalone fp32-buffer training.
    m = _mk().eval()
    m.set_input_stats(torch.arange(16).float(), torch.full((16,), 2.0))
    m.to(torch.bfloat16)
    assert m.input_mean.dtype == torch.float32
    assert m.input_std.dtype == torch.float32
    assert m.enc_proj.weight.dtype == torch.bfloat16  # params did cast


def test_input_dim_head_divisibility():
    with pytest.raises(ValueError):
        SVAE(input_dim=15, latent_dim=4, num_heads=2)


def test_diagonal_gaussian_eval_returns_mu():
    g = DiagonalGaussian().eval()
    z, mu, _ = g(torch.randn(5, 8))
    assert torch.allclose(z, mu)


def test_config_dict_and_checkpoint_roundtrip(tmp_path):
    m = _mk()
    m.set_input_stats(torch.arange(16).float(), torch.full((16,), 2.0))  # non-trivial buffers
    cfg = m.config_dict()
    assert build_svae(cfg).config_dict() == cfg

    p = tmp_path / "svae.pt"
    torch.save({"format_version": 2, "model_config": cfg, "state_dict": m.state_dict()}, p)
    m2 = load_svae(str(p))

    # persistent standardisation buffers must survive the round-trip
    assert torch.allclose(m.input_mean, m2.input_mean)
    assert torch.allclose(m.input_std, m2.input_std)

    m.eval()
    m2.eval()
    z = _inp()
    assert torch.allclose(m.encode(z), m2.encode(z))


def test_load_svae_rejects_bad_checkpoint(tmp_path):
    p = tmp_path / "bad.pt"
    torch.save({"foo": 1}, p)
    with pytest.raises(ValueError):
        load_svae(str(p))


def test_load_svae_rejects_unsupported_format_version(tmp_path):
    # A checkpoint carrying valid model_config/state_dict but a mismatched (or
    # missing) format_version must fail fast in load_svae, not deeper inside
    # load_state_dict.
    m = _mk()
    for bad in ({"format_version": 999}, {}):  # wrong version, and missing entirely
        p = tmp_path / f"v_{bad.get('format_version', 'none')}.pt"
        torch.save({**bad, "model_config": m.config_dict(), "state_dict": m.state_dict()}, p)
        with pytest.raises(ValueError, match="format_version"):
            load_svae(str(p))


def test_encode_mean_deterministic_even_in_train_mode():
    # The host model.train() can recursively put the frozen reducer in train
    # mode; encode_mean must still be deterministic and equal to eval-mode mu.
    m = _mk().train()
    z = _inp()
    a = m.encode_mean(z)
    b = m.encode_mean(z)
    assert torch.allclose(a, b)
    m.eval()
    assert torch.allclose(a, m.encode(z))


def test_encode_mean_deterministic_with_dropout_in_train_mode():
    # encode_mean must bypass Transformer-block dropout too (not only Gaussian
    # sampling): with a non-zero dropout config and the module left in train
    # mode, two calls must be identical, and the prior (train) mode is restored.
    m = SVAE(input_dim=16, latent_dim=4, num_layers=1, num_heads=2, intermediate_size=16, dropout=0.5)
    m.train()
    z = _inp()
    a = m.encode_mean(z)
    b = m.encode_mean(z)
    assert torch.allclose(a, b)
    assert m.training  # train mode restored after the temporary eval()


def test_forward_recon_target_in_standardized_space():
    m = _mk().eval()
    mean = torch.arange(16).float()
    std = torch.full((16,), 3.0)
    m.set_input_stats(mean, std)
    z = _inp()
    out = m(z)
    expected = (z - mean.view(1, 16, 1, 1, 1)) / std.view(1, 16, 1, 1, 1)
    assert torch.allclose(out["target"], expected, atol=1e-5)
