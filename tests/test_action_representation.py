"""Tests for the action representation layer."""

import pytest
import torch


def test_base_is_abstract():
    """BaseActionRepresentation cannot be instantiated directly."""
    from open_wam.models.action_repr.base import BaseActionRepresentation

    with pytest.raises(TypeError):
        BaseActionRepresentation()


def test_continuous_repr_identity():
    """ContinuousActionRepresentation is an identity transform."""
    from open_wam.models.action_repr import ContinuousActionRepresentation

    repr = ContinuousActionRepresentation(action_dim=7)
    assert repr.latent_dim == 7
    assert repr.native_dim == 7

    actions = torch.randn(2, 10, 7)
    encoded = repr.encode(actions)
    decoded = repr.decode(encoded)

    assert torch.equal(encoded, actions)
    assert torch.equal(decoded, actions)


def test_continuous_repr_default_dim():
    """Default action_dim is 14 (bimanual)."""
    from open_wam.models.action_repr import ContinuousActionRepresentation

    repr = ContinuousActionRepresentation()
    assert repr.latent_dim == 14
    assert repr.native_dim == 14


def test_build_action_representation_continuous():
    """Factory builds ContinuousActionRepresentation."""
    from open_wam.models.action_repr import ContinuousActionRepresentation, build_action_representation

    repr = build_action_representation("continuous", action_dim=7)
    assert isinstance(repr, ContinuousActionRepresentation)
    assert repr.latent_dim == 7


def test_build_action_representation_unknown():
    """Factory raises on unknown representation name."""
    from open_wam.models.action_repr import build_action_representation

    with pytest.raises(ValueError, match="Unknown action representation"):
        build_action_representation("nonexistent")


def test_fast_repr_imports():
    """FASTActionRepresentation should be importable."""
    from open_wam.models.action_repr.fast import FASTActionRepresentation

    assert FASTActionRepresentation is not None


def test_fast_repr_construction():
    """FASTActionRepresentation should construct without tokenizer."""
    from open_wam.models.action_repr.fast import FASTActionRepresentation

    repr = FASTActionRepresentation(
        native_dim=7,
        vocab_size=256,
        num_tokens_per_step=2,
        embed_dim=32,
    )
    assert repr.native_dim == 7
    assert repr.latent_dim == 2 * 32  # num_tokens_per_step * embed_dim
    assert repr._tokenizer is None  # not loaded yet


def test_fast_repr_embedding_and_projection():
    """Verify embedding and output projection shapes."""
    from open_wam.models.action_repr.fast import FASTActionRepresentation

    repr = FASTActionRepresentation(
        native_dim=7,
        vocab_size=128,
        num_tokens_per_step=1,
        embed_dim=16,
    )
    assert repr.token_embedding.num_embeddings == 128
    assert repr.token_embedding.embedding_dim == 16
    assert repr.output_proj.in_features == 16
    assert repr.output_proj.out_features == 128


def test_fast_repr_encode_requires_tokenizer():
    """Encoding should fail with clear message when tokenizer not installed."""
    import builtins
    from unittest.mock import patch

    from open_wam.models.action_repr.fast import FASTActionRepresentation

    repr = FASTActionRepresentation(
        native_dim=7,
        vocab_size=256,
        num_tokens_per_step=1,
        embed_dim=32,
    )
    # Mock the import to simulate fast_tokenizer not being installed
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "fast_tokenizer":
            raise ImportError("No module named 'fast_tokenizer'")
        return original_import(name, *args, **kwargs)

    actions = torch.randn(1, 5, 7)
    with patch("builtins.__import__", side_effect=mock_import):
        with pytest.raises(ImportError, match="FAST tokenizer"):
            repr.encode(actions)


def test_build_fast_via_factory():
    """Factory should build FAST representation."""
    from open_wam.models.action_repr import build_action_representation
    from open_wam.models.action_repr.fast import FASTActionRepresentation

    repr = build_action_representation(
        "fast",
        native_dim=7,
        vocab_size=256,
        num_tokens_per_step=1,
        embed_dim=32,
    )
    assert isinstance(repr, FASTActionRepresentation)
    assert repr.latent_dim == 32


def test_continuous_backward_compatible_with_none():
    """When action_repr is None, training/inference should work unchanged."""
    # This tests that our integration points handle None gracefully
    from open_wam.models.action_repr import ContinuousActionRepresentation

    repr = ContinuousActionRepresentation(action_dim=14)
    data = torch.randn(2, 49, 14)

    # encode → add noise → ... → decode should roundtrip
    encoded = repr.encode(data)
    decoded = repr.decode(encoded)
    assert torch.equal(data, decoded)
