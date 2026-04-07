"""Tests for multi-embodiment action space abstraction."""

import numpy as np
import pytest

from open_wam.data.embodiment import (
    CANONICAL_BIMANUAL_DIM,
    CANONICAL_SINGLE_ARM_DIM,
    EMBODIMENTS,
    ActionSpaceAdapter,
    EmbodimentConfig,
    get_embodiment,
    register_embodiment,
)


def test_canonical_dims():
    assert CANONICAL_SINGLE_ARM_DIM == 7
    assert CANONICAL_BIMANUAL_DIM == 14


def test_standard_embodiments_registered():
    """All standard embodiments should be in the registry."""
    for name in ["franka", "widowx", "google_robot", "arx-x5", "aloha"]:
        config = get_embodiment(name)
        assert config.name == name


def test_unknown_embodiment_raises():
    with pytest.raises(KeyError, match="Unknown embodiment"):
        get_embodiment("nonexistent_robot")


def test_franka_config():
    cfg = get_embodiment("franka")
    assert cfg.native_action_dim == 7
    assert cfg.bimanual is False
    assert cfg.canonical_dim == 7


def test_arx_x5_config():
    cfg = get_embodiment("arx-x5")
    assert cfg.native_action_dim == 14
    assert cfg.bimanual is True
    assert cfg.canonical_dim == 14


def test_single_arm_native_to_canonical():
    """7-DoF single arm should map directly."""
    adapter = ActionSpaceAdapter("franka")
    native = np.array([[1, 2, 3, 4, 5, 6, 7]], dtype=np.float32)
    canonical = adapter.native_to_canonical(native)

    assert canonical.shape == (1, 7)
    np.testing.assert_allclose(canonical[0], [1, 2, 3, 4, 5, 6, 7])


def test_single_arm_to_bimanual_canonical():
    """7-DoF arm should pad to 14 when target_dim=14."""
    adapter = ActionSpaceAdapter("franka", target_dim=CANONICAL_BIMANUAL_DIM)
    native = np.array([[1, 2, 3, 4, 5, 6, 7]], dtype=np.float32)
    canonical = adapter.native_to_canonical(native)

    assert canonical.shape == (1, 14)
    np.testing.assert_allclose(canonical[0, :7], [1, 2, 3, 4, 5, 6, 7])
    np.testing.assert_allclose(canonical[0, 7:], 0.0)  # Right arm zeroed


def test_bimanual_native_to_canonical():
    """14-DoF bimanual should map both arms."""
    adapter = ActionSpaceAdapter("arx-x5")
    native = np.arange(1, 15, dtype=np.float32).reshape(1, 14)
    canonical = adapter.native_to_canonical(native)

    assert canonical.shape == (1, 14)
    np.testing.assert_allclose(canonical[0], np.arange(1, 15))


def test_roundtrip_single_arm():
    """native -> canonical -> native should be identity."""
    adapter = ActionSpaceAdapter("franka")
    native = np.array([[0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 1.0]], dtype=np.float32)
    canonical = adapter.native_to_canonical(native)
    recovered = adapter.canonical_to_native(canonical)

    np.testing.assert_allclose(recovered, native, atol=1e-6)


def test_roundtrip_bimanual():
    """Bimanual roundtrip should preserve all 14 dims."""
    adapter = ActionSpaceAdapter("arx-x5")
    native = np.random.randn(5, 14).astype(np.float32)
    canonical = adapter.native_to_canonical(native)
    recovered = adapter.canonical_to_native(canonical)

    np.testing.assert_allclose(recovered, native, atol=1e-6)


def test_batch_conversion():
    """Should handle batch dimensions."""
    adapter = ActionSpaceAdapter("widowx")
    native = np.random.randn(32, 10, 7).astype(np.float32)
    canonical = adapter.native_to_canonical(native)

    assert canonical.shape == (32, 10, 7)


def test_action_scale():
    """Action scale should be applied during conversion."""
    config = EmbodimentConfig(
        name="scaled",
        native_action_dim=3,
        action_scale=np.array([2.0, 0.5, 1.0]),
        native_to_canonical={0: 0, 1: 1, 2: 2},
    )
    adapter = ActionSpaceAdapter(config)
    native = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    canonical = adapter.native_to_canonical(native)

    np.testing.assert_allclose(canonical[0, :3], [2.0, 1.0, 3.0])

    # Roundtrip should undo scaling
    recovered = adapter.canonical_to_native(canonical)
    np.testing.assert_allclose(recovered, native, atol=1e-6)


def test_custom_mapping():
    """Custom native_to_canonical mapping."""
    config = EmbodimentConfig(
        name="custom",
        native_action_dim=4,
        native_to_canonical={0: 2, 1: 0, 2: 1, 3: 6},  # Reorder + gripper
    )
    adapter = ActionSpaceAdapter(config)
    native = np.array([[10, 20, 30, 40]], dtype=np.float32)
    canonical = adapter.native_to_canonical(native)

    assert canonical[0, 2] == 10  # native[0] -> canonical[2]
    assert canonical[0, 0] == 20  # native[1] -> canonical[0]
    assert canonical[0, 1] == 30  # native[2] -> canonical[1]
    assert canonical[0, 6] == 40  # native[3] -> canonical[6] (gripper)


def test_register_new_embodiment():
    """Should be able to register custom embodiments."""
    cfg = EmbodimentConfig(name="my_robot", native_action_dim=5)
    register_embodiment("my_robot", cfg)
    assert get_embodiment("my_robot") is cfg
    # Cleanup
    del EMBODIMENTS["my_robot"]


def test_adapter_properties():
    adapter = ActionSpaceAdapter("franka", target_dim=14)
    assert adapter.canonical_dim == 14
    assert adapter.native_dim == 7


def test_import_from_package():
    """Embodiment module should be importable from open_wam.data."""
    from open_wam.data import ActionSpaceAdapter

    assert callable(ActionSpaceAdapter)
