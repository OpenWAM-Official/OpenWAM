import numpy as np
import pytest

from openwam.dataloader.utils.rot6d import (
    convert_wuji_58,
    row_rot6d_to_col_rot6d,
    wuji_58_rot6d_to_matrix,
)
from openwam.dataloader.utils.unify_action import map_to_unify, parse_unify_spec, unmap_from_unify


def _rotations(count: int = 16) -> np.ndarray:
    rng = np.random.default_rng(42)
    matrices, _ = np.linalg.qr(rng.normal(size=(count, 3, 3)))
    matrices[np.linalg.det(matrices) < 0, :, 2] *= -1
    return matrices.astype(np.float32)


def test_row_rot6d_to_col_rot6d_recovers_columns():
    matrices = _rotations()
    rows = np.concatenate((matrices[:, 0, :], matrices[:, 1, :]), axis=-1)
    expected = np.concatenate((matrices[:, :, 0], matrices[:, :, 1]), axis=-1)
    np.testing.assert_allclose(row_rot6d_to_col_rot6d(rows), expected, atol=2e-7)


def test_convert_wuji_58_reorders_blocks_and_recovers_rotations():
    matrices = _rotations(2)
    rows = np.concatenate((matrices[:, 0, :], matrices[:, 1, :]), axis=-1)
    raw = np.zeros((2, 58), dtype=np.float32)
    raw[:, 0:3] = 1
    raw[:, 3:9] = rows
    raw[:, 9:12] = 2
    raw[:, 12:18] = rows
    raw[:, 18:38] = 3
    raw[:, 38:58] = 4

    converted = convert_wuji_58(raw)
    np.testing.assert_array_equal(converted[:, 9:29], 3)
    np.testing.assert_array_equal(converted[:, 38:58], 4)
    recovered = wuji_58_rot6d_to_matrix(converted)
    np.testing.assert_allclose(recovered[:, 0], matrices, atol=3e-7)
    np.testing.assert_allclose(recovered[:, 1], matrices, atol=3e-7)


def test_wuji_canonical_mapping_round_trip():
    raw = np.arange(58, dtype=np.float32)[None]
    dst = parse_unify_spec(["0-8", "10-29", "34-42", "44-63"])
    unified, mask = map_to_unify(raw, dst)
    assert unified.shape == (1, 80)
    assert mask.sum() == 58
    np.testing.assert_array_equal(unmap_from_unify(unified, dst), raw)


@pytest.mark.parametrize("shape", [(5,), (7,), (2, 57)])
def test_rot6d_helpers_reject_wrong_width(shape):
    values = np.zeros(shape, dtype=np.float32)
    fn = row_rot6d_to_col_rot6d if shape[-1] != 57 else convert_wuji_58
    with pytest.raises(ValueError):
        fn(values)
