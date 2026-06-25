import numpy as np

from openwam.dataloader.ebench import EBENCH80_DIM_MASK, _raw19_to_ebench80


def test_raw19_to_ebench80_mapping():
    raw = np.arange(19, dtype=np.float32)
    mapped = _raw19_to_ebench80(raw)

    assert mapped.shape == (80,)
    np.testing.assert_allclose(mapped[10:16], raw[0:6])
    np.testing.assert_allclose(mapped[42:48], raw[6:12])
    np.testing.assert_allclose(mapped[16:18], raw[12:14])
    np.testing.assert_allclose(mapped[48:50], raw[14:16])
    assert mapped[9] == np.mean(raw[12:14])
    assert mapped[41] == np.mean(raw[14:16])
    np.testing.assert_allclose(mapped[64:67], raw[16:19])

    assert int(EBENCH80_DIM_MASK.sum()) == 21
    assert np.all(mapped[~EBENCH80_DIM_MASK] == 0.0)
