"""Tests for the unified-action mapping helpers in ``openwam.dataloader.utils.unify_action``.

Pins the byte-level behavior of the schema-agnostic scatter/gather layer:
spec parsing (both forms), map correctness + mask, map/unmap round-trip, and
the validation errors. Pure numpy — runs without torch.
"""

from __future__ import annotations

import numpy as np
import pytest

from openwam.dataloader.utils.unify_action import (
    map_to_unify,
    parse_unify_spec,
    unmap_from_unify,
)

# Example from the module docstring / plan:
#   raw 0..8 -> unified 0..8 ; raw 9 -> 31 ; raw 10..17 -> 33..40
EXPECTED_DST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 31, 33, 34, 35, 36, 37, 38, 39, 40]
UNIFY_DIM = 80


class TestParseSpec:
    def test_single_list_form(self):
        dst = parse_unify_spec(["0-8", 31, "33-40"], unify_dim=UNIFY_DIM)
        assert dst.tolist() == EXPECTED_DST

    def test_paired_list_form_matches_single(self):
        spec = [["0-8", "0-8"], [9, 31], ["10-17", "33-40"]]
        dst = parse_unify_spec(spec, unify_dim=UNIFY_DIM)
        assert dst.tolist() == EXPECTED_DST

    def test_paired_form_handles_non_contiguous_src_order(self):
        # src given out of order — result is indexed by src dim, so it sorts.
        spec = [[2, 50], [0, 10], [1, 20]]
        dst = parse_unify_spec(spec, unify_dim=UNIFY_DIM)
        assert dst.tolist() == [10, 20, 50]  # src 0->10, src 1->20, src 2->50

    def test_single_int_and_bare_range(self):
        assert parse_unify_spec([5], unify_dim=UNIFY_DIM).tolist() == [5]
        assert parse_unify_spec(["0-2"], unify_dim=UNIFY_DIM).tolist() == [0, 1, 2]


class TestParseSpecErrors:
    def test_dst_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            parse_unify_spec(["0-2", 80], unify_dim=UNIFY_DIM)  # 80 >= unify_dim

    def test_duplicate_destination(self):
        with pytest.raises(ValueError, match="more than one source"):
            parse_unify_spec([5, 5], unify_dim=UNIFY_DIM)

    def test_paired_length_mismatch(self):
        with pytest.raises(ValueError, match="length mismatch"):
            parse_unify_spec([["0-2", "0-1"]], unify_dim=UNIFY_DIM)  # 3 src vs 2 dst

    def test_paired_src_has_gap(self):
        with pytest.raises(ValueError, match="no gaps"):
            parse_unify_spec([[0, 10], [2, 12]], unify_dim=UNIFY_DIM)  # src 1 missing

    def test_empty_spec(self):
        with pytest.raises(ValueError, match="non-empty"):
            parse_unify_spec([], unify_dim=UNIFY_DIM)

    def test_malformed_range(self):
        with pytest.raises(ValueError, match="end < start"):
            parse_unify_spec(["5-2"], unify_dim=UNIFY_DIM)


class TestMapToUnify:
    def test_shape_and_scatter(self):
        dst = parse_unify_spec(["0-8", 31, "33-40"], unify_dim=UNIFY_DIM)
        T, N = 7, len(dst)
        action = np.arange(T * N, dtype=np.float32).reshape(T, N)
        unified, mask = map_to_unify(action, dst, unify_dim=UNIFY_DIM)

        assert unified.shape == (T, UNIFY_DIM)
        assert unified.dtype == np.float32
        # mapped slots carry the raw values
        np.testing.assert_array_equal(unified[:, dst], action)
        # non-mapped slots are zero
        other = np.setdiff1d(np.arange(UNIFY_DIM), dst)
        np.testing.assert_array_equal(unified[:, other], 0.0)

    def test_dim_mask(self):
        dst = parse_unify_spec(["0-8", 31, "33-40"], unify_dim=UNIFY_DIM)
        _, mask = map_to_unify(np.zeros((3, len(dst)), dtype=np.float32), dst, unify_dim=UNIFY_DIM)
        assert mask.shape == (UNIFY_DIM,)
        assert mask.dtype == bool
        assert mask.sum() == len(dst)
        np.testing.assert_array_equal(np.nonzero(mask)[0], np.sort(dst))

    def test_wrong_action_width_raises(self):
        dst = parse_unify_spec(["0-2"], unify_dim=UNIFY_DIM)  # N = 3
        with pytest.raises(ValueError, match="!= mapping size"):
            map_to_unify(np.zeros((4, 5), dtype=np.float32), dst, unify_dim=UNIFY_DIM)


class TestRoundTrip:
    @pytest.mark.parametrize("shape_prefix", [(), (10,), (4, 32)])
    def test_unmap_inverts_map(self, shape_prefix):
        dst = parse_unify_spec([["0-8", "0-8"], [9, 31], ["10-17", "33-40"]], unify_dim=UNIFY_DIM)
        n = len(dst)
        rng = np.random.default_rng(0)
        action = rng.standard_normal((*shape_prefix, n)).astype(np.float32)
        unified, _ = map_to_unify(action, dst, unify_dim=UNIFY_DIM)
        recovered = unmap_from_unify(unified, dst)
        assert recovered.shape == action.shape
        np.testing.assert_array_equal(recovered, action)

    def test_unmap_only_reads_mapped_slots(self):
        # Garbage in non-mapped slots must not leak into the recovered action.
        dst = parse_unify_spec(["0-2", 50], unify_dim=UNIFY_DIM)
        unified = np.full((2, UNIFY_DIM), 999.0, dtype=np.float32)
        known = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
        unified[:, dst] = known
        recovered = unmap_from_unify(unified, dst)
        np.testing.assert_array_equal(recovered, known)
