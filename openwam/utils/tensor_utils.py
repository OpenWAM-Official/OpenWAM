import torch


def move_masked_to_left_brute_force(tensor, mask):
    """
    Left-pack sequence elements per batch by moving positions where `mask` is True to the left (preserving their relative order) and padding the remainder with zeros.

    Parameters:
        tensor (torch.Tensor): Batch tensor with shape [B, N, ...]; packing is performed along dimension 1.
        mask (torch.BoolTensor): Boolean mask of shape [B, N] where True indicates elements to move to the left.

    Returns:
        results (torch.Tensor): Packed tensor with shape [B, L, ...], where L is the maximum number of True values in `mask` across the batch; positions beyond each row's valid entries are zeroed.
        new_mask (torch.BoolTensor): Boolean mask with shape [B, L] where True marks valid (packed) positions and False marks padded positions.
    """
    results = []
    new_mask = []
    for i in range(tensor.shape[0]):
        t = torch.cat([tensor[i][mask[i]], torch.zeros_like(tensor[i][~mask[i]])])
        results.append(t)
        # Convert 0-dim tensors to Python ints explicitly; ``list * tensor``
        # is deprecated and raises in recent PyTorch when the factor is a
        # tensor rather than an int.
        n_true = int(mask[i].sum().item())
        n_false = int((~mask[i]).sum().item())
        l = [True] * n_true + [False] * n_false
        new_mask.append(l)
    results = torch.stack(results)
    new_mask = torch.tensor(new_mask, dtype=torch.bool)
    # ``.item()`` so downstream slicing takes a Python int (avoids the GPU
    # sync + dtype edge cases of tensor-scalar slicing).
    max_length = int(mask.sum(dim=1).max().item())

    results = results[:, :max_length]
    new_mask = new_mask[:, :max_length]
    return results, new_mask


def move_masked_to_left_ids(tensor, mask, pad_zero=True):
    """
    Left-pack elements in each row of `tensor` according to `mask`, returning the reordered tensor and a corresponding boolean mask.

    Parameters:
        tensor (torch.Tensor): Source tensor with shape `[B, N]` (or `[B, N, ...]` flattened along dim=1 for indices); rows are reordered so masked (`True`) entries appear first preserving their original relative order.
        mask (torch.BoolTensor): Boolean tensor of shape `[B, N]` indicating which elements to move to the left.
        pad_zero (bool): If `True`, set positions outside the new mask to zero in the returned `result`. If `False`, those positions retain their gathered values.

    Returns:
        result (torch.Tensor): Tensor with masked elements moved to the left in each row and sliced to the maximum number of masked items across the batch; shape `[B, L, ...]` where `L = max(mask.sum(dim=1))`.
        new_mask (torch.BoolTensor): Boolean mask of shape `[B, L]` with `True` for valid (moved) positions and `False` for padded positions.
    """
    masked_index = mask.cumsum(dim=1) - 1
    unmasked_index = (~mask).cumsum(dim=1) - 1
    unmasked_index += mask.sum(dim=1).unsqueeze(1)
    s2t_index = torch.where(mask, masked_index, unmasked_index)
    t2s_index = torch.argsort(s2t_index, dim=1)
    result = torch.gather(tensor, 1, t2s_index)

    length = mask.sum(dim=1)
    result = result[:, : int(length.max().item())]
    new_mask = torch.arange(result.shape[1], device=length.device).unsqueeze(0) < length.unsqueeze(1)
    if pad_zero:
        result[~new_mask] = 0
    return result, new_mask


def move_masked_to_left(tensor, mask, pad_zero=True):
    """
    Left-pack values along dimension 1 so that elements where `mask` is True appear first (preserving their relative order), optionally zero-filling positions beyond each row's valid length.

    Parameters:
        tensor (torch.Tensor): Input tensor of shape [B, N, F] where B is batch, N is sequence length, and F are features.
        mask (torch.BoolTensor): Boolean mask of shape [B, N]; True entries are moved to the left per row.
        pad_zero (bool): If True, set positions outside the per-row valid length to zero in the returned tensor.

    Returns:
        result (torch.Tensor): Tensor of shape [B, L, F] where L is the maximum number of True values across the batch; each row contains its masked values left-packed (remaining positions are zero if `pad_zero` is True).
        new_mask (torch.BoolTensor): Boolean mask of shape [B, L] with True for positions that correspond to moved (valid) entries and False for padded positions.
    """
    masked_index = mask.cumsum(dim=1) - 1
    unmasked_index = (~mask).cumsum(dim=1) - 1
    unmasked_index += mask.sum(dim=1).unsqueeze(1)
    s2t_index = torch.where(mask, masked_index, unmasked_index)
    t2s_index = torch.argsort(s2t_index, dim=1)
    result = torch.gather(tensor, 1, t2s_index.unsqueeze(2).expand(-1, -1, tensor.shape[2]))

    length = mask.sum(dim=1)
    result = result[:, : int(length.max().item())]
    new_mask = torch.arange(result.shape[1], device=length.device).unsqueeze(0) < length.unsqueeze(1)
    if pad_zero:
        result[~new_mask] = 0
    return result, new_mask


def get_mask_of_last_masked_index_brute_force(mask, length):
    """
    Produce a boolean mask marking, per row, the final `length` True entries in the input mask.

    Parameters:
        mask (torch.BoolTensor): 2-D boolean tensor of shape [B, N] where True indicates masked positions.
        length (int or torch.Tensor): Either a single integer applied to all rows or a 1-D tensor of shape [B] providing a per-row count of how many masked positions (counting from the end) to mark.

    Returns:
        torch.BoolTensor: A 2-D boolean tensor of shape [B, N] with `True` at the positions corresponding to the last `length` True values in each row (counting from right to left) and `False` elsewhere.
    """
    results = []
    for i in range(mask.shape[0]):
        len = length if isinstance(length, int) else length[i].item()
        l = [False] * mask.shape[1]
        for j in range(mask.shape[1] - 1, -1, -1):
            if mask[i][j] and len > 0:
                l[j] = True
                len -= 1
            else:
                l[j] = False
        results.append(l)
    return torch.tensor(results, dtype=torch.bool)


def get_mask_of_last_masked_index(mask, length):
    """
    Return a boolean mask marking, for each row, the last `length` true entries in `mask`.

    Parameters:
        mask (torch.BoolTensor): 2-D boolean tensor of shape [B, N] indicating masked positions.
        length (int or torch.Tensor): Integer or 1-D tensor of shape [B] specifying how many masked entries at the end to include per row.

    Returns:
        torch.BoolTensor: Boolean tensor of shape [B, N] where `True` denotes positions corresponding to the last `length` masked `True` values in each row.
    """
    cumsum = mask.cumsum(dim=1)
    new_length = mask.sum(dim=1) - length
    last_masked_index = (cumsum > new_length.unsqueeze(1)) & mask
    return last_masked_index


def test_move_masked_to_left():
    b = 10
    n = 20
    tensor = torch.randn(b, n, 5)
    mask = torch.randint(0, 2, (b, n)).bool()
    result_1, mask_1 = move_masked_to_left(tensor, mask)
    result_2, mask_2 = move_masked_to_left_brute_force(tensor, mask)
    assert (result_1 == result_2).all()
    assert (mask_1 == mask_2).all()
    assert (mask.sum(dim=1) == mask_1.sum(dim=1)).all()

    for i in range(b):
        l = mask[i].sum()
        assert mask_1[i][:l].all()
        assert not mask_1[i][l:].any()


def test_get_mask_of_last_masked_index():
    """
    Validates that the vectorized and brute-force implementations produce identical masks for the last masked indices.

    Generates random boolean masks and per-batch lengths, computes masks with both `get_mask_of_last_masked_index`
    and `get_mask_of_last_masked_index_brute_force`, and asserts the results are equal.
    """
    b = 10
    n = 20
    mask = torch.randint(0, 2, (b, n)).bool()
    length = torch.randint(0, n // 2, (b,))
    last_masked_index_1 = get_mask_of_last_masked_index(mask, length)
    last_masked_index_2 = get_mask_of_last_masked_index_brute_force(mask, length)
    assert (last_masked_index_1 == last_masked_index_2).all()


if __name__ == "__main__":
    test_move_masked_to_left()
    test_get_mask_of_last_masked_index()
