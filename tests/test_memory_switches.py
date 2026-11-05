"""umT5 weight offload preserves encoder outputs and releases device weights."""

from types import SimpleNamespace

import pytest
import torch

from openwam.model.video_backbone.wan import encode as wan_encode


class _Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(64, 32)
        self.proj = torch.nn.Linear(32, 32)
        self.register_buffer("bias", torch.randn(32))

    def forward(self, ids, mask):
        return self.proj(self.embed(ids)) * mask.unsqueeze(-1) + self.bias


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_text_encoder_offload_gives_the_same_rows():
    torch.manual_seed(0)
    encoder = _Encoder().to("cuda", torch.bfloat16).eval().requires_grad_(False)
    ids = torch.randint(0, 64, (3, 16), device="cuda")
    mask = (torch.rand(3, 16, device="cuda") > 0.2).long()
    reference = encoder(ids, mask)

    cache = SimpleNamespace(offload=wan_encode.TextEncoderOffload(encoder))
    for _ in range(3):
        out = wan_encode._run_encoder(encoder, ids, mask, device="cuda", cache=cache)
        assert torch.equal(out, reference)
        tensors = list(encoder.parameters()) + list(encoder.buffers())
        assert all(t.device.type == "cpu" and t.is_pinned() for t in tensors)
    assert cache.offload.loads == 2
