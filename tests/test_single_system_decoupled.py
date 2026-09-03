"""Verify SingleSystem architectures own their forward end-to-end.

After the decoupling refactor, the architecture's forward drives the video
DiT block loop directly and calls action-backbone helpers (encode / decode
/ apply_expert) — the action backbone never invokes ``vb.run_block``. This
file exercises that contract end-to-end with a stub video backbone so we
don't need real Wan checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from openwam.model.architectures.single_system.moe import SingleSystemMoEArchitecture
from openwam.model.architectures.single_system.vanilla import SingleSystemVanillaArchitecture


@dataclass
class _StubBlockLoopState:
    """Minimal shape mirroring ``BlockLoopState``; only the fields the architecture
    forwards actually read from / write to.

    ``t_mod`` is a 4D dummy so the SingleSystem fail-fast on per-token t_mod mode
    (architecture forward enforces ``vstate.time_mod.dim() == 4``) is satisfied without
    plumbing real per-token AdaLN values through the stub."""

    hidden_states: torch.Tensor
    time_mod: torch.Tensor = None  # type: ignore[assignment]
    grid_height: int = 1
    grid_width: int = 1
    extras: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.time_mod is None:
            self.time_mod = torch.zeros(1, 1, 6, 1)


class _StubVideoBackbone(nn.Module):
    """Minimal video backbone whose state.hidden_states records the action-token slot.

    Tracks how many times each block ran and which block_ids saw action
    tokens, so tests can assert the architecture iterated through the
    full loop and inject/extract round-tripped correctly.
    """

    def __init__(self, dim: int, num_layers: int, video_seq: int):
        super().__init__()
        self.dim = dim
        self._num_layers = num_layers
        self._video_seq = video_seq
        # The block list is real torch.nn.Linears so grad flows through if needed.
        self.blocks = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])
        self.run_block_calls: list[int] = []
        self.seen_shared_attention_masks: list[torch.Tensor | None] = []
        self.injected_action_tokens = 0
        self.injected_state_tokens = 0

    @property
    def num_layers(self) -> int:
        return self._num_layers

    def prepare(self, **_kw):
        B = 1
        return _StubBlockLoopState(
            hidden_states=torch.zeros(B, self._video_seq, self.dim),
            grid_height=1,
            grid_width=self._video_seq,
            extras={},
        )

    def run_block(self, block_id, state):
        self.run_block_calls.append(block_id)
        self.seen_shared_attention_masks.append(state.extras.get("shared_attention_mask"))
        state.hidden_states = self.blocks[block_id](state.hidden_states)
        return state

    def build_video_to_video_mask(self, *, video_seq_len, video_tokens_per_frame, device):  # noqa: ARG002
        return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)

    def assert_ready_for_shared_tokens(self, state):  # noqa: ARG002
        return None

    def finalize(self, state):
        # Caller has already extracted action tokens; just return a fixed-shape
        # video noise prediction whose value depends on state.hidden_states so grads can
        # propagate during backward tests.
        return state.hidden_states.sum(dim=-1, keepdim=True)

    def inject_action_tokens(self, state, action_tokens, n_action, *, timestep=None):  # noqa: ARG002
        self.injected_action_tokens = n_action
        state.hidden_states = torch.cat([state.hidden_states, action_tokens], dim=1)
        return state

    def inject_shared_tokens(
        self,
        state,
        action_tokens,
        n_action,
        *,
        state_tokens=None,
        n_state=0,
        timestep=None,
    ):  # noqa: ARG002
        self.injected_action_tokens = n_action
        self.injected_state_tokens = n_state
        pieces = [action_tokens]
        if n_state:
            pieces.append(state_tokens)
        state.hidden_states = torch.cat([state.hidden_states, *pieces], dim=1)
        return state

    def extract_action_tokens(self, state, n_action):
        action_tail = state.hidden_states[:, -n_action:, :]
        state.hidden_states = state.hidden_states[:, :-n_action, :]
        return state, action_tail

    def extract_shared_tokens(self, state, n_action, *, n_state=0):
        n_tail = n_action + n_state
        n_video = state.hidden_states.shape[1] - n_tail
        action_tail = state.hidden_states[:, n_video : n_video + n_action, :]
        state.hidden_states = state.hidden_states[:, :n_video, :]
        return state, action_tail


def _make_vanilla(video_dim=64, num_layers=4):
    arch = SingleSystemVanillaArchitecture(
        cfg={
            "framework": "single_system",
            "variant": "vanilla",
            "action_dim": 7,
            "video_dim": video_dim,
            "max_action_len": 32,
        }
    )
    arch.video_backbone = _StubVideoBackbone(dim=video_dim, num_layers=num_layers, video_seq=20)
    return arch


def _make_moe(video_dim=64, num_layers=4, bridge_layers=(0, 2)):
    arch = SingleSystemMoEArchitecture(
        cfg={
            "framework": "single_system",
            "variant": "moe",
            "action_dim": 7,
            "video_dim": video_dim,
            "expert_ffn_dim": 128,
            "bridge_layers": list(bridge_layers),
        }
    )
    arch.video_backbone = _StubVideoBackbone(dim=video_dim, num_layers=num_layers, video_seq=20)
    return arch


def test_vanilla_forward_runs_full_block_loop_with_action_tokens():
    arch = _make_vanilla(video_dim=64, num_layers=4)
    arch.eval()
    vb = arch.video_backbone

    B, T = 1, 5
    actions = torch.randn(B, T, 7)
    timestep = torch.tensor([100.0])

    with torch.no_grad():
        video_pred, action_pred = arch.forward(actions, timestep)

    # Architecture must iterate through every video DiT block.
    assert vb.run_block_calls == list(range(vb.num_layers))
    # Action tokens were injected via vb.inject_action_tokens, not via the
    # action backbone calling vb.run_block itself.
    assert vb.injected_action_tokens == T
    assert action_pred.shape == (B, T, 7)
    assert video_pred is not None


def test_vanilla_forward_passes_joint_mask_to_every_video_block():
    arch = _make_vanilla(video_dim=64, num_layers=4)
    arch.eval()
    vb = arch.video_backbone

    B, T = 1, 5
    actions = torch.randn(B, T, 7)
    timestep = torch.tensor([100.0])

    with torch.no_grad():
        arch.forward(actions, timestep)

    assert len(vb.seen_shared_attention_masks) == vb.num_layers
    assert all(mask is not None for mask in vb.seen_shared_attention_masks)
    first_mask = vb.seen_shared_attention_masks[0]
    assert all(mask is first_mask for mask in vb.seen_shared_attention_masks)
    assert first_mask.shape == (vb._video_seq + T, vb._video_seq + T)
    assert first_mask.dtype == torch.bool
    assert not first_mask[: vb._video_seq, vb._video_seq :].any()
    assert first_mask[vb._video_seq :, : vb._video_seq].all()
    assert first_mask[vb._video_seq :, vb._video_seq :].all()


def test_vanilla_forward_video_only_when_actions_none():
    arch = _make_vanilla(video_dim=64, num_layers=3)
    arch.eval()
    vb = arch.video_backbone

    with torch.no_grad():
        video_pred, action_pred = arch.forward(None, None)

    assert action_pred is None
    assert vb.run_block_calls == list(range(vb.num_layers))
    # No action-token injection in video-only mode.
    assert vb.injected_action_tokens == 0
    assert video_pred is not None


def test_moe_forward_calls_apply_expert_only_at_expert_layers():
    arch = _make_moe(video_dim=64, num_layers=5, bridge_layers=(0, 2, 4))
    arch.eval()
    vb = arch.video_backbone
    ab = arch.action_backbone

    # Spy on apply_expert so the test can prove it ran exactly at expert layers.
    apply_expert_calls = []
    original = ab.apply_expert

    def _spy(layer_id, x_action, t_mod):
        apply_expert_calls.append(layer_id)
        return original(layer_id, x_action, t_mod)

    ab.apply_expert = _spy

    B, T = 1, 6
    actions = torch.randn(B, T, 7)
    timestep = torch.tensor([100.0])

    with torch.no_grad():
        _, action_pred = arch.forward(actions, timestep)

    assert vb.run_block_calls == list(range(vb.num_layers))
    assert apply_expert_calls == [0, 2, 4]
    assert action_pred.shape == (B, T, 7)


def test_moe_forward_passes_joint_mask_to_every_video_block():
    arch = _make_moe(video_dim=64, num_layers=5, bridge_layers=(0, 2, 4))
    arch.eval()
    vb = arch.video_backbone

    B, T = 1, 6
    actions = torch.randn(B, T, 7)
    timestep = torch.tensor([100.0])

    with torch.no_grad():
        arch.forward(actions, timestep)

    assert len(vb.seen_shared_attention_masks) == vb.num_layers
    assert all(mask is not None for mask in vb.seen_shared_attention_masks)
    first_mask = vb.seen_shared_attention_masks[0]
    assert all(mask is first_mask for mask in vb.seen_shared_attention_masks)
    assert first_mask.shape == (vb._video_seq + T, vb._video_seq + T)
    assert first_mask.dtype == torch.bool
    assert not first_mask[: vb._video_seq, vb._video_seq :].any()
    assert first_mask[vb._video_seq :, : vb._video_seq].all()
    assert first_mask[vb._video_seq :, vb._video_seq :].all()


def test_action_backbone_does_not_implement_run_block():
    """Crucial: SingleSystem's action backbone must NOT carry a run_block method
    that would let it drive the video loop. The architecture owns control flow."""
    vanilla = _make_vanilla()
    moe = _make_moe()

    # nn.Module objects don't accidentally inherit a run_block from anywhere
    # we control after Phase 1's ABC slim. Only video backbones do.
    def _own_attrs(obj):
        return set(vars(type(obj)))

    assert "run_block" not in _own_attrs(vanilla.action_backbone)
    assert "run_block" not in _own_attrs(moe.action_backbone)


def test_action_backbone_grads_flow_through_forward():
    """End-to-end backward pass should populate grads on action backbone params."""
    arch = _make_vanilla(video_dim=64, num_layers=2)
    arch.train()

    actions = torch.randn(1, 4, 7, requires_grad=False)
    timestep = torch.tensor([100.0])
    _, action_pred = arch.forward(actions, timestep)
    action_pred.sum().backward()

    # At least one action_backbone parameter must have received a non-zero grad.
    grads = [p.grad for p in arch.action_backbone.parameters() if p.requires_grad and p.grad is not None]
    assert grads, "Expected at least one action_backbone parameter to have a gradient."
    assert any(g.abs().sum() > 0 for g in grads), "Action backbone gradients should not all be zero."
