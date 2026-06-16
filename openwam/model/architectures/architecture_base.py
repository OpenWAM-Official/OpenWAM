"""Abstract base class for WAM (World-Action Model) architectures.

Supported architecture families:

1. **Shared Backbone** (`framework=shared_backbone`)
   Action tokens are concatenated to the video DiT sequence and ride
   through the shared blocks. Variants: `vanilla` (no extra capacity) /
   `moe` (expert FFN at selected layers).

2. **Dual-System** (`framework=dual_system`)
   A separate ActionDiT consumes features from the video DiT. Variants:
   `joint_cross_attn` (bridge cross-attention after a full video forward)
   / `joint_self_attn` (MMDiT-style mixed attention at every layer, driven
   by :class:`MoTJointDriver`).

3. **Tri-System** (`framework=tri_system`)
   Motus-style mixture of transformers: Wan video DiT + action expert +
   frozen VLM / understanding expert, with mixed attention implemented via
   the video backbone adapter.

Each architecture composes the backbones it owns and implements its own
``forward()``.
"""

import functools
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from openwam.model.compile_options import compile_mode


def _wrap_single_forward(module: nn.Module) -> None:
    """Wrap a single module's ``forward`` in ``torch.no_grad``. Idempotent."""
    if getattr(module, "_openwam_no_grad_wrapped", False):
        return
    original_forward = module.forward

    @functools.wraps(original_forward)
    def wrapped(*args, **kwargs):
        with torch.no_grad():
            return original_forward(*args, **kwargs)

    module.forward = wrapped
    module._openwam_no_grad_wrapped = True


def _wrap_forward_in_no_grad(module: nn.Module) -> None:
    """Wrap ``forward`` of ``module`` AND every submodule in its subtree in ``torch.no_grad``.

    Recursion matters because callers commonly bypass the root forward and call a
    nested submodule directly. The canonical case in OpenWAM is
    ``Qwen3VLBackbone.extract_features`` which calls ``self.vlm_model.model(...)``
    (the inner ``Qwen3VLModel``, skipping the LM head) — wrapping only
    ``vlm_model.forward`` would leave that path grad-tracking. Recursively wrapping
    every descendant makes the semantic complete: any entry point into the frozen
    subtree is in ``no_grad``.

    Idempotent — a marker attribute on each module prevents double-wrapping if
    ``freeze_modules`` runs more than once. ``nn.Module.modules()`` deduplicates
    via its internal memo, so cyclic registrations (e.g. test fakes with
    ``self.model = self``) are visited once.

    Safe to apply to any module: if the caller already wraps the call in a
    ``no_grad`` context (e.g. ``prepare_inputs``), the inner ``no_grad`` is a
    no-op; if the caller is inside a grad-tracking forward (the tri_system VLM
    case this actually saves memory in), it short-circuits activation saving.

    **Subtree-level semantic, not per-parameter**: a trainable child under a frozen
    parent will NOT receive gradients, because every descendant ``forward`` is
    wrapped in ``no_grad``. For partial-freeze setups (e.g. LoRA on a frozen base,
    or training only the LM head of an otherwise frozen VLM), do NOT pass the
    parent's dotted path to ``freeze_modules``; pass the specific leaves you want
    frozen instead. The current freeze list in ``configs/model/*.yaml``
    only names complete subtrees, so this limitation does not bite today.
    """
    for sub in module.modules():
        _wrap_single_forward(sub)


logger = logging.getLogger(__name__)

# Prefix for VLM backbone parameters in the architecture state_dict.
# VLM weights are saved as a separate checkpoint directory (not in safetensors)
# to avoid tied-weight deduplication complexity.
VLM_STATE_DICT_PREFIX = "vlm_backbone."


def _exclude_vlm_from_state_dict(state_dict: dict[str, "Tensor"]) -> dict[str, "Tensor"]:
    """Filter out VLM backbone parameters from a state dict.

    Note: this exclusion is prefix-based (``vlm_backbone.*``).  Future
    trainable modules on the VLM (e.g. LoRA adapters) must be registered at
    the architecture top level (as siblings of ``vlm_backbone``), NOT as
    children under ``vlm_backbone``, otherwise they will be silently excluded
    from the checkpoint.
    """
    return {k: v for k, v in state_dict.items() if not k.startswith(VLM_STATE_DICT_PREFIX)}


def _assert_decode_video_supported(vb) -> None:
    """Fail-fast guard for ``generate(decode_video=True)`` against backbones
    wired to an irreversible external encoder (DINOv3 / V-JEPA2).

    Silently returning ``video=None`` would mask a config mismatch (the
    caller asked for pixels but the encoder cannot produce them). Pulled
    out of :meth:`BaseWAMArchitecture.generate` so it is independently
    unit-testable without standing up the full denoising loop.
    """
    enc = getattr(vb, "video_encoder", None)
    if enc is not None and not enc.spec.is_reversible:
        raise ValueError(
            f"generate(decode_video=True) but the configured encoder "
            f"({type(enc).__name__}) is irreversible (spec.is_reversible=False). "
            "Pass decode_video=False to retrieve raw latents."
        )


if TYPE_CHECKING:
    from openwam.model.action_backbone.actionbackbone_base import ActionBackbone
    from openwam.model.video_backbone.videobackbone_base import VideoBackbone


@dataclass
class ActionState:
    """Mutable state container used by the joint self-attention path.

    Only ``DualSystemSelfAttnArchitecture`` needs this — its action stream is
    threaded through ``MoTJointDriver``, which mutates the payload across
    layers. SharedBackbone and DualSystem cross-attn don't go through this
    container.

    Fields:
        action_latents: (B, T_action, action_dim) noisy actions (input to forward).
        timestep: action diffusion timestep (raw shape preserved for the action
            backbone's internal use).
        payload: backbone-specific per-forward state (typically
            ``ActionDiTState``).
    """

    action_latents: Optional[Tensor] = None
    timestep: Optional[Tensor] = None
    payload: Optional[Any] = None


class BaseWAMArchitecture(ABC, nn.Module):
    """Base class for WAM architecture variants.

    Composes a ``video_backbone`` and an ``action_backbone`` plus optional
    extra backbones. Subclasses instantiate the appropriate ActionBackbone
    subclass in ``__init__`` and own the complete ``forward()`` control flow.

    Args:
        cfg: Architecture-specific configuration (OmegaConf DictConfig or dict).
    """

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.video_backbone: Optional["VideoBackbone"] = None
        self.action_backbone: Optional["ActionBackbone"] = None
        self._device = torch.device("cuda")
        self._dtype = torch.bfloat16

        # Forward-time training runtime flags. Trainer calls
        # ``set_training_runtime`` once during construction so ``prepare_inputs``
        # can read these without the trainer having to thread them through.
        self._use_gradient_checkpointing = False
        self._use_gradient_checkpointing_offload = False
        self._max_timestep_boundary = 1.0
        self._min_timestep_boundary = 0.0

        # Optional action normalizer for deployment. ``generate`` uses it to
        # return real-scale actions; deploy-side proprio preprocessing uses it
        # to normalize raw robot state into the model's training space.
        self.normalizer = None

        if cfg is not None:
            self._init_video_backbone(cfg)

    @staticmethod
    def _cfg_get(cfg, key, default=None):
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def _init_video_backbone(self, cfg):
        """Build video backbone from config.

        Supports two source types in ``cfg.video_backbone``:
        - ``_source``: direct model directory or dict with components → deploy-time path
        - ``name``: registry key → training-time path

        Both paths flow through the public :func:`build_video_backbone`.

        Optional ``video_backbone.encoder`` block (yaml-whitelisted to
        ``{name, model_path}``) swaps the backbone's native VAE for an
        external :class:`VideoEncoder`. The encoder block is **only** read
        when ``video_backbone.from_scratch=true`` — the DiT must be
        reinitialized when its latent space changes. When the block is set
        but ``from_scratch=false`` we silently route through the native
        ``pipe.vae`` (with an INFO log explaining what happened) so that the
        default yaml's documentation-friendly ``encoder:`` block doesn't
        break the default training command.
        """
        from openwam.model.video_backbone import build_video_backbone

        vb_cfg = cfg.get("video_backbone", {}) if isinstance(cfg, dict) else getattr(cfg, "video_backbone", None)
        if vb_cfg is None:
            return

        source = vb_cfg.get("_source") if isinstance(vb_cfg, dict) else getattr(vb_cfg, "_source", None)
        vb_name = vb_cfg.get("name") if isinstance(vb_cfg, dict) else getattr(vb_cfg, "name", None)
        from_scratch = bool(self._cfg_get(vb_cfg, "from_scratch", False))

        # ------------------------------------------------------------------
        # External encoder gate. Four cases, only one of which builds an
        # encoder:
        #   - encoder set + from_scratch=true  → build external encoder
        #   - encoder set + from_scratch=false → INFO log + skip (silent
        #     ignore is the right UX since the default yaml ships an
        #     encoder: block for documentation discoverability, and we
        #     don't want the default training command to fail)
        #   - encoder unset + from_scratch=true → reset DiT weights only
        #   - encoder unset + from_scratch=false → no-op (default path)
        # ------------------------------------------------------------------
        enc_cfg = vb_cfg.get("encoder") if isinstance(vb_cfg, dict) else getattr(vb_cfg, "encoder", None)
        external_encoder = None
        # Single gate, identical for training and deploy: encoder block is
        # honored ONLY when ``from_scratch=true``. The framework yamls ship
        # an inline ``encoder:`` block for discoverability even at default
        # ``from_scratch=false`` (see commit 6588044) — that block must be
        # silently ignored on both paths so default training and deploy of
        # ``from_scratch=false`` checkpoints (state_dict topology
        # ``_pipe.vae.*``) keep working bit-exactly.
        if enc_cfg is not None and from_scratch:
            # Yaml whitelist applies to both paths so an in-tree typo /
            # extra field never silently slips through deploy. The
            # always-allowed pair is {name, model_path}; each encoder
            # class extends the set via ``optional_yaml_keys()`` for
            # runtime knobs (e.g. V-JEPA 2.1's ``vjepa2_1_forward``).
            from openwam.model.video_backbone.encoder import _VIDEO_ENCODER_REGISTRY

            allowed = BaseWAMArchitecture._compute_encoder_yaml_whitelist(enc_cfg, _VIDEO_ENCODER_REGISTRY)
            extras = BaseWAMArchitecture._encoder_yaml_extras(enc_cfg, allowed)
            if extras:
                raise ValueError(
                    f"video_backbone.encoder allows only {sorted(allowed)} in yaml; got extra "
                    f"fields {sorted(extras)}. Spec fields like z_dim are derived from the "
                    "loaded weights — yaml cannot override them."
                )

            if source is None:
                # Training: build the encoder from yaml + model_path.
                from openwam.model.video_backbone.encoder import build_video_encoder

                external_encoder = build_video_encoder(enc_cfg)
            else:
                # Deploy: reconstruct the encoder skeleton from the saved
                # components entry; weights filled in by the architecture's
                # subsequent ``load_checkpoint`` strict load. ``source`` is
                # the dict produced by deploy/model_loader.py. ``_ckpt_dir``
                # is plumbed onto ``vb_cfg`` by model_loader and forwarded
                # to :meth:`VideoEncoder.from_skeleton` so each encoder can
                # consult the checkpoint-local artifacts that its
                # :meth:`VideoEncoder.copy_deploy_artifacts` wrote at save
                # time. For example, V-JEPA 2.1 prefers
                # ``<ckpt_dir>/manifest.json`` with a fallback to
                # ``encoder.model_path``. The user-side weight directory does
                # not need to be reachable on the deploy host.
                ckpt_dir_for_encoder = (
                    vb_cfg.get("_ckpt_dir") if isinstance(vb_cfg, dict) else getattr(vb_cfg, "_ckpt_dir", None)
                )
                external_encoder = self._build_external_encoder_skeleton(enc_cfg, source, ckpt_dir=ckpt_dir_for_encoder)
        elif enc_cfg is not None and source is None:
            # Training with encoder block set but from_scratch=false. Two
            # sub-cases:
            #   (a) ``encoder.name == "wan_vae"`` (the default-yaml template
            #       value) — stay silent (INFO only). Native ``pipe.vae``
            #       and the wan_vae external encoder are bit-identical, so
            #       nothing is lost; the default yaml ships the
            #       ``encoder:`` block as a discoverable hint and
            #       fail-fast here would break every default config.
            #   (b) ``encoder.name`` is anything else (``vjepa2_1`` /
            #       ``dinov3`` / ...) — that is an explicit
            #       choice that *cannot* take effect under
            #       ``from_scratch=false``: the pre-trained DiT's first
            #       conv channels are bound to the native Wan VAE's
            #       ``z_dim`` and there is no way to wire a different
            #       encoder's latent space through without re-initializing
            #       the DiT. Silently INFO-logging would produce a Wan-VAE
            #       run that *looks* like a V-JEPA run from the yaml, so
            #       fail-fast.
            enc_name = ""
            if isinstance(enc_cfg, dict):
                enc_name = str(enc_cfg.get("name", ""))
            else:
                enc_name = str(getattr(enc_cfg, "name", ""))
            if enc_name and enc_name != "wan_vae":
                raise ValueError(
                    f"video_backbone.encoder.name='{enc_name}' is incompatible "
                    "with from_scratch=false: the pre-trained DiT's first conv "
                    "channels are bound to native Wan VAE's z_dim and cannot "
                    "consume a different encoder's latent space. Set "
                    "from_scratch=true to activate the encoder swap (and re-init "
                    "the DiT), or remove the encoder block to keep the native "
                    "Wan VAE path. See docs/external_video_encoder.md §1."
                )
            logger.info(
                "video_backbone.encoder is set but from_scratch=false; "
                "encoder block IGNORED, using native pipe.vae. Set from_scratch=true "
                "to activate the encoder swap. See docs/external_video_encoder.md."
            )

        text_dim = self._cfg_get(cfg, "text_dim", None)
        text_dim = None if text_dim in (None, 0) else int(text_dim)
        if source is not None:
            ckpt_dir = vb_cfg.get("_ckpt_dir") if isinstance(vb_cfg, dict) else getattr(vb_cfg, "_ckpt_dir", None)
            self.video_backbone = build_video_backbone(
                vb_name,
                cfg,
                source=source,
                device="cpu",
                ckpt_dir=ckpt_dir,
                external_encoder=external_encoder,
                text_dim=text_dim,
            )
        elif vb_name is not None:
            self.video_backbone = build_video_backbone(
                vb_name, cfg, external_encoder=external_encoder, text_dim=text_dim
            )

        # Cross-check: yaml-declared temporal contract must match what the
        # backbone actually exposes (sourced from external encoder spec on the
        # external path, native VAE defaults otherwise). Drift here would let
        # the dataloader enforce the wrong divisibility rule and let the
        # mask-downsampler produce a wrong-length tail, so we fail-fast at
        # backbone init. We read from the backbone (not directly from the
        # encoder spec) so the contract has a single owner — see A1's
        # dit_patch_size ABC-property design.
        if self.video_backbone is not None:
            declared_tc = self._cfg_get(vb_cfg, "temporal_compression", 4)
            declared_causal = self._cfg_get(vb_cfg, "causal_temporal", True)
            actual_tc = self.video_backbone.temporal_compression
            actual_causal = self.video_backbone.causal_temporal
            if external_encoder is not None:
                encoder_src = f"external encoder {type(external_encoder).__name__}"
            else:
                encoder_src = "native VAE"
            if (declared_tc, declared_causal) != (actual_tc, actual_causal):
                raise ValueError(
                    f"video_backbone.temporal_compression / causal_temporal yaml "
                    f"({declared_tc}, {declared_causal}) does not match {encoder_src} "
                    f"({actual_tc}, {actual_causal}). Update the yaml fields to match."
                )

        # Optional from-scratch DiT: keep the Wan video backbone structure
        # but discard the loaded DiT weights and re-randomize them in place.
        # VAE and the text encoder stay pretrained and are frozen by the
        # training strategy yaml. Reproducibility comes from
        # ``cfg.project.seed`` which ``OpenWAMTrainer`` applies before
        # architecture construction. Applies uniformly to every architecture
        # that builds its video backbone via this method (dual_system /
        # shared_backbone / tri_system).
        #
        # IMPORTANT: gated on ``source is None`` (training path only). On
        # deploy, ``cfg.video_backbone.from_scratch`` is True because the
        # config was saved from a from-scratch training run, but DiT weights
        # come from the checkpoint, NOT from a re-initialization. Calling
        # reinit here would silently wipe the trained DiT weights and the
        # subsequent ``load_checkpoint`` would overwrite them again — wasted
        # work in the best case, but if the checkpoint had any missing keys
        # the strict load would surface them against zeroed weights instead
        # of the random init, masking the diagnostic.
        if self.video_backbone is not None and from_scratch and source is None:
            if getattr(self.video_backbone, "dit", None) is None:
                logger.warning(
                    "video_backbone.from_scratch=true but backbone has no 'dit'; skipping. (Non-Wan backbone?)"
                )
            else:
                from openwam.model.video_backbone.wan.reinit import reinit_dit_from_scratch

                reinit_dit_from_scratch(
                    self.video_backbone,
                    external_encoder=external_encoder,
                    dit_patch_size=self.video_backbone.dit_patch_size,
                )
                logger.info(
                    "video_backbone.from_scratch=true: DiT re-initialized; VAE / text_encoder keep pretrained weights"
                )

        # Deploy path with an external encoder: DiT was just constructed
        # from the saved Wan ``components[dit].extra_kwargs`` (in/out_dim
        # = native Wan VAE z_dim, e.g. 48), but the saved checkpoint
        # stores the external-encoder-adapted shapes (in/out_dim =
        # encoder.spec.z_dim, e.g. 1408 for V-JEPA 2.1 ViT-g). The
        # training-side ``reinit_dit_from_scratch`` performs this
        # reshape before the random-init reset; on deploy we want the
        # reshape WITHOUT the reset so the subsequent strict
        # ``load_checkpoint`` can populate ``patch_embedding`` /
        # ``head.head`` from the safetensors. Gated on
        # ``external_encoder is not None`` so the native-VAE deploy
        # path (where Wan's saved components already match the
        # checkpoint) stays untouched.
        if self.video_backbone is not None and source is not None and external_encoder is not None:
            if getattr(self.video_backbone, "dit", None) is not None:
                from openwam.model.video_backbone.wan.reinit import adapt_dit_to_external_encoder

                adapt_dit_to_external_encoder(
                    self.video_backbone,
                    external_encoder,
                    self.video_backbone.dit_patch_size,
                )
                logger.info(
                    "Deploy with external encoder %s: DiT patch_embedding / "
                    "head.head reshaped to z_dim=%d before strict load",
                    type(external_encoder).__name__,
                    external_encoder.spec.z_dim,
                )

    @staticmethod
    def _compute_encoder_yaml_whitelist(enc_cfg, registry) -> set[str]:
        """Encoder-aware allowed-yaml-keys set.

        Always includes ``{"name", "model_path"}``; extends with the
        ``optional_yaml_keys()`` set of the registered encoder class
        named by ``enc_cfg["name"]`` (e.g. ``"vjepa2_1_forward"`` for
        V-JEPA 2.1). Unknown encoder names yield just the base pair.

        Failure ordering when the encoder name is unknown:
        - if ``enc_cfg`` carries no extras beyond ``{name, model_path}``,
          the caller's whitelist check passes and the downstream
          ``build_video_encoder`` registry lookup raises the proper
          ``KeyError`` with the available-encoders hint;
        - if ``enc_cfg`` carries extras, the caller's whitelist check
          raises ``ValueError`` first (the ``KeyError`` is masked). This
          is intentional: an unknown encoder name + non-whitelisted field
          is most likely a typo, and the whitelist diagnostic surfaces
          both problems together.
        """
        base = {"name", "model_path"}
        if isinstance(enc_cfg, dict):
            enc_name = enc_cfg.get("name")
        else:
            enc_name = getattr(enc_cfg, "name", None)
        if not enc_name or enc_name not in registry:
            return base
        return base | registry[enc_name].optional_yaml_keys()

    @staticmethod
    def _encoder_yaml_extras(enc_cfg, allowed: set[str]) -> set[str]:
        """Keys present in ``enc_cfg`` with a non-null value that are NOT
        in ``allowed``.

        Yaml-``null`` is treated as "field absent" for whitelist purposes
        so the framework yamls (``configs/model/backbone/wan.yaml``) can
        keep an inline ``encoder:`` block with discoverability fields
        like an encoder-specific knob set to ``null`` — those fields stay
        visible to operators but do not trip the whitelist when the active
        encoder (e.g. V-JEPA 2.1, V-JEPA 2, Wan VAE) does not declare them in
        ``optional_yaml_keys()``. An explicit non-null value still
        triggers the whitelist error, so a user who actually sets the
        field on the wrong encoder still gets a fail-fast.
        """
        if isinstance(enc_cfg, dict):
            items = list(enc_cfg.items())
        else:
            # OmegaConf DictConfig — iterate via .items() if available,
            # otherwise fall back to (key, getattr) pairs.
            try:
                items = list(enc_cfg.items())
            except AttributeError:
                items = [(k, getattr(enc_cfg, k, None)) for k in enc_cfg.keys()]
        return {k for k, v in items if v is not None and k not in allowed}

    @staticmethod
    def _build_external_encoder_skeleton(enc_cfg, source, *, ckpt_dir=None):
        """Deploy-time external encoder constructor.

        Reads the encoder ``name`` (subject to the same yaml whitelist as
        the training path) and reaches into the saved ``source`` dict for
        the ``components`` list to find the ``attr == "vae"`` entry. That
        entry's ``model_class`` / ``extra_kwargs`` is handed to the
        encoder class's :meth:`VideoEncoder.from_skeleton` classmethod,
        which instantiates the underlying module with zero weights. The
        architecture's :meth:`load_checkpoint` strict load fills in the
        weights immediately after.

        ``ckpt_dir`` is forwarded to ``from_skeleton`` so encoders that
        depend on side files can read them from the checkpoint dir itself,
        not from the user-side weight directory. Two patterns coexist:
        V-JEPA 2.1 prefers ``<ckpt_dir>/manifest.json`` and falls back to
        ``encoder.model_path`` for older checkpoints.

        Refuses to silently fall back to the native VAE path here: if the
        cfg has an encoder block but the components list is missing a vae
        entry (e.g. corrupted save), raise so the operator sees the
        mismatch up front.
        """
        from openwam.model.video_backbone.encoder import _VIDEO_ENCODER_REGISTRY

        # Deploy-side whitelist matches the training-side whitelist in
        # ``_init_video_backbone`` — same field set, same plumbing via
        # ``_compute_encoder_yaml_whitelist`` (encoder class extends the
        # always-allowed ``{name, model_path}`` with its
        # ``optional_yaml_keys()``).
        allowed = BaseWAMArchitecture._compute_encoder_yaml_whitelist(enc_cfg, _VIDEO_ENCODER_REGISTRY)
        extras = BaseWAMArchitecture._encoder_yaml_extras(enc_cfg, allowed)
        if extras:
            raise ValueError(
                f"video_backbone.encoder allows only {sorted(allowed)} in yaml; got extra fields {sorted(extras)}."
            )
        enc_name = enc_cfg["name"] if isinstance(enc_cfg, dict) else enc_cfg.name
        if enc_name not in _VIDEO_ENCODER_REGISTRY:
            available = ", ".join(sorted(_VIDEO_ENCODER_REGISTRY)) or "(none)"
            raise KeyError(f"Unknown video encoder '{enc_name}'. Available: {available}")

        components = (source or {}).get("components") if isinstance(source, dict) else None
        if not components:
            raise RuntimeError(
                "Deploy with encoder block but saved config has no "
                "video_backbone.components — cannot reconstruct encoder skeleton. "
                "Re-save the checkpoint with the current code, or strip the "
                "encoder block from config.yaml to fall back to native VAE."
            )
        vae_entry = next((e for e in components if e.get("attr") == "vae"), None)
        if vae_entry is None:
            raise RuntimeError(
                "Deploy with encoder block but components list has no attr=vae "
                "entry to construct the encoder skeleton from."
            )
        encoder_cls = _VIDEO_ENCODER_REGISTRY[enc_name]
        return encoder_cls.from_skeleton(vae_entry, encoder_cfg=enc_cfg, ckpt_dir=ckpt_dir)

    def _resolve_video_dim(self, cfg) -> int:
        """Resolve video_dim from config or video_backbone; raise if neither provides it."""
        dim = int(cfg.get("video_dim", 0)) if isinstance(cfg, dict) else int(getattr(cfg, "video_dim", 0))
        if dim == 0 and self.video_backbone is not None:
            dim = self.video_backbone.dim
        if not dim:
            raise ValueError("video_dim must be specified in config or inferred from video_backbone")
        return dim

    def _resolve_text_dim(self, cfg, *, default: int = 4096) -> int:
        """Resolve the architecture-level raw text/context dim.

        ``text_dim`` is shared by the video backbone, action backbone, and
        proprio-as-context encoder. It therefore lives under ``architecture:``
        in yaml, not under ``action_backbone:``. If omitted, the video backbone's
        declared ``text_dim`` is used, with Wan's 4096 as the final default.
        When both are present they must match, otherwise video/action context
        projections would be built for different input widths.
        """
        raw = self._cfg_get(cfg, "text_dim", None)
        cfg_dim = None if raw in (None, 0) else int(raw)
        vb_dim = None
        if self.video_backbone is not None:
            raw_vb_dim = getattr(self.video_backbone, "text_dim", None)
            vb_dim = None if raw_vb_dim in (None, 0) else int(raw_vb_dim)
        if cfg_dim is not None and vb_dim is not None and cfg_dim != vb_dim:
            raise ValueError(
                f"architecture.text_dim={cfg_dim} does not match video_backbone.text_dim={vb_dim}. "
                "Use a single shared raw context dimension for both video and action streams."
            )
        if cfg_dim is not None:
            return cfg_dim
        if vb_dim is not None:
            return vb_dim
        return int(default)

    @property
    def backbones(self) -> dict[str, nn.Module]:
        """All backbone modules owned by this architecture.

        Subclasses with additional backbones (e.g. TriSystem with a VLM
        backbone) should override this to include them. The returned dict
        is used by ``init_training_schedulers``, ``set_dtype_device``,
        ``move_frozen_to_device``, and ``get_component_specs`` to iterate
        over all backbones generically.
        """
        result = {}
        if self.video_backbone is not None:
            result["video_backbone"] = self.video_backbone
        if self.action_backbone is not None:
            result["action_backbone"] = self.action_backbone
        return result

    # --- Action-side properties (delegate to action_backbone) ---

    @property
    def action_scheduler(self):
        """Flow-matching scheduler for the action stream (owned by action_backbone)."""
        if self.action_backbone is None:
            raise RuntimeError("action_backbone is not initialized")
        return self.action_backbone.scheduler

    @property
    def video_scheduler(self):
        """Flow-matching scheduler for the video stream (owned by video_backbone)."""
        if self.video_backbone is None:
            raise RuntimeError("video_backbone is not initialized")
        return self.video_backbone.scheduler

    @property
    def external_encoder(self):
        """The video backbone's external encoder, or ``None`` (native VAE / no video
        backbone). Train/deploy read this instead of reaching into video_backbone
        internals (the layering boundary: only the architecture talks to backbones)."""
        vb = self.video_backbone
        return vb.external_encoder if vb is not None else None

    @property
    def action_dim(self) -> int:
        return self.action_backbone.action_dim if self.action_backbone is not None else 0

    @property
    def bridge_layers(self) -> tuple:
        return getattr(self.action_backbone, "bridge_layers", ()) if self.action_backbone is not None else ()

    @property
    def expert_layers(self) -> tuple:
        return getattr(self.action_backbone, "expert_layers", ()) if self.action_backbone is not None else ()

    @property
    def trainable_action_module(self) -> Optional[nn.Module]:
        """The nn.Module whose parameters are trained as the action model."""
        return self.action_backbone

    @property
    def uses_proprioception(self) -> bool:
        return bool(getattr(self, "_use_proprioception_context", False)) or (
            self.action_backbone is not None and self.action_backbone.uses_proprioception
        )

    def _init_proprio_context(self, cfg, *, text_dim: int = 4096) -> None:
        """Initialize FastWAM-style proprio-as-context conditioning."""
        enabled = bool(self._cfg_get(cfg, "use_proprioception", False))
        self._use_proprioception_context = enabled
        self.proprio_encoder: Optional[nn.Module] = None
        self.proprio_dim = 0
        self.context_dim = int(text_dim)
        if not enabled:
            return
        state_dim = int(self._cfg_get(cfg, "state_dim", 0) or 0)
        if state_dim <= 0:
            raise ValueError("use_proprioception=True requires explicit state_dim for context-token proprio.")
        self.proprio_dim = state_dim
        self.proprio_encoder = nn.Linear(state_dim, self.context_dim)

    def _append_proprio_context_token(self, pipeline_inputs: dict, proprio_state: Optional[Tensor]) -> dict:
        """Append one proprio token to raw text context and extend context_mask.

        If the caller passed a per-sample mask via ``pipeline_inputs['_proprio_sample_mask']``
        (shape (B,) or (B, 1) bool), masked samples get a zero token and a False
        attention mask entry. This isolates proprio_encoder gradients on those
        samples (input * 0 -> weight grad = 0; attention mask cuts the forward path).
        """
        # Always strip the internal routing key so it doesn't leak into downstream forwards,
        # even when proprio context is globally disabled.
        pipeline_inputs = dict(pipeline_inputs)
        sample_mask = pipeline_inputs.pop("_proprio_sample_mask", None)

        if not bool(getattr(self, "_use_proprioception_context", False)):
            return pipeline_inputs
        if self.proprio_encoder is None:
            raise RuntimeError("proprio context is enabled but proprio_encoder is not initialized.")
        if proprio_state is None:
            raise ValueError("use_proprioception=True requires `proprio_state` from sample['proprio'] or obs['state'].")
        if proprio_state.ndim == 1:
            proprio_state = proprio_state.unsqueeze(0)
        elif proprio_state.ndim == 3 and proprio_state.shape[1] == 1:
            proprio_state = proprio_state[:, 0, :]
        if proprio_state.ndim != 2:
            raise ValueError(f"proprio_state must be [B, D] or [B, 1, D], got shape {tuple(proprio_state.shape)}")
        if proprio_state.shape[1] != self.proprio_dim:
            raise ValueError(f"proprio_state last dim must be {self.proprio_dim}, got {proprio_state.shape[1]}")

        context = pipeline_inputs["context"]
        if context.shape[0] != proprio_state.shape[0]:
            if proprio_state.shape[0] == 1 and context.shape[0] > 1:
                proprio_state = proprio_state.expand(context.shape[0], -1)
            else:
                raise ValueError(
                    f"Batch mismatch between context and proprio_state: {context.shape[0]} vs {proprio_state.shape[0]}"
                )

        # Normalize sample_mask to (B, 1) bool on context's device.
        # Accepted incoming shapes (after stacking in _collect_inputs):
        #   * (B,)         — legacy 1D enable-per-sample
        #   * (B, 1)       — legacy "rank-2 enable"
        #   * (B, 1, D)    — 2D per-dim mask emitted by the post-migration
        #                    RoboCOIN/EgoDex/RoboTwin/OXE readers; the
        #                    sample-level enable is ``mask.any(dim=-1)`` so any
        #                    real dim still gates the token in.
        if sample_mask is None:
            sample_mask = torch.ones(
                (proprio_state.shape[0], 1),
                dtype=torch.bool,
                device=context.device,
            )
        else:
            sample_mask = sample_mask.to(device=context.device, dtype=torch.bool)
            if sample_mask.ndim == 3:
                # (B, 1, D) -> (B, 1): True if any per-dim slot is real
                sample_mask = sample_mask.any(dim=-1)
            elif sample_mask.ndim == 1:
                sample_mask = sample_mask.unsqueeze(-1)
            if sample_mask.shape != (proprio_state.shape[0], 1):
                raise ValueError(
                    f"_proprio_sample_mask shape {tuple(sample_mask.shape)} must be ({proprio_state.shape[0]}, 1)"
                )

        proprio_token = (
            self.proprio_encoder(proprio_state.to(device=context.device, dtype=self.proprio_encoder.weight.dtype))
            .to(dtype=context.dtype)
            .unsqueeze(1)
        )
        # Token * mask cuts the weight grad for mask=False samples
        # (input goes to zero so the encoder weight gradient contribution is zero).
        sample_mask_f = sample_mask.to(proprio_token.dtype).unsqueeze(-1)  # (B, 1, 1)
        proprio_token = proprio_token * sample_mask_f

        context_mask = pipeline_inputs.get("context_mask")
        if context_mask is None:
            seq_lens = pipeline_inputs.get("seq_lens")
            if seq_lens is not None:
                seq_lens = seq_lens.to(device=context.device)
                positions = torch.arange(context.shape[1], device=context.device).unsqueeze(0)
                context_mask = positions < seq_lens.unsqueeze(1)
            else:
                context_mask = torch.ones((context.shape[0], context.shape[1]), dtype=torch.bool, device=context.device)
        else:
            context_mask = context_mask.to(device=context.device, dtype=torch.bool)

        updated = dict(pipeline_inputs)
        updated["context"] = torch.cat([context, proprio_token], dim=1)
        updated["context_mask"] = torch.cat([context_mask, sample_mask], dim=1)
        # The appended proprio token can sit after padded text tokens, so the
        # resulting valid tokens are not necessarily a contiguous prefix.
        # Keep the original text seq_lens and make context_mask authoritative.
        return updated

    @property
    def action_mean(self) -> Tensor:
        if self.action_backbone is not None:
            return self.action_backbone.action_mean
        return torch.zeros(self.action_dim)

    @property
    def action_std(self) -> Tensor:
        if self.action_backbone is not None:
            return self.action_backbone.action_std
        return torch.ones(self.action_dim)

    # --- Device / dtype (top-level authority) ---

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def set_dtype_device(self, dtype: torch.dtype, device: torch.device) -> None:
        """Dispatch to each backbone — they own their own dtype/device handling."""
        self._dtype = dtype
        self._device = device
        proprio_encoder = getattr(self, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.to(dtype=dtype, device=device)
        for bb in self.backbones.values():
            bb.set_dtype_device(dtype, device)

    def attach_normalizer(self, normalizer) -> None:
        """Attach (or clear) an action normalizer used by ``generate``.

        Deployment paths build the same normalizer used by training from
        ``normalization_stats.npy``. ``generate`` uses it to return real-scale actions,
        while server-side proprio preprocessing uses it to normalize raw robot
        state into the model's training space. Pass ``None`` to clear.
        """
        self.normalizer = normalizer

    def normalize_deploy_proprio(self, proprio_state):
        """Normalize raw deploy proprio (array-like) into a float32 tensor; ``None`` passes through.

        The denoising loop re-casts to the model device/dtype, so a CPU tensor is fine.
        """
        if proprio_state is None:
            return None

        import numpy as np
        import torch

        arr = np.asarray(proprio_state, dtype=np.float32)
        normalizer = getattr(self, "normalizer", None)
        if normalizer is not None:
            arr = normalizer.normalize(arr)
        return torch.from_numpy(arr)

    # --- Checkpoint save / load ---

    def save_checkpoint(self, path: str) -> None:
        """Save architecture state to safetensors.

        VLM backbone parameters are excluded — the VLM checkpoint is saved
        as a separate directory by the trainer. This avoids tied-weight
        deduplication complexity and keeps the file small.
        """
        from safetensors.torch import save_file

        state_dict = self.state_dict()
        if getattr(self, "vlm_backbone", None) is not None:
            state_dict = _exclude_vlm_from_state_dict(state_dict)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        save_file(state_dict, path)

    def load_checkpoint(self, path: str, strict: bool = True) -> None:
        """Load architecture state from a safetensors checkpoint.

        VLM backbone weights are not stored in the safetensors file (they
        are saved as a separate directory). When a VLM backbone is present,
        missing ``vlm_backbone.*`` keys are tolerated; unexpected or missing
        non-VLM keys still raise under ``strict=True``.

        Meta-device sub-modules (self-contained deploy empty shells built
        via ``from_empty`` / ``init_empty_weights``) need
        ``load_state_dict(..., assign=True)`` — the default in-place copy is
        a silent no-op against meta tensors and leaves the shells
        unpopulated. ``assign=True`` rebinds the parameter slot to the
        safetensors tensor instead. We only flip the flag when meta params
        actually exist so the training-resume path (real-device params,
        in-place copy preserves identity) is unchanged.
        """
        from safetensors.torch import load_file

        state_dict = load_file(path)
        has_vlm = getattr(self, "vlm_backbone", None) is not None
        has_meta = any(p.device.type == "meta" for p in self.parameters())
        missing, unexpected = self.load_state_dict(state_dict, strict=False, assign=has_meta)
        if strict and not has_vlm:
            if missing or unexpected:
                raise RuntimeError(f"Strict load failed: missing={missing}, unexpected={unexpected}")
        elif strict and has_vlm:
            non_vlm_missing = [k for k in missing if not k.startswith(VLM_STATE_DICT_PREFIX)]
            if non_vlm_missing or unexpected:
                raise RuntimeError(
                    f"Strict load failed (VLM keys excluded): missing={non_vlm_missing}, unexpected={unexpected}"
                )

    # --- Training: module management ---

    def init_training_schedulers(self, num_timesteps: int = 1000) -> None:
        """Initialize all backbone schedulers for training.

        Single source of truth for each stream's α-shift:
        ``video_backbone.shift_video`` and ``action_backbone.shift_action``.
        The same properties are read by the deploy schedule at inference time,
        so the discrete training sigma buffer and the inference denoising
        trajectory are sampled from the same shifted schedule — train/inference
        cannot drift, and the shift is owned by the checkpoint config (not a
        separate deploy-time knob).

        ``shift is None`` (the default for configs without an explicit override)
        falls back to each scheduler's template default (Wan/action = 5.0),
        i.e. bit-identical pre-shift behavior.
        """
        # ``getattr`` (rather than direct attribute access) so test doubles
        # / mocks that don't carry the shift property still work — they fall
        # back to the scheduler's template default, matching the production
        # no-override path.
        for name, bb in self.backbones.items():
            if not hasattr(bb, "scheduler"):
                continue
            kwargs = {"training": True}
            shift = getattr(bb, "shift_video" if name == "video_backbone" else "shift_action", None)
            if shift is not None:
                kwargs["shift"] = float(shift)
            bb.scheduler.set_timesteps(num_timesteps, **kwargs)

    def freeze_modules(self, names: list[str]) -> list[str]:
        """Freeze named sub-modules by dotted path. Returns actually frozen names.

        Single-point freeze API. Two effects per frozen submodule:

        1. ``module.requires_grad_(False)`` — optimizer cannot update its params.
        2. ``module.forward`` is wrapped in ``torch.no_grad`` so the frozen
           subtree never saves activations for backward. This is the full
           semantic of "freeze" — neither the trainer nor any backbone needs to
           inspect freeze status separately.

        For text_encoder / vae, which are already called under the
        ``@torch.no_grad()`` ``prepare_inputs`` decorator, the wrapper is a
        no-op (nested ``no_grad``). For modules called inside the training
        forward graph (e.g. tri_system's frozen Qwen3-VL backbone), the
        wrapper is what actually saves activation memory.

        Uses ``nn.Module.get_submodule()`` so dotted paths like
        ``video_backbone.text_encoder`` work naturally; unknown names
        are silently skipped, so a freeze list mentioning modules absent on a
        given architecture (e.g. ``vlm_backbone.vlm_model`` on dual_system)
        is harmless.
        """
        frozen = []
        for name in names:
            try:
                module = self.get_submodule(name)
            except (AttributeError, KeyError):
                module = None
            if module is not None:
                module.requires_grad_(False)
                # Set eval mode on the frozen subtree. Use modules() instead
                # of .eval() to avoid infinite recursion when a submodule has
                # self-referential aliases (e.g. HF model.model = self).
                for sub in module.modules():
                    sub.training = False
                _wrap_forward_in_no_grad(module)
                frozen.append(name)
        return frozen

    def get_trainable_modules(self, freeze_list: list[str] = ()) -> dict[str, nn.Module]:
        """Return top-level trainable sub-modules.

        Walks ``self.named_children()`` and returns modules that have at
        least one parameter with ``requires_grad=True``, excluding those
        in *freeze_list*. Used by ``optimizer_groups.build_trainable_parameters``
        to source the param groups for the optimizer.
        """
        result = {}
        freeze_set = set(freeze_list)
        for name, mod in self.named_children():
            if name in freeze_set:
                continue
            if any(p.requires_grad for p in mod.parameters()):
                result[name] = mod
        return result

    def move_frozen_to_device(self, device: torch.device, names: tuple[str, ...] = ("text_encoder", "vae")) -> None:
        """Move named frozen modules to device.

        Searches via ``get_submodule`` on self first, then on each backbone.
        """
        for name in names:
            mod = None
            try:
                mod = self.get_submodule(name)
            except (AttributeError, KeyError):
                pass
            if mod is None:
                for bb in self.backbones.values():
                    found = None
                    try:
                        found = bb.get_submodule(name)
                    except (AttributeError, KeyError):
                        found = None
                    if found is not None:
                        mod = found
                        break
            if mod is not None:
                mod.to(device=device)

    def get_component_specs(self, model_path: str) -> Optional[dict]:
        """Get component specs from all backbones for self-contained checkpoint config."""
        for bb in self.backbones.values():
            if hasattr(bb, "get_component_specs"):
                specs = bb.get_component_specs(model_path)
                if specs is not None:
                    return specs
        return None

    def copy_deploy_artifacts(self, output_dir: str, cfg) -> None:
        """Delegate to each backbone so deploy-time artifacts land in ``output_dir``.

        Backbones with no external artifacts can leave the default no-op.
        """
        for bb in self.backbones.values():
            if hasattr(bb, "copy_deploy_artifacts"):
                bb.copy_deploy_artifacts(output_dir, cfg)

    def save_assets_for_deployment(self, output_dir: str, cfg) -> None:
        """Make the checkpoint directory self-contained for deploy.

        One unified entry the trainer calls once per checkpoint save, BEFORE
        ``save_config`` writes ``config.yaml``:

        1. Merge each backbone's component-reconstruction specs into ``cfg`` (so
           deploy rebuilds the module skeleton from ``config.yaml`` without the
           training-time ``model_path`` being reachable). Params land in
           ``config.yaml`` via the subsequent ``save_config``.
        2. Copy backbone artifact files (tokenizer / processor) into
           ``output_dir``.

        Replaces the former separate ``get_component_specs`` + config-merge +
        ``copy_deploy_artifacts`` dance in the trainer.
        """
        from omegaconf import OmegaConf, open_dict

        model_path = OmegaConf.select(cfg, "model.video_backbone.model_path", default=None)
        if model_path and os.path.isdir(str(model_path)):
            specs = self.get_component_specs(str(model_path))
            if specs is not None:
                with open_dict(cfg):
                    vb_cfg = cfg.model.video_backbone
                    if "components" not in vb_cfg:
                        OmegaConf.update(cfg, "model.video_backbone.components", specs["components"])
                    if "tokenizer" in specs and "tokenizer" not in vb_cfg:
                        OmegaConf.update(cfg, "model.video_backbone.tokenizer", specs["tokenizer"])
        self.copy_deploy_artifacts(output_dir, cfg)

    # --- Training: preprocessing ---

    @torch.no_grad()
    def preprocess(self, **kwargs) -> dict:
        """Encode raw frames/text into latents + context for training.

        Delegates to ``video_backbone.preprocess_input_for_train()``. External code
        (trainer) should call this instead of touching video_backbone directly.
        """
        return self.video_backbone.preprocess_input_for_train(**kwargs)

    def set_training_runtime(
        self,
        *,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        max_timestep_boundary: float = 1.0,
        min_timestep_boundary: float = 0.0,
    ) -> None:
        """Set forward-time training flags consumed by ``prepare_inputs``.

        Trainers call this once during construction. Keeping these on the
        architecture keeps ``prepare_inputs(batch)`` self-contained — the
        trainer no longer needs to thread these flags through every loss call.
        """
        self._use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self._use_gradient_checkpointing_offload = bool(use_gradient_checkpointing_offload)
        self._max_timestep_boundary = float(max_timestep_boundary)
        self._min_timestep_boundary = float(min_timestep_boundary)

    @torch.no_grad()
    def prepare_inputs(self, batch: list[dict]) -> dict:
        """Aggregate a list of dataset samples into a batched inputs dict.

        Absorbs the per-sample field collection that previously lived in
        ``OpenWAMTrainer._forward_batch``. The returned dict is designed to be
        unpacked directly into ``compute_loss`` via ``**inputs``.

        Args:
            batch: List of dataset samples (each a dict). A single dict is
                accepted as well and treated as a one-sample batch.

        Returns:
            Dict with all preprocessed video latents, text embeddings, action
            tensors, masks, and forward-time flags ready for ``compute_loss``.
        """
        from openwam.dataloader.transforms.pipeline import FirstFrameConditioningTransform
        from openwam.utils import downsample_video_mask_to_latent

        if isinstance(batch, dict):
            batch = [batch]

        if not hasattr(self, "_pipeline_transform_instance"):
            self._pipeline_transform_instance = FirstFrameConditioningTransform()
        samples = [self._pipeline_transform_instance.apply(s) for s in batch]

        _dtype = self.dtype
        _device = self.device

        all_frames: list = []
        all_prompts: list = []
        all_vace_videos: list = []
        all_ref_images: list = []
        all_pre_encoded_text: list = []
        all_actions: list = []
        all_proprios: list = []
        all_proprio_masks: list = []
        all_action_masks: list = []
        all_video_masks: list = []

        for sample in samples:
            all_frames.append(sample["video"])
            all_prompts.append(sample["prompt"])
            all_vace_videos.append(sample.get("vace_video"))
            all_ref_images.append(sample.get("first_frame_image"))
            all_pre_encoded_text.append(sample.get("pre_encoded_text"))

            action = sample.get("action")
            if action is not None:
                if isinstance(action, np.ndarray):
                    action = torch.from_numpy(action)
                action = action.to(dtype=_dtype, device=_device).unsqueeze(0)
            all_actions.append(action)

            proprio = sample.get("proprio")
            if self.uses_proprioception:
                if proprio is None:
                    raise ValueError(
                        "use_proprioception=True requires sample['proprio']; action[0] fallback is disabled."
                    )
                if isinstance(proprio, np.ndarray):
                    proprio = torch.from_numpy(proprio)
                proprio = proprio.to(dtype=_dtype, device=_device)
                if proprio.ndim == 1:
                    pass
                elif proprio.ndim == 2 and proprio.shape[0] == 1:
                    proprio = proprio[0]
                else:
                    raise ValueError(f"sample['proprio'] must be [D] or [1, D], got shape {tuple(proprio.shape)}")
            all_proprios.append(proprio)

            # Collect per-sample proprio_mask. Two accepted shapes:
            #   * 1D ``(1,) bool`` — legacy "is the proprio token enabled".
            #   * 2D ``(1, D) bool`` — per-dim mask; sample-level enable is
            #     ``pmask.any(dim=-1)``. Used by RoboCOIN/EgoDex/RoboTwin/OXE
            #     after the 2D mask migration.
            # Default for readers that don't emit the field: all True (1,).
            # (Mixed 1-D / 2-D ranks across a batch are reconciled just before
            # the stack below, so the bare (1,) default is safe.)
            pmask = sample.get("proprio_mask")
            if pmask is None:
                pmask = torch.ones(1, dtype=torch.bool)
            else:
                if isinstance(pmask, np.ndarray):
                    pmask = torch.from_numpy(pmask)
                pmask = pmask.to(dtype=torch.bool)
                if pmask.ndim == 0:
                    pmask = pmask.unsqueeze(0)
            all_proprio_masks.append(pmask)

            amask = sample.get("action_mask", None)
            vmask = sample.get("video_mask", None)
            if isinstance(amask, np.ndarray):
                amask = torch.from_numpy(amask)
            if isinstance(vmask, np.ndarray):
                vmask = torch.from_numpy(vmask)
            all_action_masks.append(amask)
            all_video_masks.append(vmask)

        ref_flags = [r is not None for r in all_ref_images]
        if any(ref_flags) and not all(ref_flags):
            raise ValueError("Mixed reference images in batch: all samples must be consistent.")

        # Optional per-sample pre-encoded text embedding (e.g. Reason1 cached
        # offline for the Cosmos25 backbone). Backbones that don't consume it
        # (Wan) silently drop the kwarg via ``**kw``. All-or-nothing per batch;
        # uniform L required for fixed-shape stacking — padded variant deferred.
        pre_text_flags = [t is not None for t in all_pre_encoded_text]
        preprocess_extra: dict = {}
        if any(pre_text_flags):
            if not all(pre_text_flags):
                raise ValueError(
                    "Mixed pre_encoded_text in batch: every sample must carry the "
                    "field, or none. Check the dataloader cache wiring."
                )
            tensors: list = []
            for t in all_pre_encoded_text:
                if isinstance(t, np.ndarray):
                    t = torch.from_numpy(t)
                if t.ndim == 3 and t.shape[0] == 1:
                    t = t[0]
                if t.ndim != 2:
                    raise ValueError(f"pre_encoded_text must be (L, D) or (1, L, D); got {tuple(t.shape)}")
                tensors.append(t)
            lens = {t.shape[0] for t in tensors}
            if len(lens) > 1:
                raise ValueError(
                    f"Inconsistent sequence length across pre_encoded_text batch: "
                    f"{sorted(lens)}. Phase 3.x requires uniform L within a batch; "
                    f"padded variant is deferred."
                )
            preprocess_extra["pre_encoded_text"] = torch.stack(
                [t.to(dtype=_dtype, device=_device) for t in tensors], dim=0
            )

        preprocessed = self.preprocess(
            frames=all_frames,
            text=all_prompts,
            vace_videos=all_vace_videos,
            ref_images=all_ref_images if ref_flags[0] else None,
            **preprocess_extra,
        )

        action_data = torch.cat(all_actions, dim=0) if all_actions[0] is not None else None

        inputs = {
            **preprocessed,
            "latents": None,
            "cfg_scale": 1,
            "cfg_merge": False,
            "tiled": False,
            "use_gradient_checkpointing": self._use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self._use_gradient_checkpointing_offload,
            "max_timestep_boundary": self._max_timestep_boundary,
            "min_timestep_boundary": self._min_timestep_boundary,
            "actions": action_data,
        }

        if self.uses_proprioception:
            inputs["proprio_state"] = torch.stack(all_proprios, dim=0).contiguous()
            # Reconcile mixed 1-D (1,) / 2-D (1, D) proprio_masks before stacking:
            # promote any 1-D enable-flag to (1, D) (broadcasts the sample-level
            # flag across all dims) so a batch mixing a 2-D reader mask with a
            # 1-D default/external mask doesn't raise on rank mismatch. All-1-D
            # and all-2-D batches are left untouched.
            if len({m.ndim for m in all_proprio_masks}) > 1:
                pdim = max((m.shape[-1] for m in all_proprio_masks if m.ndim == 2), default=1)
                all_proprio_masks = [
                    m if m.ndim == 2 else m.reshape(m.shape[0], 1).expand(m.shape[0], pdim) for m in all_proprio_masks
                ]
            inputs["proprio_mask"] = torch.stack(all_proprio_masks, dim=0).contiguous()

        if all_action_masks[0] is not None:
            inputs["action_is_pad"] = torch.stack([~m for m in all_action_masks], dim=0).to(device=_device)
        if all_video_masks[0] is not None:
            # ``latent[0]`` is a clean conditioning frame (and must be excluded
            # from the loss mask) when either:
            #   (a) the input batch carries ``first_frame_latents`` (Wan TI2V
            #       / cosmos25 TI2V — per-batch signal), in which case
            #       ``base.compute_loss`` will clean-replace ``latents[:, :, 0:1]``
            #       on every step; or
            #   (b) the backbone's *configuration* always reserves ``latent[0]``
            #       for conditioning (only TI2V via the
            #       ``fuse_vae_embedding_in_latents`` / per-token-t=0 path
            #       today).
            # Wan I2V: side-channel ``y`` carries the first-frame reference;
            # ``latent[0]`` itself is fully noised on both train and deploy
            # and must be supervised — NOT in the skip list.
            # Wan VACE: first-frame condition rides on ``vace_context``;
            # video latents are fully noised, ``latent[0]`` enters the loss
            # as a predicted frame. NOT in the skip list.
            # Cosmos25 T2V: no first-frame conditioning at all — both
            # signals off.
            skip_first = inputs.get("first_frame_latents") is not None or self.video_backbone.needs_first_frame_skip
            # Pass the backbone's temporal_compression so the tail-grouping
            # divisor matches the actual latent-T produced by the encoder.
            # The default 4 in ``downsample_video_mask_to_latent`` is the Wan
            # VAE legacy; for V-JEPA / other encoders it would silently emit
            # a wrong-length mask. See VideoBackbone.temporal_compression for
            # the source-of-truth contract.
            temporal_factor = int(self.video_backbone.temporal_compression)
            latent_masks = [
                downsample_video_mask_to_latent(~m, temporal_factor=temporal_factor, skip_first=skip_first)
                for m in all_video_masks
            ]
            inputs["video_is_pad"] = torch.stack(latent_masks, dim=0).to(device=_device)

        return inputs

    # --- ZeRO-3 external-parameter protocol ---
    #
    # The MoT driver reads several leaf ``nn.Parameter`` (e.g. ``block.modulation``,
    # ``block.wan_und_qkv``) directly inside the architecture forward, bypassing the
    # owning submodule's ``__call__``. Under DeepSpeed ZeRO-3 those leaves are
    # partitioned and the forward-pre-hook that would gather them never fires for
    # the owner. The fix is the standard external-parameter protocol: register
    # the leaves against ``self`` (the architecture) and call ``self(...)`` so the
    # architecture-level forward-pre-hook gathers them before the raw read.

    def _iter_zero3_external_params(self):
        """Yield each raw-access leaf ``nn.Parameter`` that the MoT path reads.

        Default: empty. Overridden by every architecture whose training
        forward pulls partitioned leaves outside the owner submodule's
        ``__call__`` — concretely, the MoT-driven variants:

        - ``DualSystemSelfAttnArchitecture`` — video + action ``block.modulation``
        - ``DualSystemIDMArchitecture`` — same as joint_self_attn; IDM's
          ``compute_loss`` override routes its 3-branch forward through
          ``self.__call__`` so the same protocol applies.
        - ``TriSystemJointSelfAttnArchitecture`` — also understanding
          ``block.wan_und_qkv``.

        Cross-attn and shared variants go through standard ``block.__call__``
        and don't need to override.
        """
        return ()

    def _register_zero3_externals(self) -> None:
        """Register raw-access leaves as DeepSpeed ZeRO-3 external params of ``self``.

        Required because the MoT driver reads these leaves directly, bypassing
        the owning submodule's ``__call__``. Registering them makes DeepSpeed
        gather them on ``self.__call__``'s forward-pre-hook and hold through
        backward — without this AccumulateGrad sees a size-0 leaf.

        Idempotent + no-op when deepspeed isn't importable or params lack
        ``ds_id`` (non-ZeRO-3 paths, CPU mock tests). The gate only seals after
        at least one successful register so a pre-``accelerator.prepare`` call
        (params still un-partitioned) can be retried post-prepare. Architectures
        whose iterator is empty by design (cross-attn, shared variants) seal
        immediately — they will never need to register anything.
        """
        if getattr(self, "_zero3_externals_registered", False):
            return
        leaves = list(self._iter_zero3_external_params())
        if not leaves:
            self._zero3_externals_registered = True
            return
        try:
            from deepspeed.runtime.zero import register_external_parameter
        except ImportError:
            self._zero3_externals_registered = True
            return
        registered_any = False
        for p in leaves:
            if getattr(p, "ds_id", None) is not None:
                register_external_parameter(self, p)
                registered_any = True
        if registered_any:
            self._zero3_externals_registered = True

    # --- Training: loss computation ---

    def compute_loss(
        self,
        *,
        actions: Optional[torch.Tensor] = None,
        lambda_video: float = 1.0,
        lambda_action: float = 1.0,
        current_step: int = 0,
        decoupled_sampler=None,
        action_timestep_per_token: bool = False,
        **inputs,
    ) -> dict:
        """Compute joint video-action flow matching loss.

        This is the single entry point for training loss computation.
        Handles timestep sampling, noise injection, forward pass, and
        loss calculation internally.

        Callers should produce ``inputs`` via ``self.prepare_inputs(batch)``
        (preferred) or assemble it manually with the same keys: the output of
        ``self.preprocess()`` plus any of ``actions / proprio_state /
        action_is_pad / video_is_pad / use_gradient_checkpointing[_offload] /
        max_timestep_boundary / min_timestep_boundary``.

        Args:
            actions: (B, T_action, action_dim) ground truth actions. May also
                be passed via ``inputs["actions"]``.
            lambda_video: Weight for video loss term.
            lambda_action: Weight for action loss term.
            current_step: Current training step.
            decoupled_sampler: Optional DecoupledFlowMatchLoss.
            action_timestep_per_token: Per-token action timestep sampling.
            **inputs: Preprocessed video/text tensors plus forward-time flags.

        Returns:
            dict with keys: loss, loss_video, loss_action.
        """
        vb = self.video_backbone
        action_scheduler = self.action_backbone.scheduler
        _dtype = self.dtype
        _device = self.device

        if actions is None:
            actions = inputs.pop("actions", None)
        else:
            inputs.pop("actions", None)

        max_tb = int(inputs.pop("max_timestep_boundary", 1) * len(vb.scheduler.timesteps))
        min_tb = int(inputs.pop("min_timestep_boundary", 0) * len(vb.scheduler.timesteps))
        B = inputs["input_latents"].shape[0]
        if action_timestep_per_token:
            raise ValueError(
                "action_timestep_per_token=True is not supported in the FastWAM-compatible path; "
                "action timestep must be per-sample [B]."
            )

        # --- Sample video timesteps ---
        if decoupled_sampler is not None:
            video_t, decoupled_action_t = decoupled_sampler.sample_timesteps(B, current_step=current_step, device="cpu")
            num_ts = len(vb.scheduler.timesteps)
            video_timestep_ids = (
                (video_t / decoupled_sampler.num_train_timesteps * num_ts).long().clamp(min_tb, max_tb - 1)
            )
        else:
            decoupled_action_t = None
            video_timestep_ids = torch.randint(min_tb, max_tb, (B,))

        video_timesteps = vb.scheduler.timesteps[video_timestep_ids].to(dtype=_dtype, device=_device)
        video_sigmas = vb.scheduler.sigmas[video_timestep_ids].to(dtype=_dtype, device=_device)

        # --- Add video noise (flow-matching: linear interp + velocity target) ---
        video_noise = torch.randn_like(inputs["input_latents"])
        sigma_bc = video_sigmas.view(B, 1, 1, 1, 1)
        inputs["latents"] = (1 - sigma_bc) * inputs["input_latents"] + sigma_bc * video_noise
        video_target = video_noise - inputs["input_latents"]

        if inputs.get("first_frame_latents") is not None:
            inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]

        # --- Prepare action noise ---
        noisy_actions, action_target, action_timesteps, action_timestep_ids, action_sigmas = (
            None,
            None,
            None,
            None,
            None,
        )
        if lambda_action > 0 and actions is not None:
            if decoupled_action_t is not None:
                num_ts_a = len(action_scheduler.timesteps)
                action_timestep_ids = (
                    (decoupled_action_t / decoupled_sampler.num_train_timesteps * num_ts_a)
                    .long()
                    .clamp(0, num_ts_a - 1)
                )
            else:
                action_timestep_ids = torch.randint(0, len(action_scheduler.timesteps), (B,))

            action_timesteps = action_scheduler.timesteps[action_timestep_ids].to(dtype=_dtype, device=_device)
            action_sigmas = action_scheduler.sigmas[action_timestep_ids].to(dtype=_dtype, device=_device)

            actions = actions.to(dtype=_dtype, device=_device)
            if actions.dim() == 2:
                actions = actions.unsqueeze(0)

            action_noise = torch.randn_like(actions)
            if action_sigmas.dim() == 1:
                a_sigma_bc = action_sigmas.view(B, 1, 1)
            else:
                a_sigma_bc = action_sigmas.unsqueeze(-1)
            noisy_actions = action_scheduler.add_noise(actions, action_noise, a_sigma_bc)
            action_target = action_scheduler.training_target(actions, action_noise)

        # --- Joint forward pass ---
        forward_inputs = dict(inputs)
        proprio_state = forward_inputs.pop("proprio_state", None)
        proprio_mask = forward_inputs.pop("proprio_mask", None)
        use_grad_ckpt = forward_inputs.pop("use_gradient_checkpointing", False)
        use_grad_ckpt_offload = forward_inputs.pop("use_gradient_checkpointing_offload", False)
        # Padding masks are kept in `inputs` for loss-side masking but dropped
        # from `forward_inputs` so they don't leak into vb.prepare(). Per FastWAM
        # MoT design, attention itself does not consume sample-level padding.
        forward_inputs.pop("action_is_pad", None)
        forward_inputs.pop("video_is_pad", None)

        # Route per-sample proprio_mask through pipeline_inputs to
        # ``_append_proprio_context_token``. Internal-only key; pop'd there.
        if proprio_mask is not None:
            forward_inputs["_proprio_sample_mask"] = proprio_mask

        # Use ``self(...)`` (not ``self.forward(...)``) so ``nn.Module.__call__``
        # is invoked and the architecture-level forward-pre-hook fires. Under
        # DeepSpeed ZeRO-3 that hook gathers the leaves registered by
        # ``_register_zero3_externals`` (raw-access params read by the MoT
        # driver). On non-ZeRO-3 paths this is a no-op detour through the empty
        # hook chain.
        self._register_zero3_externals()
        video_noise_pred, action_noise_pred = self(
            noisy_actions if lambda_action > 0 else None,
            action_timesteps if lambda_action > 0 else None,
            proprio_state=proprio_state,
            use_gradient_checkpointing=use_grad_ckpt,
            use_gradient_checkpointing_offload=use_grad_ckpt_offload,
            **forward_inputs,
            timestep=video_timesteps,
        )

        # --- Video loss ---
        loss_video = self._compute_video_loss(
            video_noise_pred,
            video_target,
            video_timestep_ids,
            inputs,
            _device,
        )

        if lambda_action == 0 or action_noise_pred is None:
            return {
                "loss": lambda_video * loss_video,
                "loss_video": lambda_video * loss_video.detach(),
                "loss_action": torch.tensor(0.0, device=loss_video.device),
            }

        # --- Action loss ---
        loss_action = self._compute_action_loss(
            action_noise_pred,
            action_target,
            action_timestep_ids,
            action_scheduler,
            inputs,
            _device,
        )

        if lambda_video == 0:
            loss = lambda_action * loss_action
        else:
            loss = lambda_video * loss_video + lambda_action * loss_action

        return {
            "loss": loss,
            "loss_video": lambda_video * loss_video.detach(),
            "loss_action": lambda_action * loss_action.detach(),
        }

    def _compute_video_loss(self, noise_pred, target, timestep_ids, inputs, device):
        """Per-sample weighted video MSE loss."""
        import torch.nn.functional as F

        num_clean_prefix = int(inputs.get("num_clean_prefix_frames", 0) or 0)
        video_is_pad = inputs.get("video_is_pad")

        n_skip = 0
        if inputs.get("first_frame_latents") is not None:
            # TI2V (Wan + cosmos25): trim the leading clean conditioning
            # latent(s) from the loss. Wan adapter emits
            # ``num_clean_prefix_frames=0`` (one implicit conditioning latent
            # at index 0); cosmos25 wrapper emits ``num_clean_prefix_frames=1``
            # (explicit count). Both should drop exactly the conditioning
            # latent(s), so use ``max(prefix, 1)``. VACE never enters this
            # branch — its conditioning rides on ``vace_context``, the video
            # latent path is fully noised + fully supervised.
            n_skip = max(num_clean_prefix, 1)
        elif num_clean_prefix > 0:
            # Clean-prefix flagged without first_frame_latents: trim prefix
            # plus the first VAE-conditioning latent that
            # ``downsample_video_mask_to_latent`` also excludes from the mask.
            n_skip = num_clean_prefix + 1
        elif video_is_pad is not None and video_is_pad.shape[-1] < noise_pred.shape[2]:
            # Production tail-mask convention (no ref-prefix backbone such as
            # I2V): ``video_is_pad`` is sized to T_lat minus the leading
            # conditioning latents. Trim noise_pred / target to match.
            n_skip = noise_pred.shape[2] - video_is_pad.shape[-1]

        if n_skip > 0:
            noise_pred = noise_pred[:, :, n_skip:]
            target = target[:, :, n_skip:]

        vb = self.video_backbone
        tw = vb.scheduler.linear_timesteps_weights[timestep_ids].to(dtype=torch.float32, device=device)

        per_element = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
        per_frame = per_element.mean(dim=(1, 3, 4))

        if video_is_pad is not None:
            if video_is_pad.shape[-1] != noise_pred.shape[2]:
                raise ValueError(
                    f"video_is_pad length {video_is_pad.shape[-1]} does not match "
                    f"trimmed noise_pred T={noise_pred.shape[2]} (n_skip={n_skip}). "
                    "Expected mask sized to T_lat minus leading conditioning latents."
                )
            video_is_pad = video_is_pad.to(device=per_frame.device, dtype=torch.bool)
            valid_mask = ~video_is_pad
            per_frame = per_frame * valid_mask.float()
            valid_count = valid_mask.float().sum(dim=1).clamp(min=1)
            per_sample = per_frame.sum(dim=1) / valid_count
        else:
            per_sample = per_frame.mean(dim=1)

        return (per_sample * tw).mean()

    def _compute_action_loss(self, noise_pred, target, timestep_ids, scheduler, inputs, device):
        """Per-sample weighted action MSE loss.

        Supports two action_is_pad shapes:
          * **(B, T) bool** — legacy per-timestep mask (pre-2D migration).
            Each masked timestep contributes 0 to per-sample loss; per_sample =
            sum_t(loss_t) / N_valid_t, where loss_t = mean over D dims.
          * **(B, T, D) bool** — 2-D mask covering both time AND per-dim
            validity (new default for RoboCOIN/EgoDex/RoboTwin/OXE). Each
            (t, d) cell contributes only when mask[t, d] is True; per_sample =
            sum_{t,d}(loss_{t,d}) / N_valid_cells.

        The two paths are *mathematically equivalent* whenever the 2D mask
        is a broadcast of the 1D time mask across all D dims (i.e. every
        valid timestep has every dim valid): both reduce to
        sum_{t,d}(loss_{t,d}) / (N_valid_t * D). Verified by §7.4 regression
        tests under plans/oxe_dataloaders.md.

        A (T, 2) per-hand mask is handled by the legacy 2D path
        (rank-3 mask, per-element broadcasting), unchanged by this migration.
        """
        import torch.nn.functional as F

        tw = scheduler.training_weight(timestep_ids).to(dtype=torch.float32, device=device)
        if tw.ndim != 1:
            raise ValueError(f"action loss weights must be per-sample [B], got shape {tuple(tw.shape)}")
        per_element = F.mse_loss(noise_pred.float(), target.float(), reduction="none")

        action_is_pad = inputs.get("action_is_pad")

        if action_is_pad is None:
            per_sample = per_element.mean(dim=(1, 2))
            return (per_sample * tw).mean()

        action_is_pad = action_is_pad.to(device=per_element.device, dtype=torch.bool)
        valid_mask_f = (~action_is_pad).float()

        # New default path: (B, T, D) per-element mask that exactly matches
        # per_element's shape. Used by RoboCOIN/EgoDex/RoboTwin/OXE after the
        # 2D mask migration.
        if valid_mask_f.shape == per_element.shape:
            weighted = per_element * valid_mask_f
            per_sample = weighted.sum(dim=(1, 2)) / valid_mask_f.sum(dim=(1, 2)).clamp(min=1)
            return (per_sample * tw).mean()

        # Legacy fallback path. Handles:
        #   (a) 1D ``(B, T)`` per-timestep mask (pre-migration contract; still
        #       valid for any reader that didn't migrate).
        #   (b) ``(B, T, K)`` mask whose K dim does NOT match per_element's D
        #       (e.g. predictor stub used in tests where pred ∈ R^{T×2} but
        #       reader emits 14-dim mask; OR a per-hand mask). Collapse
        #       to per-step via ``any(dim=-1)`` so "any dim valid → timestep
        #       contributes" — preserves legacy semantics.
        if valid_mask_f.ndim == 3:
            valid_mask_f = (valid_mask_f > 0).any(dim=-1).float()
        per_step = per_element.mean(dim=2)
        per_step = per_step * valid_mask_f
        valid_count = valid_mask_f.sum(dim=1).clamp(min=1)
        per_sample = per_step.sum(dim=1) / valid_count
        return (per_sample * tw).mean()

    # --- Inference: generation ---

    @torch.no_grad()
    def generate(
        self,
        schedule,
        prompt: str,
        *,
        vace_video=None,
        first_frame_image=None,
        num_frames: int = 49,
        action_num_frames: Optional[int] = None,
        height: int = 384,
        width: int = 320,
        seed: int = 42,
        tiled: bool = True,
        input_video_latents: Optional[Tensor] = None,
        num_inference_steps: int = 50,
        shift: float = 5.0,
        tile_size: tuple = None,
        tile_stride: tuple = None,
        dit_cache=None,
        decode_video: bool = True,
        profile: bool = False,
        vace_cache: Optional[dict] = None,
        prompt_embed_cache: Optional[dict] = None,
        proprio_state: Optional[Tensor] = None,
        cfg_scale: float = 1.0,
        cfg_merge: bool = False,
        pre_encoded_text: Optional[Tensor] = None,
        uncond_pre_encoded_text: Optional[Tensor] = None,
        **extra_pipeline_inputs: Any,
    ) -> dict:
        """Execute joint video-action denoising driven by a schedule.

        This is the single entry point for inference. External code
        (engine) should call this instead of touching video_backbone directly.

        Args:
            num_frames: Video frame count passed to the video backbone. For
                RoboTwin this is the post-``video_stride`` count seen during
                training, not the raw action window length.
            action_num_frames: Raw state/action window length. Generated
                action chunk length is ``action_num_frames - 1``. Defaults to
                ``num_frames`` for datasets whose video/action rates match.

        Returns:
            dict with ``video`` (list of PIL images or None) and
            ``actions`` ((T, action_dim) numpy array).
        """
        import time

        from tqdm import tqdm

        # Defensive: deploy/model_loader.py:161 already flips eval at load,
        # but ad-hoc callers (notebooks, mid-training eval callbacks) might
        # invoke `generate()` without going through that path. Idempotent
        # — guards CFG dropout (e.g. Cosmos25 §14.7) and any other
        # training-only behavior from firing during inference.
        self.eval()

        vb = self.video_backbone
        device = self.device
        dtype = self.dtype

        t0 = time.time()

        # §15 — validate CFG up front. CFG vs dit_cache (§15.D6): cond/uncond
        # use the same cache key, so re-running uncond off a cond-tagged hit
        # would silently corrupt the velocity prediction. MVP keeps it simple
        # — disable the cache whenever CFG is on; future work can add a
        # (cond, uncond) slot. Everything else flows through ``InferenceInputs``
        # which has unambiguous defaults; backbones that don't implement CFG
        # (Wan) simply ignore those fields.
        cfg_scale_f = float(cfg_scale)
        if cfg_scale_f < 1.0:
            raise ValueError(f"cfg_scale must be >= 1.0; got {cfg_scale!r}.")
        if cfg_scale_f > 1.0:
            dit_cache = None

        action_num_frames = int(action_num_frames if action_num_frames is not None else num_frames)

        from openwam.model.inference_inputs import InferenceInputs

        inference_inputs = InferenceInputs(
            prompt=prompt,
            vace_video=vace_video,
            first_frame_image=first_frame_image,
            pre_encoded_text=pre_encoded_text,
            uncond_pre_encoded_text=uncond_pre_encoded_text,
            num_frames=num_frames,
            height=height,
            width=width,
            seed=seed,
            num_inference_steps=num_inference_steps,
            shift=shift,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
            vace_cache=vace_cache,
            prompt_embed_cache=prompt_embed_cache,
            cfg_scale=cfg_scale_f,
            cfg_merge=cfg_merge,
        )
        inputs_shared = vb.preprocess_input_for_inference(inference_inputs)

        if profile:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            logger.info("[WAM_PROFILE] pipeline_prep: %.3fs", time.time() - t0)

        if input_video_latents is not None:
            inputs_shared["latents"] = input_video_latents
        # Architecture-specific pipeline inputs (e.g. tri_system's vlm_inputs /
        # vlm_hidden / vlm_attention_mask) are forwarded as-is. Subclasses
        # extract what they recognize in their forward(); unrelated architectures
        # never see these keys because callers only pass them via super().generate.
        for key, value in extra_pipeline_inputs.items():
            if value is not None:
                inputs_shared[key] = value
        ref_latents = inputs_shared.get("first_frame_latents")
        if ref_latents is not None:
            latents = inputs_shared["latents"].clone()
            latents[:, :, : ref_latents.shape[2]] = ref_latents
            inputs_shared["latents"] = latents
        if self.uses_proprioception:
            if proprio_state is None:
                raise ValueError("use_proprioception=True requires `proprio_state` during generation.")
            inputs_shared["proprio_state"] = proprio_state.to(device=device, dtype=dtype)

        action_latents = torch.randn(
            1,
            action_num_frames - 1,
            self.action_dim,
            device=device,
            dtype=dtype,
            generator=torch.Generator(device=device).manual_seed(seed),
        )

        num_train_ts_v = float(self.video_scheduler.num_train_timesteps)
        num_train_ts_a = float(self.action_scheduler.num_train_timesteps)

        t_loop = time.time()

        for i in tqdm(range(len(schedule) - 1), desc="Joint denoising"):
            t_v, t_a = schedule[i]
            t_v_next, t_a_next = schedule[i + 1]

            sigma_v = t_v / num_train_ts_v
            sigma_a = t_a / num_train_ts_a
            sigma_v_next = t_v_next / num_train_ts_v
            sigma_a_next = t_a_next / num_train_ts_a

            video_stepping = sigma_v != sigma_v_next
            action_stepping = sigma_a != sigma_a_next

            if not video_stepping and not action_stepping:
                continue

            v_timestep = torch.tensor([t_v], dtype=dtype, device=device)
            a_timestep = torch.tensor([t_a], dtype=dtype, device=device) if action_stepping else None

            if dit_cache is not None and video_stepping and not dit_cache.should_recompute(sigma_v):
                # Reuse cached video noise prediction; still call forward() with
                # noisy_actions=None to skip the action stream cleanly. This is
                # only valid when video_stepping=True (the only path that
                # populates the cache).
                noise_pred = dit_cache.get_cached()
                action_noise_pred = None
                if action_stepping:
                    # Re-run action with a fresh forward pass; without cached
                    # bridges we just rerun video too. Acceptable at this scale.
                    torch.compiler.cudagraph_mark_step_begin()
                    noise_pred, action_noise_pred = self.forward(
                        action_latents,
                        a_timestep,
                        **inputs_shared,
                        timestep=v_timestep,
                    )
            else:
                forward_action_latents = action_latents if action_stepping else None
                if cfg_scale_f > 1.0:
                    # _forward_with_cfg runs its own cudagraph_mark_step_begin()
                    # before each inner forward (1 for cfg_merge, 2 for sequential).
                    noise_pred, action_noise_pred = self._forward_with_cfg(
                        action_latents=forward_action_latents,
                        a_timestep=a_timestep,
                        inputs_shared=inputs_shared,
                        v_timestep=v_timestep,
                        cfg_scale=cfg_scale_f,
                        cfg_merge=bool(cfg_merge),
                    )
                else:
                    torch.compiler.cudagraph_mark_step_begin()
                    noise_pred, action_noise_pred = self.forward(
                        forward_action_latents,
                        a_timestep,
                        **inputs_shared,
                        timestep=v_timestep,
                    )
                if dit_cache is not None and video_stepping:
                    dit_cache.update(noise_pred, sigma_v)

            if video_stepping:
                new_latents = inputs_shared["latents"] + noise_pred * (sigma_v_next - sigma_v)
                ref_latents = inputs_shared.get("first_frame_latents")
                if ref_latents is not None:
                    new_latents = new_latents.clone()
                    new_latents[:, :, : ref_latents.shape[2]] = ref_latents
                inputs_shared["latents"] = new_latents

            if action_stepping and action_noise_pred is not None:
                action_latents = self.action_scheduler.flow_step(
                    action_noise_pred, sigma_a, sigma_a_next, action_latents
                )

        if profile:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            logger.info("[WAM_PROFILE] denoising_loop: %.3fs", time.time() - t_loop)

        # VAE decode. Fail-fast when the backbone is wired to an irreversible
        # external encoder — silently returning None would mask a config
        # mismatch (caller asked for pixels but the encoder cannot produce them).
        if decode_video:
            _assert_decode_video_supported(vb)
            video_frames = vb.decode_video(inputs_shared["latents"], tiled=tiled)
        else:
            video_frames = None

        actions = action_latents.squeeze(0).float().cpu().numpy()
        normalizer = getattr(self, "normalizer", None)
        if normalizer is not None:
            actions = normalizer.unnormalize(actions)

        return {"video": video_frames, "actions": actions}

    # --- §15: Classifier-Free Guidance helpers (inference-time) ---

    def _forward_with_cfg(
        self,
        *,
        action_latents: Optional[Tensor],
        a_timestep: Optional[Tensor],
        inputs_shared: dict,
        v_timestep: Tensor,
        cfg_scale: float,
        cfg_merge: bool,
    ) -> tuple:
        """Run cond + uncond forwards and combine via ``pred = uncond + s·(cond - uncond)``.

        Two paths gated on ``cfg_merge``:

        - ``cfg_merge=False`` (default): two sequential forwards. The cond
          branch consumes ``inputs_shared['context']`` unchanged; the uncond
          branch temporarily swaps in ``inputs_shared['uncond_context']`` and
          swaps back via ``try/finally``.
        - ``cfg_merge=True``: stack ``[uncond, cond]`` along batch axis 0 for
          all batch-shaped tensors in ``inputs_shared`` plus ``action_latents``
          and the timesteps, then chunk the merged output. One forward, ~B=2
          memory peak.

        Action and video streams share the same text context, so a single
        ``cfg_scale`` is applied to both ``noise_pred`` and (when present)
        ``action_noise_pred``.
        """
        uncond_context = inputs_shared.get("uncond_context")
        if not isinstance(uncond_context, Tensor):
            raise RuntimeError(
                "CFG combine requested but `inputs_shared['uncond_context']` is missing "
                "or not a tensor. `preprocess_input_for_inference` should populate it when "
                "cfg_scale > 1.0."
            )

        if cfg_merge:
            expanded, exp_al, exp_vt, exp_at = _expand_inputs_for_cfg(
                inputs_shared,
                action_latents=action_latents,
                v_timestep=v_timestep,
                a_timestep=a_timestep,
            )
            torch.compiler.cudagraph_mark_step_begin()
            merged_noise, merged_action = self.forward(exp_al, exp_at, **expanded, timestep=exp_vt)
            uncond_noise, cond_noise = merged_noise.chunk(2, dim=0)
            noise_pred = _combine_cfg(uncond_noise, cond_noise, cfg_scale)
            if isinstance(merged_action, Tensor):
                uncond_action, cond_action = merged_action.chunk(2, dim=0)
                action_noise_pred = _combine_cfg(uncond_action, cond_action, cfg_scale)
            else:
                action_noise_pred = None
            return noise_pred, action_noise_pred

        # Sequential path: cond → uncond → combine. Mark before each forward
        # so CUDA Graph tree sees both as distinct dispatch sites.
        torch.compiler.cudagraph_mark_step_begin()
        cond_noise, cond_action = self.forward(action_latents, a_timestep, **inputs_shared, timestep=v_timestep)
        saved_context = inputs_shared["context"]
        inputs_shared["context"] = uncond_context
        try:
            torch.compiler.cudagraph_mark_step_begin()
            uncond_noise, uncond_action = self.forward(action_latents, a_timestep, **inputs_shared, timestep=v_timestep)
        finally:
            inputs_shared["context"] = saved_context

        noise_pred = _combine_cfg(uncond_noise, cond_noise, cfg_scale)
        if isinstance(cond_action, Tensor) and isinstance(uncond_action, Tensor):
            action_noise_pred = _combine_cfg(uncond_action, cond_action, cfg_scale)
        else:
            # One side dropped the action stream; keep cond as-is.
            action_noise_pred = cond_action
        return noise_pred, action_noise_pred

    # --- Deploy helpers (combine action module + video backbone) ---

    def apply_compile_optimizations(self, compile_cfg) -> None:
        """Apply architecture-specific deploy-time compile optimizations."""
        mode = compile_mode(compile_cfg, default="none", strict=True)
        if mode in (None, "auto", "none"):
            return
        logger.warning(
            "torch.compile mode '%s' is not implemented for %s; running eager.",
            mode,
            type(self).__name__,
        )

    @abstractmethod
    def forward(
        self,
        noisy_actions: Optional[Tensor],
        action_timestep: Optional[Tensor],
        *,
        proprio_state: Optional[Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **pipeline_inputs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Run the joint video + action forward.

        Each concrete architecture implements its own forward end-to-end —
        runs the video DiT block loop, captures or interleaves with the
        action stream as appropriate, and returns
        ``(video_noise_pred, action_noise_pred)``. When ``noisy_actions`` is
        None (CFG nega pass / video-only generation) the action term is
        None.
        """
        ...


# ----------------------------------------------------------------------
# §15 — Classifier-Free Guidance helpers (module-level so they stay
# stateless / testable without a full architecture instance).
# ----------------------------------------------------------------------


def _combine_cfg(uncond: Tensor, cond: Tensor, scale: float) -> Tensor:
    """Linear CFG combine: ``uncond + scale·(cond - uncond)``.

    Matches upstream Cosmos formula in
    ``cosmos_predict2/_src/predict2/models/text2world_model_rectified_flow.py:484-488``.
    """
    return uncond + float(scale) * (cond - uncond)


# Keys in ``inputs_shared`` that carry a leading batch axis and therefore
# need duplication when stacking ``[uncond, cond]`` for cfg_merge=True.
_CFG_BATCH_AXIS_KEYS: tuple = (
    "latents",
    "input_latents",
    "proprio_state",
    "first_frame_latents",
    "seq_lens",
    "context_mask",
    # cosmos25 TI2V emits ``condition_mask`` of shape (B, 1, T_lat, H_lat, W_lat)
    # in ``_finalize_ti2v_inputs`` and the wrapper cats it to ``x_in`` along
    # dim=1; cfg_merge=True must double B here or that cat shape-mismatches.
    "condition_mask",
)


def _expand_inputs_for_cfg(
    inputs_shared: dict,
    *,
    action_latents: Optional[Tensor],
    v_timestep: Tensor,
    a_timestep: Optional[Tensor],
) -> Tuple[dict, Optional[Tensor], Tensor, Optional[Tensor]]:
    """Stack ``[uncond, cond]`` along batch axis for the cfg_merge=True path.

    Returns ``(expanded_inputs_shared, action_latents, v_timestep, a_timestep)``
    where the inputs_shared copy has:

    - ``context`` replaced by ``cat([uncond_context, cond_context], dim=0)``
    - ``uncond_context`` cleared (downstream forwards don't read it)
    - every other batch-axis tensor in ``_CFG_BATCH_AXIS_KEYS`` duplicated

    Scalar / None entries are passed through unchanged.
    """
    uncond_context = inputs_shared["uncond_context"]
    cond_context = inputs_shared["context"]
    expanded = dict(inputs_shared)
    expanded["context"] = torch.cat([uncond_context, cond_context], dim=0)
    expanded["uncond_context"] = None

    # proprio_state can come in raw 1D ``(D,)`` shape (the architecture's
    # ``_compute_proprio_state`` normalises inside forward); cfg_merge
    # stacks BEFORE forward so we must normalise to ``(B, D)`` first,
    # otherwise ``cat([(D,), (D,)], dim=0)`` lands on ``(2·D,)`` and the
    # last-dim check downstream raises. Mirrors the (B, 1, D) → (B, D)
    # squeeze the architecture itself does.
    proprio = expanded.get("proprio_state")
    if isinstance(proprio, Tensor):
        if proprio.ndim == 1:
            expanded["proprio_state"] = proprio.unsqueeze(0)
        elif proprio.ndim == 3 and proprio.shape[1] == 1:
            expanded["proprio_state"] = proprio[:, 0, :]

    for key in _CFG_BATCH_AXIS_KEYS:
        v = expanded.get(key)
        if isinstance(v, Tensor):
            expanded[key] = torch.cat([v, v], dim=0)
    al = torch.cat([action_latents, action_latents], dim=0) if isinstance(action_latents, Tensor) else None
    vt = torch.cat([v_timestep, v_timestep], dim=0) if isinstance(v_timestep, Tensor) else v_timestep
    at = torch.cat([a_timestep, a_timestep], dim=0) if isinstance(a_timestep, Tensor) else None
    return expanded, al, vt, at
