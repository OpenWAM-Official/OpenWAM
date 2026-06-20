"""Semantic VAE (S-VAE) feature reducer for external video-encoder latents.

A small per-token variational autoencoder that compresses a high-dimensional
per-token feature vector (e.g. 1408-d V-JEPA 2.1 latents) down to a compact
latent (e.g. 48-d, to match the Wan VAE channel count) and reconstructs it.
It is trained separately as a reconstruction + KL objective, then frozen; the
compact latent — the posterior mean at eval time — becomes the world-model
prediction proxy.

Design notes
------------
* Operates on a channel-first 5D tensor ``(B, C, T, H, W)`` so it drops
  straight into the encoder's ``batch_encode`` latent. Internally it reshapes
  to ``(B*T, H*W, C)`` so the Transformer blocks attend over the per-frame
  spatial tokens (compression itself is per token, on the channel axis).
* Optional per-channel input standardisation (buffers filled from dataset
  statistics) keeps the reconstruction objective from being dominated by a
  few high-variance channels. The whole network operates in the standardised
  space; ``decode`` de-standardises back to the raw feature scale.
* The Transformer blocks run at the full ``input_dim`` (a flat encoder /
  decoder around a single ``input_dim -> 2*latent_dim`` Gaussian bottleneck),
  so the only width reduction is the latent projection itself.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# On-disk checkpoint schema version for the standalone-trained ``svae.pt``
# ({format_version, model_config, state_dict, ...}). Bump when the structural
# config layout changes; ``load_svae`` rejects anything else. The trainer
# stamps the same constant on save (single source of truth). v2 dropped the
# ``hidden_dim``/``mlp_ratio`` knobs (flat Transformer at ``input_dim``) and
# added the absolute ``intermediate_size`` FFN width.
_CHECKPOINT_FORMAT_VERSION = 2

# LayerNorm epsilon used throughout (matches the reference ViT-style block).
_LN_EPS = 1e-12


class DiagonalGaussian(nn.Module):
    """Split a ``(*, 2*d)`` tensor into ``mu``/``logvar`` and sample.

    Reparameterised sample while training; deterministic ``mu`` at eval. The
    ``self.training`` flag is toggled by ``nn.Module.train()/eval()`` on the
    parent module, so the same forward serves both phases.
    """

    # logvar clamp keeps ``exp(logvar)`` finite under bf16 autocast.
    _LOGVAR_MIN = -30.0
    _LOGVAR_MAX = 20.0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = x.chunk(2, dim=-1)
        logvar = logvar.clamp(self._LOGVAR_MIN, self._LOGVAR_MAX)
        if self.training:
            std = (0.5 * logvar).exp()
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        return z, mu, logvar


def _transformer_block(dim: int, num_heads: int, intermediate_size: int, dropout: float) -> nn.Module:
    """A pre-LN Transformer encoder block (MHSA + GELU MLP), batch-first.

    Equivalent to the reference ViT-style block: pre-LayerNorm self-attention
    + pre-LayerNorm GELU feed-forward, with ``intermediate_size`` the absolute
    FFN hidden width and ``_LN_EPS`` the LayerNorm epsilon.
    """
    return nn.TransformerEncoderLayer(
        d_model=dim,
        nhead=num_heads,
        dim_feedforward=int(intermediate_size),
        dropout=dropout,
        activation="gelu",
        layer_norm_eps=_LN_EPS,
        batch_first=True,
        norm_first=True,
    )


class SVAE(nn.Module):
    """Per-token semantic VAE: ``input_dim`` <-> ``latent_dim``.

    Encoder ``Es``: ``num_layers`` Transformer blocks at ``input_dim`` (self-attn
    over the per-frame spatial tokens) -> ``LayerNorm`` -> ``Linear(input_dim ->
    2*latent_dim)`` -> diagonal-Gaussian reparameterisation. Decoder ``Ds``:
    ``Linear(latent_dim -> input_dim)`` -> ``num_layers`` Transformer blocks at
    ``input_dim`` -> ``LayerNorm``.

    All I/O tensors are channel-first 5D ``(B, C, T, H, W)``.
    """

    def __init__(
        self,
        input_dim: int = 1408,
        latent_dim: int = 48,
        num_heads: int = 16,
        num_layers: int = 3,
        intermediate_size: int = 2048,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim % num_heads != 0:
            raise ValueError(f"input_dim ({input_dim}) must be divisible by num_heads ({num_heads}).")
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.intermediate_size = int(intermediate_size)
        self.dropout = float(dropout)

        # Per-channel input standardisation. Default identity (mean 0 / std 1);
        # filled from dataset statistics before training. Persistent so they
        # travel with the checkpoint and reproduce the exact train-time map.
        self.register_buffer("input_mean", torch.zeros(self.input_dim))
        self.register_buffer("input_std", torch.ones(self.input_dim))

        # Encoder: blocks at input_dim, then a single Gaussian projection head.
        self.enc_blocks = nn.ModuleList(
            [_transformer_block(self.input_dim, num_heads, intermediate_size, dropout) for _ in range(self.num_layers)]
        )
        self.enc_norm = nn.LayerNorm(self.input_dim, eps=_LN_EPS)
        self.enc_proj = nn.Linear(self.input_dim, self.latent_dim * 2)

        self.gaussian = DiagonalGaussian()

        # Decoder: lift latent back to input_dim, blocks at input_dim, final LN.
        self.dec_proj = nn.Linear(self.latent_dim, self.input_dim)
        self.dec_blocks = nn.ModuleList(
            [_transformer_block(self.input_dim, num_heads, intermediate_size, dropout) for _ in range(self.num_layers)]
        )
        self.dec_norm = nn.LayerNorm(self.input_dim, eps=_LN_EPS)

    # ------------------------------------------------------------------ #
    # config (used for the deploy-side self-contained sidecar)
    # ------------------------------------------------------------------ #
    def config_dict(self) -> Dict[str, Any]:
        """Structural hyper-parameters needed to rebuild an identical shell."""
        return {
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "intermediate_size": self.intermediate_size,
            "dropout": self.dropout,
        }

    def set_input_stats(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6) -> None:
        """Install per-channel standardisation statistics (in-place)."""
        mean = mean.reshape(-1).to(self.input_mean)
        std = std.reshape(-1).to(self.input_std).clamp_min(eps)
        if mean.numel() != self.input_dim or std.numel() != self.input_dim:
            raise ValueError(
                f"input stats must have {self.input_dim} channels, got mean={mean.numel()}, std={std.numel()}."
            )
        self.input_mean.copy_(mean)
        self.input_std.copy_(std)

    def _apply(self, fn, recurse=True):
        # Keep the per-channel standardisation stats in fp32 across a host
        # ``.to(dtype)`` (e.g. the deploy-time bf16 cast). They are
        # applied via an explicit ``.to(x.dtype)`` in ``_standardize``, so fp32
        # storage preserves their precision (these are dataset statistics, not
        # compute tensors) without forcing the reduce off the host's bf16 compute
        # dtype. This keeps the standardisation identical between standalone
        # S-VAE training (fp32 buffers) and in-pipeline application.
        super()._apply(fn, recurse=recurse)
        if self.input_mean.dtype != torch.float32:
            self.input_mean = self.input_mean.float()
        if self.input_std.dtype != torch.float32:
            self.input_std = self.input_std.float()
        return self

    # ------------------------------------------------------------------ #
    # 5D <-> sequence helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_seq(z: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int, int, int]]:
        """``(B, C, T, H, W)`` -> ``(B*T, H*W, C)`` plus shape metadata."""
        B, C, T, H, W = z.shape
        x = z.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)
        return x, (B, C, T, H, W)

    @staticmethod
    def _from_seq(x: torch.Tensor, meta: Tuple[int, int, int, int, int], out_c: int) -> torch.Tensor:
        """``(B*T, H*W, out_c)`` -> ``(B, out_c, T, H, W)``."""
        B, _C, T, H, W = meta
        return x.reshape(B, T, H, W, out_c).permute(0, 4, 1, 2, 3).contiguous()

    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.input_mean.to(x.dtype)) / self.input_std.to(x.dtype)

    def _destandardize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.input_std.to(x.dtype) + self.input_mean.to(x.dtype)

    # ------------------------------------------------------------------ #
    # core (operate in the standardised feature space)
    # ------------------------------------------------------------------ #
    def _encode_params(self, x_raw: torch.Tensor) -> torch.Tensor:
        """Standardised raw features -> concatenated ``(mu, logvar)`` params."""
        x = self._standardize(x_raw)
        for blk in self.enc_blocks:
            x = blk(x)
        x = self.enc_norm(x)
        return self.enc_proj(x)  # (B*T, N, 2*latent_dim)

    def _encode_seq(self, x_raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.gaussian(self._encode_params(x_raw))  # z, mu, logvar

    def _decode_seq_std(self, z: torch.Tensor) -> torch.Tensor:
        """Latent -> reconstruction in the *standardised* feature space."""
        x = self.dec_proj(z)
        for blk in self.dec_blocks:
            x = blk(x)
        return self.dec_norm(x)

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def encode(self, z: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Raw 5D features -> compact 5D latent.

        Training: returns ``(z, mu, logvar)`` (sampled ``z``). Eval: returns
        ``mu`` only (deterministic). NOTE the return type is mode-dependent —
        the world-model main path should call :meth:`encode_mean`, whose
        contract is stable regardless of ``self.training``.
        """
        x, meta = self._to_seq(z)
        z_l, mu, logvar = self._encode_seq(x)
        if self.training:
            return (
                self._from_seq(z_l, meta, self.latent_dim),
                self._from_seq(mu, meta, self.latent_dim),
                self._from_seq(logvar, meta, self.latent_dim),
            )
        return self._from_seq(mu, meta, self.latent_dim)

    def encode_mean(self, z: torch.Tensor) -> torch.Tensor:
        """Raw 5D features -> deterministic compact 5D latent (posterior mean).

        Stable-contract entry point for the world-model main path: always
        returns ``mu`` deterministically irrespective of ``self.training`` — a
        recursive ``host.train()`` cannot flip this frozen reducer into a
        stochastic mode. We run under a temporary ``eval()`` so that BOTH the
        Gaussian sampling AND any Transformer-block dropout (when a non-zero
        ``dropout`` config is used) are bypassed, then restore the prior mode.
        """
        was_training = self.training
        if was_training:
            self.eval()
        try:
            x, meta = self._to_seq(z)
            mu = self._encode_params(x).chunk(2, dim=-1)[0]
            return self._from_seq(mu, meta, self.latent_dim)
        finally:
            if was_training:
                self.train()

    def decode(self, z_latent: torch.Tensor) -> torch.Tensor:
        """Compact 5D latent -> reconstructed raw 5D features (de-standardised).

        Used only for standalone-training reconstruction monitoring; it is not
        on the world-model main path.
        """
        B, C, T, H, W = z_latent.shape
        x = z_latent.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)
        rec_std = self._decode_seq_std(x)
        rec_raw = self._destandardize(rec_std)
        return rec_raw.reshape(B, T, H, W, self.input_dim).permute(0, 4, 1, 2, 3).contiguous()

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Full reconstruction pass for training.

        Returns standardised-space ``recon``/``target`` (so the MSE objective
        is balanced across channels) plus ``mu``/``logvar`` for the KL term.
        All tensors are channel-first 5D.
        """
        x, meta = self._to_seq(z)
        z_l, mu, logvar = self._encode_seq(x)
        recon_std = self._decode_seq_std(z_l)
        target_std = self._standardize(x)
        return {
            "recon": self._from_seq(recon_std, meta, self.input_dim),
            "target": self._from_seq(target_std, meta, self.input_dim),
            "mu": self._from_seq(mu, meta, self.latent_dim),
            "logvar": self._from_seq(logvar, meta, self.latent_dim),
        }


def svae_loss(
    out: Dict[str, torch.Tensor],
    beta: float,
    free_bits_per_dim: float = 0.0,
    cos_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Reconstruction (MSE + cosine) + ``beta`` * KL loss.

    ``out`` is the dict returned by :meth:`SVAE.forward`. Reconstruction is in
    the standardised space; KL uses the closed-form diagonal-Gaussian form. The
    reference objective is plain mean KL (``free_bits_per_dim=0.0``); a positive
    ``free_bits_per_dim`` adds a per-dimension KL floor (an off-by-default rescue
    knob against posterior collapse, not part of the aligned default).
    """
    recon, target = out["recon"], out["target"]
    mu, logvar = out["mu"], out["logvar"]

    c_in = recon.shape[1]
    rec_f = recon.permute(0, 2, 3, 4, 1).reshape(-1, c_in)
    tgt_f = target.permute(0, 2, 3, 4, 1).reshape(-1, c_in)
    # accumulate both reconstruction terms in fp32 for bf16-autocast stability.
    mse = F.mse_loss(rec_f.float(), tgt_f.float())
    cos = 1.0 - F.cosine_similarity(rec_f.float(), tgt_f.float(), dim=-1).mean()

    c_lat = mu.shape[1]
    mu_f = mu.permute(0, 2, 3, 4, 1).reshape(-1, c_lat).float()
    lv_f = logvar.permute(0, 2, 3, 4, 1).reshape(-1, c_lat).float()
    # per-dim KL, averaged over tokens, then an optional free-bits floor per dim.
    kl_per_dim = (-0.5 * (1.0 + lv_f - mu_f.pow(2) - lv_f.exp())).mean(dim=0)
    if free_bits_per_dim > 0.0:
        kl_per_dim = kl_per_dim.clamp_min(free_bits_per_dim)
    kl = kl_per_dim.mean()

    loss = mse + cos_weight * cos + beta * kl
    stats = {
        "mse": mse.detach(),
        "cos": cos.detach(),
        "kl": kl.detach(),
        "loss": loss.detach(),
    }
    return loss, stats


def build_svae(config: Dict[str, Any]) -> SVAE:
    """Construct an :class:`SVAE` from a structural-config dict."""
    return SVAE(**config)


def load_svae(path: str, *, map_location: Union[str, torch.device] = "cpu") -> SVAE:
    """Load a standalone-trained S-VAE checkpoint into a frozen :class:`SVAE`.

    The checkpoint is the dict written by the training script:
    ``{"format_version", "model_config", "state_dict", ...}``.
    """
    ckpt = torch.load(path, map_location=map_location)
    if "model_config" not in ckpt or "state_dict" not in ckpt:
        raise ValueError(f"{path!r} is not a valid S-VAE checkpoint (need 'model_config' + 'state_dict').")
    # Validate the on-disk format version explicitly so an incompatible/older
    # checkpoint (or a wrong file that happens to carry the two keys above)
    # fails here with a clear message instead of deep inside load_state_dict.
    fmt = ckpt.get("format_version")
    if fmt != _CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"{path!r} has unsupported S-VAE checkpoint format_version={fmt!r} "
            f"(this build writes/reads version {_CHECKPOINT_FORMAT_VERSION})."
        )
    model = SVAE(**ckpt["model_config"])
    model.load_state_dict(ckpt["state_dict"])
    return model
