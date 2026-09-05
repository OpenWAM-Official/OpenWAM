"""Observation decoding / validation for the policy server.

Turns a client obs payload (base64 / bytes / PIL images + prompt + optional
state) into the composite PIL image and wrapped prompt the inference pipeline
expects. Engine-free and constructed from the resolved view config, so it can
be unit-tested without a GPU or a live engine.
"""

import base64
import io
import logging

import numpy as np

logger = logging.getLogger(__name__)


class ObsValidationError(ValueError):
    """Raised when a client observation payload does not match the server's
    view configuration (single-view vs multi-view, missing cameras, bad image
    encoding, etc.). The server turns it into a structured WebSocket
    ``{"type": "error", ...}`` response.
    """


def _cfg_select(cfg, path: str, default=None):
    cur = cfg
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            if part not in cur:
                return default
            cur = cur[part]
            continue
        try:
            cur = getattr(cur, part)
        except (AttributeError, KeyError):
            return default
    return cur


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


def _resolve_requires_proprio(cfg, engine) -> bool:
    cfg_value = _cfg_select(cfg, "model.architecture.use_proprioception", None)
    if cfg_value is None:
        cfg_value = _cfg_select(cfg, "model.params.use_proprioception", None)
    if cfg_value is not None:
        return _parse_bool(cfg_value)

    arch = getattr(engine, "architecture", None)
    if arch is not None:
        try:
            return bool(arch.uses_proprioception)
        except Exception:
            return bool(getattr(arch, "uses_proprioception", False))
    return False


class ObsPreprocessor:
    """Validate + preprocess a client obs payload against a fixed view config.

    Unified payload contract across single- and multi-view checkpoints: the
    client sends ``obs["images"]`` with fixed keys (``head_camera`` required,
    ``left_wrist_camera`` / ``right_wrist_camera`` optional and may be ``None``).

    - ``multiview=False``: use ``head_camera`` only, ``crop_and_resize`` to (W, H);
      wrist fields are ignored.
    - ``multiview=True``: black-fill missing/None wrists, then compose the L-shape
      layout keyed by ``camera_layout``.
    """

    def __init__(
        self,
        *,
        multiview: bool,
        camera_layout,
        img_height: int,
        img_width: int,
        requires_proprio: bool = False,
    ):
        self.multiview = bool(multiview)
        self.camera_layout = list(camera_layout)
        self.img_height = int(img_height)
        self.img_width = int(img_width)
        self.requires_proprio = bool(requires_proprio)

    @classmethod
    def from_cfg(cls, cfg, engine=None) -> "ObsPreprocessor":
        """Resolve the view config from the saved checkpoint cfg (+ engine fallback)."""
        from openwam.dataloader.transforms.multiview import DEFAULT_MULTIVIEW_CAMERA_LAYOUT

        dl = getattr(cfg, "dataloader", None)
        multiview = bool(getattr(dl, "multiview", False)) if dl is not None else False
        _layout = getattr(dl, "camera_layout", None) if dl is not None else None
        camera_layout = list(_layout) if _layout is not None else list(DEFAULT_MULTIVIEW_CAMERA_LAYOUT)
        # Output canvas size: prefer inference.{height,width}, fall back to dataloader.
        _inf = getattr(cfg, "inference", None)
        _h = getattr(_inf, "height", None) if _inf is not None else None
        _w = getattr(_inf, "width", None) if _inf is not None else None
        if _h is None and dl is not None:
            _h = getattr(dl, "height", 384)
        if _w is None and dl is not None:
            _w = getattr(dl, "width", 320)
        return cls(
            multiview=multiview,
            camera_layout=camera_layout,
            img_height=int(_h if _h is not None else 384),
            img_width=int(_w if _w is not None else 320),
            requires_proprio=_resolve_requires_proprio(cfg, engine),
        )

    def preprocess(self, obs: dict) -> dict:
        """Validate + preprocess ``obs`` in place.

        On success the returned obs always has:
            obs["image"]  -> PIL.Image sized (img_width, img_height)
            obs["prompt"] -> str, forwarded verbatim from the client (empty -> "")
            obs["state"]  -> np.ndarray (when a state was sent)

        Raises :class:`ObsValidationError` on malformed payload.
        """
        from PIL import Image

        from openwam.dataloader.transforms.multiview import (
            assemble_multiview_layout,
            crop_and_resize,
        )

        def _as_pil(x, *, ctx: str) -> Image.Image:
            if isinstance(x, Image.Image):
                return x if x.mode == "RGB" else x.convert("RGB")
            if isinstance(x, (bytes, bytearray)):
                try:
                    return Image.open(io.BytesIO(bytes(x))).convert("RGB")
                except Exception as e:
                    raise ObsValidationError(f"{ctx}: failed to decode raw image bytes ({e})")
            if isinstance(x, str):
                try:
                    raw = base64.b64decode(x)
                    return Image.open(io.BytesIO(raw)).convert("RGB")
                except Exception as e:
                    raise ObsValidationError(f"{ctx}: failed to decode base64 image ({e})")
            raise ObsValidationError(
                f"{ctx}: expected base64 image string, raw bytes, or PIL.Image, got {type(x).__name__}"
            )

        # --- Payload shape validation ---
        if "images" not in obs or not isinstance(obs.get("images"), dict):
            raise ObsValidationError(
                "client must send 'images' dict with head_camera key "
                "(left_wrist_camera / right_wrist_camera optional, may be null). "
                "The legacy single-field 'image' payload is no longer supported."
            )

        imgs = obs["images"]
        head_raw = imgs.get("head_camera")
        if head_raw is None:
            raise ObsValidationError(
                "head_camera is required in obs['images'] (got None or missing). "
                "The head camera feed is never optional on either single-view or multi-view servers."
            )
        head_pil = _as_pil(head_raw, ctx="images['head_camera']")

        # --- Dispatch by server's configured view mode ---
        if not self.multiview:
            if imgs.get("left_wrist_camera") is not None or imgs.get("right_wrist_camera") is not None:
                logger.info("[obs] single-view mode; ignoring wrist camera inputs.")
            obs["image"] = crop_and_resize(head_pil, self.img_height, self.img_width)
        else:
            if len(self.camera_layout) < 3:
                raise ObsValidationError(
                    f"multi-view server requires camera_layout with >= 3 entries; "
                    f"got {self.camera_layout}. Check the checkpoint's config.yaml."
                )

            def _decode_or_black(raw, ctx: str) -> Image.Image:
                if raw is None:
                    return Image.new("RGB", (self.img_width, self.img_height), (0, 0, 0))
                return _as_pil(raw, ctx=ctx)

            left_pil = _decode_or_black(imgs.get("left_wrist_camera"), ctx="images['left_wrist_camera']")
            right_pil = _decode_or_black(imgs.get("right_wrist_camera"), ctx="images['right_wrist_camera']")

            # Map the fixed client-side keys to camera_layout positions:
            #   head_camera        -> layout[0]  (top)
            #   left_wrist_camera  -> layout[1]  (bottom-left)
            #   right_wrist_camera -> layout[2]  (bottom-right)
            frames = {
                self.camera_layout[0]: head_pil,
                self.camera_layout[1]: left_pil,
                self.camera_layout[2]: right_pil,
            }
            obs["image"] = assemble_multiview_layout(
                frames,
                camera_layout=self.camera_layout,
                out_h=self.img_height,
                out_w=self.img_width,
            )

        # --- Prompt passthrough (server is prompt-agnostic; normalize missing to "") ---
        obs["prompt"] = obs.get("prompt", "") or ""

        # --- Proprio state passthrough ---
        # State-dim validation is intentionally NOT enforced: unify_action ckpts send RAW
        # proprio (pre-unify, width != state_dim) which the normalizer scatters to UNIFY_DIM,
        # so a fixed state_dim check would wrongly reject valid states. Tradeoff: a wrong-width
        # state on a non-unify ckpt is not caught here — it surfaces
        # downstream (Normalizer broadcast / model). Deliberate: accept proprio of any width.
        if "state" in obs and obs["state"] is not None:
            try:
                state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
            except (TypeError, ValueError) as exc:
                raise ObsValidationError(f"state must be a flat numeric list/array ({exc})") from exc
            obs["state"] = state
        elif self.requires_proprio:
            raise ObsValidationError(
                "this checkpoint requires obs['state']; send raw proprio state for proprio-conditioned checkpoints."
            )

        return obs
