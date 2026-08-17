"""Payload helpers and the shared error model for the OpenWAM policy server.

Canonical location for code that downstream benchmark adapters / real-robot
integrations should import. The wire transport lives in
``benchmarks.utils.transport`` (WebSocket); this module owns only payload
construction and the structured ``ServerError`` both ends share.

Client → server obs message (validated server-side by ``ObsPreprocessor.preprocess``):

    {
      "images": {
        "head_camera":        <base64 PNG>,       # required
        "left_wrist_camera":  <base64 PNG>|null,  # optional
        "right_wrist_camera": <base64 PNG>|null   # optional
      },
      "prompt": "<prompt fed to the model verbatim>",
      "state":  [float, ...]                       # optional
    }

Server reads the checkpoint's ``config.yaml`` and performs final image
composition. Benchmark adapters whose training readers pre-resized individual
L-shape tiles first reproduce that resize with :func:`resize_for_lshape_slot`;
the others send native camera sizes. The server returns actions already
denormalized to physical units and forwards ``prompt`` verbatim.
"""

import base64
from pathlib import Path
from typing import Optional

# Canonical 384x320 L-shape slots used by the benchmark dataloaders that
# pre-resize each decoded camera with Pillow LANCZOS before composition.
L_SHAPE_HEAD_SIZE = (320, 256)  # (width, height)
L_SHAPE_WRIST_SIZE = (160, 128)


def resize_for_lshape_slot(image, slot: str):
    """Reproduce the training readers' pre-composition LANCZOS resize.

    This is intentionally an eval/client helper: the affected training readers
    already decode the head camera at 320x256 and wrists at 160x128. Sending
    those exact slot sizes makes the server's subsequent same-size paste a
    no-op geometrically, without changing any training code.
    """
    import numpy as np
    from PIL import Image

    sizes = {
        "head_camera": L_SHAPE_HEAD_SIZE,
        "left_wrist_camera": L_SHAPE_WRIST_SIZE,
        "right_wrist_camera": L_SHAPE_WRIST_SIZE,
    }
    if slot not in sizes:
        raise ValueError(f"unknown L-shape image slot {slot!r}; expected one of {tuple(sizes)}")
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"{slot} image must be HxWx3 RGB, got {value.shape}")
    value = np.ascontiguousarray(value.astype(np.uint8, copy=False))
    pil = Image.fromarray(value)
    target = sizes[slot]
    if pil.size != target:
        pil = pil.resize(target, Image.Resampling.LANCZOS)
    return np.asarray(pil, dtype=np.uint8)


class ServerError(RuntimeError):
    """Structured 4xx / 5xx response from the OpenWAM policy server.

    Carries the parsed error body so call-site logs and rollout harnesses can
    show the actual server-side message instead of a bare transport error.
    """

    def __init__(self, status: int, code: str = "", message: str = "", raw_body: str = ""):
        descriptor = f"[{status}] {code or 'http_error'}: {message or raw_body or '<empty body>'}"
        super().__init__(descriptor)
        self.status = status
        self.code = code
        self.message = message
        self.raw_body = raw_body


def server_error_from_body(status: int, body: dict, raw_body: str = "") -> "ServerError":
    """Build a ``ServerError`` from a parsed server error body.

    Single source for the ``{"type":"error","code","message"}`` shape the
    WebSocket transport maps to a status before raising.
    """
    info = body if isinstance(body, dict) else {}
    return ServerError(
        status=status,
        code=info.get("code", ""),
        message=info.get("message", ""),
        raw_body=raw_body,
    )


def encode_path_b64(path: str) -> str:
    """Read a JPEG/PNG file and return base64 of its raw bytes."""
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


def encode_numpy_b64(image) -> str:
    """Encode an H×W×3 RGB uint8 numpy array as lossless base64 PNG.

    **Do not resize on the client.** All crop / resize / multi-view
    composition normally happens server-side. Benchmarks whose training reader
    pre-resizes L-shape tiles must explicitly call :func:`resize_for_lshape_slot`
    first; RoboTwin deliberately sends its original camera sizes because its
    training reader also composes directly from the decoded originals.

    Lazy-imports Pillow so stdlib-only consumers of the other helpers in
    this module aren't forced to install it.
    """
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG", compress_level=1)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def build_payload(
    head: str,
    left_wrist: Optional[str] = None,
    right_wrist: Optional[str] = None,
    prompt: str = "",
    state: Optional[list] = None,
) -> dict:
    """Assemble an obs payload from base64-encoded images.

    ``head`` is the base64 string for ``head_camera`` (required).
    ``left_wrist`` / ``right_wrist`` may be None → server black-fills when
    multiview=True, or ignores when multiview=False.
    """
    payload = {
        "images": {
            "head_camera": head,
            "left_wrist_camera": left_wrist,
            "right_wrist_camera": right_wrist,
        },
        "prompt": prompt,
    }
    if state is not None:
        payload["state"] = list(state)
    return payload
