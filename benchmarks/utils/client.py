"""Payload helpers and the shared error model for the OpenWAM policy server.

Canonical location for code that downstream benchmark adapters / real-robot
integrations should import. The wire transport lives in
``benchmarks.utils.transport`` (WebSocket); this module owns only payload
construction and the structured ``ServerError`` both ends share.

Client → server obs message (validated server-side by ``ObsPreprocessor.preprocess``):

    {
      "images": {
        "head_camera":        <base64 JPEG>,       # required
        "left_wrist_camera":  <base64 JPEG>|null,  # optional
        "right_wrist_camera": <base64 JPEG>|null   # optional
      },
      "prompt": "<prompt fed to the model verbatim>",
      "state":  [float, ...]                       # optional
    }

Server reads the checkpoint's ``config.yaml`` and handles all image preprocessing
(crop / resize / multi-view composition) internally, then returns actions already
denormalized to physical units. The server is prompt-agnostic: it forwards the
``prompt`` field to the model verbatim, so the caller sends the exact prompt the
model should see (each benchmark owns its own prompt template).
"""

import base64
from pathlib import Path
from typing import Optional


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
    """Encode an H×W×3 RGB uint8 numpy array as base64 JPEG at source resolution.

    **Do not resize on the client.** All crop / resize / multi-view
    composition happens server-side using the canvas size and interpolation
    (Pillow LANCZOS for single-view, BILINEAR for the L-shape layout) that
    the checkpoint was trained with. Any client-side resize would layer a
    second, interpolation-mismatched step on top of that, diverging from
    training-time preprocessing.

    Lazy-imports Pillow so stdlib-only consumers of the other helpers in
    this module aren't forced to install it.
    """
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="JPEG")
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
    ``left_wrist`` / ``right_wrist`` may be None: the server ignores them when
    multiview=False, but rejects a None wrist with an obs-validation error when
    multiview=True (a multi-view checkpoint needs both wrist frames — see
    ``ObsPreprocessor``).
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
