"""Client-side payload helpers and HTTP wrappers for the OpenWAM policy server.

Canonical location for code that downstream benchmark adapters / real-robot
integrations should import.

Client → server contract (must match ``PolicyServer._decode_obs``):

    {
      "images": {
        "head_camera":        <base64 JPEG>,       # required
        "left_wrist_camera":  <base64 JPEG>|null,  # optional
        "right_wrist_camera": <base64 JPEG>|null   # optional
      },
      "prompt": "<base task prompt>",
      "state":  [float, ...]                       # optional
    }

Server reads the checkpoint's ``config.yaml`` and handles all image preprocessing
(crop / resize / multi-view composition) and prompt wrapping internally, then
returns actions already denormalized to physical units.
"""

import base64
import json
from pathlib import Path
from typing import Optional
from urllib import error as _urlerror
from urllib import request


class ServerError(RuntimeError):
    """Structured 4xx / 5xx response from the OpenWAM policy server.

    Raised by ``post`` / ``get`` / ``reset`` when the server returns a
    non-2xx status. Carries the parsed error body so call-site logs and
    rollout harnesses can show the actual server-side message instead of
    a bare ``HTTPError: HTTP Error 400: Bad Request``.
    """

    def __init__(self, status: int, code: str = "", message: str = "", raw_body: str = ""):
        descriptor = f"[{status}] {code or 'http_error'}: {message or raw_body or '<empty body>'}"
        super().__init__(descriptor)
        self.status = status
        self.code = code
        self.message = message
        self.raw_body = raw_body


def _read_http_error(exc: _urlerror.HTTPError) -> "ServerError":
    """Decode a ``urllib.error.HTTPError`` into a ``ServerError``.

    The server's error responses follow
    ``{"type": "error", "code": ..., "message": ...}``. If the body isn't
    valid JSON we still preserve the raw text so the caller sees *something*.
    """
    try:
        raw = exc.read().decode("utf-8", errors="replace")
    except Exception:
        raw = ""
    try:
        body = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        body = {}
    return ServerError(
        status=exc.code,
        code=body.get("code", "") if isinstance(body, dict) else "",
        message=body.get("message", "") if isinstance(body, dict) else "",
        raw_body=raw,
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
    """Assemble a /predict payload from base64-encoded images.

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


def post(server: str, endpoint: str, payload: Optional[dict] = None, timeout: float = 300) -> dict:
    """POST JSON to server and return parsed response.

    Raises :class:`ServerError` on 4xx / 5xx so callers can log the server's
    own ``{code, message}`` instead of a bare ``HTTPError``. On success the
    behaviour is identical to the previous version (returns the parsed dict).
    """
    url = f"{server.rstrip('/')}{endpoint}"
    data = json.dumps(payload or {}).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except _urlerror.HTTPError as exc:
        raise _read_http_error(exc) from exc


def get(server: str, endpoint: str, timeout: float = 10) -> dict:
    """GET from server and return parsed response.

    Raises :class:`ServerError` on 4xx / 5xx (same contract as :func:`post`).
    """
    url = f"{server.rstrip('/')}{endpoint}"
    try:
        with request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except _urlerror.HTTPError as exc:
        raise _read_http_error(exc) from exc


def reset(server: str, timeout: float = 10) -> dict:
    """Clear the server's episode state.

    Drops ``obs_history``, the action buffer, and the ensemble buffer. Call
    this between episodes so the next ``predict`` starts from a clean slate.
    """
    return post(server, "/reset", {}, timeout=timeout)
