"""WebSocket transport client for the OpenWAM policy server.

One persistent ``websockets.sync`` connection carries obs / reset / ping. A
dropped socket is reconnected once for obs / reset; ``ping`` (liveness) fails
fast. Payload construction and the ``ServerError`` model live in
``benchmarks.utils.client``; this module owns only the wire transport.
"""

import json
from typing import Optional

from benchmarks.utils.client import server_error_from_body
from openwam import ws_protocol as wsp

# Errors that mean "the socket dropped" — reconnect once and retry. A live
# ``ServerError`` is raised later from the parsed body, never from here.
try:
    from websockets.exceptions import ConnectionClosed as _ConnectionClosed

    _DROP_ERRORS: tuple = (OSError, _ConnectionClosed)
except ImportError:
    _DROP_ERRORS = (OSError,)


class WSPolicyClient:
    """websockets.sync transport — one persistent connection.

    ``timeout`` bounds a message round-trip (inference can take many seconds);
    ``open_timeout`` bounds connection setup and stays short so a liveness
    ``ping`` against an unreachable server fails fast instead of blocking for
    the full inference timeout.
    """

    label = "ws"

    def __init__(
        self,
        ws_url: str,
        timeout: float = 300.0,
        open_timeout: float = 10.0,
        compression: Optional[str] = None,
    ):
        self.ws_url = ws_url
        self.timeout = timeout
        self.open_timeout = open_timeout
        # None disables permessage-deflate so payloads go over the wire raw.
        self.compression = compression
        self._ws = None

    def _connect(self) -> None:
        if self._ws is not None:
            return
        try:
            from websockets.sync.client import connect
        except ImportError as exc:
            raise ImportError(
                "WebSocket transport needs websockets>=12 (websockets.sync). Install with: pip install -U websockets"
            ) from exc
        # max_size=None: multi-camera base64 obs routinely exceed the 1 MB default.
        self._ws = connect(self.ws_url, max_size=None, compression=self.compression, open_timeout=self.open_timeout)

    def _roundtrip(self, message: dict, *, reconnect: bool = True) -> dict:
        body = json.dumps(message)
        # A persistent socket can be closed server-side between calls. predict /
        # reset reconnect once and resend; ping is a liveness probe — it fails
        # fast and never resends (resending an obs would re-run inference).
        attempts = (0, 1) if reconnect else (0,)
        for attempt in attempts:
            self._connect()
            try:
                self._ws.send(body)
                try:
                    raw = self._ws.recv(timeout=self.timeout)
                except TypeError:
                    raw = self._ws.recv()  # websockets < 13 sync recv has no timeout kwarg
                break
            except _DROP_ERRORS:
                self.close()
                if attempt == attempts[-1]:
                    raise
        data = json.loads(raw)
        if isinstance(data, dict) and data.get("type") == wsp.ERROR:
            # Mirror the server's status split: internal_error -> 500, else 400.
            status = 500 if data.get("code") == wsp.ERR_INTERNAL else 400
            raise server_error_from_body(status, data, raw)
        return data

    def predict(self, payload: dict) -> dict:
        return self._roundtrip({**payload, "type": wsp.OBS})

    def reset(self) -> dict:
        return self._roundtrip({"type": wsp.RESET})

    def ping(self) -> dict:
        return self._roundtrip({"type": wsp.PING}, reconnect=False)

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            finally:
                self._ws = None

    def __enter__(self) -> "WSPolicyClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
