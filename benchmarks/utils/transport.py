"""Transport clients for the OpenWAM policy server.

Two interchangeable transports behind one ``PolicyClient`` interface so
callers (benchmarks, real-robot loops, the continuous-test harness) can swap
HTTP for WebSocket without touching payload code:

  * ``http`` — http.client over one reused keep-alive connection (reconnect-once on drop).
  * ``ws``   — websockets.sync, one persistent connection.

Payload construction, encoding and the ``ServerError`` model live in
``benchmarks.utils.client``; this module owns only the wire transport.
"""

import http.client
import json
from typing import Optional
from urllib.parse import urlparse

from benchmarks.utils.client import server_error_from_body

TRANSPORTS = ("http", "ws")


class PolicyClient:
    """Common interface + lifecycle for every transport.

    Subclasses implement ``predict`` / ``reset``. ``close`` and the context
    manager are shared so callers can always ``with make_policy_client(...)``.
    """

    label = "policy"

    def predict(self, payload: dict) -> dict:
        raise NotImplementedError

    def reset(self) -> dict:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self) -> "PolicyClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class HTTPPolicyClient(PolicyClient):
    """HTTP transport — keep-alive connection reused, reconnect-once on drop."""

    label = "http"

    def __init__(self, http_url: str, timeout: float = 300.0):
        self.http_url = http_url.rstrip("/")
        self.timeout = timeout
        parsed = urlparse(self.http_url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"http_url must be http:// or https://, got '{http_url}'")
        self._https = parsed.scheme == "https"
        self._host = parsed.hostname
        self._port = parsed.port or (443 if self._https else 80)
        self._conn: Optional[http.client.HTTPConnection] = None

    def _new_conn(self) -> http.client.HTTPConnection:
        cls = http.client.HTTPSConnection if self._https else http.client.HTTPConnection
        return cls(self._host, self._port, timeout=self.timeout)

    def predict(self, payload: dict) -> dict:
        return self._request("POST", "/predict", payload)

    def reset(self) -> dict:
        return self._request("POST", "/reset", {})

    def health(self) -> dict:
        return self._request("GET", "/health", None)

    def _request(self, method: str, endpoint: str, payload: Optional[dict]) -> dict:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"} if body is not None else {}
        # One retry: a kept-alive socket can be closed server-side between calls.
        for attempt in (0, 1):
            if self._conn is None:
                self._conn = self._new_conn()
            try:
                self._conn.request(method, endpoint, body=body, headers=headers)
                resp = self._conn.getresponse()
                raw = resp.read().decode("utf-8")
                if resp.status >= 400:
                    try:
                        parsed_body = json.loads(raw) if raw else {}
                    except json.JSONDecodeError:
                        parsed_body = {}
                    raise server_error_from_body(resp.status, parsed_body, raw)
                return json.loads(raw)
            except (http.client.HTTPException, ConnectionError, OSError):
                self.close()
                if attempt == 1:
                    raise
        raise RuntimeError("unreachable: HTTP retry loop exhausted")  # for type-checkers

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


class WSPolicyClient(PolicyClient):
    """websockets.sync transport — one persistent connection for obs/reset."""

    label = "ws"

    def __init__(self, ws_url: str, timeout: float = 300.0, compression: Optional[str] = None):
        self.ws_url = ws_url
        self.timeout = timeout
        # None disables permessage-deflate so we compare raw transport vs HTTP.
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
        self._ws = connect(self.ws_url, max_size=None, compression=self.compression, open_timeout=self.timeout)

    def _roundtrip(self, message: dict) -> dict:
        self._connect()
        self._ws.send(json.dumps(message))
        try:
            raw = self._ws.recv(timeout=self.timeout)
        except TypeError:
            raw = self._ws.recv()  # websockets < 13 sync recv has no timeout kwarg
        data = json.loads(raw)
        if isinstance(data, dict) and data.get("type") == "error":
            # Mirror the server's status split: internal_error -> 500, else 400.
            status = 500 if data.get("code") == "internal_error" else 400
            raise server_error_from_body(status, data, raw)
        return data

    def predict(self, payload: dict) -> dict:
        return self._roundtrip({**payload, "type": "obs"})

    def reset(self) -> dict:
        return self._roundtrip({"type": "reset"})

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            finally:
                self._ws = None


def make_policy_client(
    transport: str,
    *,
    http_url: str,
    ws_url: str,
    timeout: float = 300.0,
) -> PolicyClient:
    """Build the client for a transport name: ``http`` | ``ws``."""
    if transport == "http":
        return HTTPPolicyClient(http_url, timeout=timeout)
    if transport == "ws":
        return WSPolicyClient(ws_url, timeout=timeout)
    raise ValueError(f"Unknown transport '{transport}'. Choose from: {', '.join(TRANSPORTS)}")
