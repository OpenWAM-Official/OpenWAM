"""Unit tests for the benchmarks.utils transport clients (no live server).

Covers the glue most prone to regression: error-body mapping, the transport
factory, HTTP(S) URL parsing, the keep-alive reconnect-once retry, and the WS
error status mapping. Network is faked via monkeypatch so these run anywhere.
"""

import http.client
import json

import pytest

from benchmarks.utils import (
    HTTPPolicyClient,
    ServerError,
    WSPolicyClient,
    make_policy_client,
    server_error_from_body,
)


def test_server_error_from_body_maps_fields():
    err = server_error_from_body(500, {"code": "internal_error", "message": "boom"}, "raw")
    assert isinstance(err, ServerError)
    assert (err.status, err.code, err.message, err.raw_body) == (500, "internal_error", "boom", "raw")
    # Non-dict body degrades gracefully.
    err2 = server_error_from_body(400, None, "x")
    assert (err2.status, err2.code, err2.message) == (400, "", "")


def test_make_policy_client_selects_type():
    assert isinstance(make_policy_client("http", http_url="http://h:8848", ws_url="ws://h:8850"), HTTPPolicyClient)
    assert isinstance(make_policy_client("ws", http_url="http://h:8848", ws_url="ws://h:8850"), WSPolicyClient)
    with pytest.raises(ValueError):
        make_policy_client("nope", http_url="http://h", ws_url="ws://h")


def test_http_url_parsing():
    assert (HTTPPolicyClient("http://example.com")._host, HTTPPolicyClient("http://example.com")._port) == (
        "example.com",
        80,
    )
    https = HTTPPolicyClient("https://example.com")
    assert https._port == 443 and https._https is True
    custom = HTTPPolicyClient("https://example.com:9000/")
    assert custom._port == 9000 and custom._https is True
    with pytest.raises(ValueError):
        HTTPPolicyClient("ftp://example.com")


def test_http_new_conn_class():
    assert isinstance(HTTPPolicyClient("http://h")._new_conn(), http.client.HTTPConnection)
    assert isinstance(HTTPPolicyClient("https://h")._new_conn(), http.client.HTTPSConnection)


class _FakeResp:
    def __init__(self, status, body):
        self.status = status
        self._body = body.encode("utf-8")

    def read(self):
        return self._body


class _FakeConn:
    """Scripted fake HTTP connection: one step per ``request`` call."""

    def __init__(self, script):
        self.script = list(script)
        self.i = 0

    def request(self, *a, **k):
        step = self.script[self.i]
        self.i += 1
        if step[0] == "raise":
            raise step[1]

    def getresponse(self):
        step = self.script[self.i - 1]
        return _FakeResp(step[1], step[2])

    def close(self):
        pass


def _http_client_with(monkeypatch, conn_scripts):
    """Build an HTTPPolicyClient whose ``_new_conn`` yields scripted fakes."""
    client = HTTPPolicyClient("http://h:8848")
    scripts = iter(conn_scripts)
    conns = []

    def _fake_new_conn():
        conn = _FakeConn(next(scripts))
        conns.append(conn)
        return conn

    monkeypatch.setattr(client, "_new_conn", _fake_new_conn)
    return client, conns


def test_http_request_success(monkeypatch):
    client, conns = _http_client_with(monkeypatch, [[("ok", 200, json.dumps({"action": [1, 2]}))]])
    assert client.predict({"x": 1}) == {"action": [1, 2]}
    assert len(conns) == 1


def test_http_request_retries_once_on_drop(monkeypatch):
    client, conns = _http_client_with(
        monkeypatch,
        [[("raise", ConnectionResetError())], [("ok", 200, json.dumps({"ok": True}))]],
    )
    assert client.reset() == {"ok": True}
    assert len(conns) == 2  # reconnected exactly once


def test_http_4xx_raises_without_retry(monkeypatch):
    body = json.dumps({"type": "error", "code": "obs_validation_error", "message": "bad"})
    client, conns = _http_client_with(monkeypatch, [[("ok", 400, body)]])
    with pytest.raises(ServerError) as ei:
        client.predict({})
    assert ei.value.status == 400 and ei.value.code == "obs_validation_error"
    assert len(conns) == 1  # a server 4xx is a real response, not a dropped socket


class _FakeWS:
    def __init__(self, response):
        self._response = response
        self.sent = []

    def send(self, msg):
        self.sent.append(msg)

    def recv(self, timeout=None):
        return self._response

    def close(self):
        pass


def _ws_client_with(monkeypatch, response):
    client = WSPolicyClient("ws://h:8850")
    fake = _FakeWS(response)
    monkeypatch.setattr(client, "_connect", lambda: setattr(client, "_ws", fake))
    return client, fake


def test_ws_predict_wraps_obs_and_returns_action(monkeypatch):
    client, fake = _ws_client_with(monkeypatch, json.dumps({"type": "action", "action": [1, 2], "latency_ms": 5}))
    out = client.predict({"images": {}, "prompt": "p"})
    assert out["action"] == [1, 2]
    sent = json.loads(fake.sent[0])
    assert sent["type"] == "obs" and sent["prompt"] == "p"  # type added, payload preserved


def test_ws_error_status_mapping(monkeypatch):
    client, _ = _ws_client_with(monkeypatch, json.dumps({"type": "error", "code": "internal_error", "message": "x"}))
    with pytest.raises(ServerError) as ei:
        client.predict({})
    assert ei.value.status == 500  # internal_error -> 500

    client2, _ = _ws_client_with(
        monkeypatch, json.dumps({"type": "error", "code": "obs_validation_error", "message": "x"})
    )
    with pytest.raises(ServerError) as ei2:
        client2.predict({})
    assert ei2.value.status == 400  # everything else -> 400
