"""Unit tests for the benchmarks.utils WebSocket transport client (no live server).

Covers the glue most prone to regression: error-body mapping, obs/ping framing,
the WS error status mapping, and the reconnect-once retry. Network is faked via
monkeypatch so these run anywhere.
"""

import json

import pytest

from benchmarks.utils import ServerError, WSPolicyClient, server_error_from_body


def test_server_error_from_body_maps_fields():
    err = server_error_from_body(500, {"code": "internal_error", "message": "boom"}, "raw")
    assert isinstance(err, ServerError)
    assert (err.status, err.code, err.message, err.raw_body) == (500, "internal_error", "boom", "raw")
    # Non-dict body degrades gracefully.
    err2 = server_error_from_body(400, None, "x")
    assert (err2.status, err2.code, err2.message) == (400, "", "")


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
    client = WSPolicyClient("ws://h:8848")
    fake = _FakeWS(response)
    monkeypatch.setattr(client, "_connect", lambda: setattr(client, "_ws", fake))
    return client, fake


def test_ws_predict_wraps_obs_and_returns_action(monkeypatch):
    client, fake = _ws_client_with(monkeypatch, json.dumps({"type": "action", "action": [1, 2], "latency_ms": 5}))
    out = client.predict({"images": {}, "prompt": "p"})
    assert out["action"] == [1, 2]
    sent = json.loads(fake.sent[0])
    assert sent["type"] == "obs" and sent["prompt"] == "p"  # type added, payload preserved


def test_ws_predict_once_wraps_obs_without_reconnect(monkeypatch):
    client, fake = _ws_client_with(
        monkeypatch,
        json.dumps({"type": "action", "action": [3, 4]}),
    )

    out = client.predict_once({"images": {}, "prompt": "once"})

    assert out["action"] == [3, 4]
    assert json.loads(fake.sent[0]) == {
        "type": "obs",
        "images": {},
        "prompt": "once",
    }


def test_ws_predict_once_never_resends_after_ambiguous_drop(monkeypatch):
    dropping = _FakeWS("")
    dropping.send = lambda msg: (_ for _ in ()).throw(OSError("ambiguous drop"))
    healthy = _FakeWS(json.dumps({"type": "action", "action": [9]}))
    conns = iter([dropping, healthy])
    connects = {"n": 0}
    client = WSPolicyClient("ws://h:8848")

    def _fake_connect():
        if client._ws is None:
            connects["n"] += 1
            client._ws = next(conns)

    monkeypatch.setattr(client, "_connect", _fake_connect)

    with pytest.raises(OSError, match="ambiguous drop"):
        client.predict_once({})
    assert connects["n"] == 1
    assert healthy.sent == []


def test_ws_reset_and_ping(monkeypatch):
    client, fake = _ws_client_with(monkeypatch, json.dumps({"type": "reset_ack"}))
    assert client.reset() == {"type": "reset_ack"}
    assert json.loads(fake.sent[0])["type"] == "reset"

    client2, fake2 = _ws_client_with(monkeypatch, json.dumps({"type": "pong"}))
    assert client2.ping() == {"type": "pong"}
    assert json.loads(fake2.sent[0])["type"] == "ping"


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


def test_ws_reconnect_once_on_drop(monkeypatch):
    """A socket closed server-side between calls is reconnected exactly once."""
    dropping = _FakeWS(json.dumps({"type": "action", "action": [0]}))
    dropping.send = lambda msg: (_ for _ in ()).throw(OSError("socket closed"))
    healthy = _FakeWS(json.dumps({"type": "action", "action": [9]}))
    conns = iter([dropping, healthy])

    client = WSPolicyClient("ws://h:8848")

    def _fake_connect():
        if client._ws is None:
            client._ws = next(conns)

    monkeypatch.setattr(client, "_connect", _fake_connect)
    assert client.predict({})["action"] == [9]  # first conn dropped, retried on the second


def test_ws_reraises_when_retry_also_drops(monkeypatch):
    client = WSPolicyClient("ws://h:8848")

    def _always_dropping():
        if client._ws is None:
            ws = _FakeWS("")
            ws.send = lambda msg: (_ for _ in ()).throw(OSError("down"))
            client._ws = ws

    monkeypatch.setattr(client, "_connect", _always_dropping)
    with pytest.raises(OSError):
        client.predict({})


def test_ws_ping_does_not_reconnect(monkeypatch):
    """ping is a liveness probe: one connect attempt, never resends on drop."""
    connects = {"n": 0}
    client = WSPolicyClient("ws://h:8848")

    def _fake_connect():
        if client._ws is None:
            connects["n"] += 1
            ws = _FakeWS("")
            ws.send = lambda msg: (_ for _ in ()).throw(OSError("down"))
            client._ws = ws

    monkeypatch.setattr(client, "_connect", _fake_connect)
    with pytest.raises(OSError):
        client.ping()
    assert connects["n"] == 1  # failed fast, no reconnect
