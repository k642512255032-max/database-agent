"""Hermetic tests for dashboard/netlify.py: requests is monkeypatched, nothing leaves the machine."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard import netlify  # noqa: E402
from dashboard.netlify import API, NetlifyClient, PublishError, publish, site_name  # noqa: E402

TOKEN = "nfp_super-secret-token"


class FakeResponse:
    def __init__(self, status: int, body):
        self.status_code, self._body = status, body
        self.text = str(body)

    def json(self):
        return self._body


class FakeNetlify:
    """Records calls and plays a scripted deploy state sequence."""

    def __init__(self, states=("processing", "ready"), create_status=201):
        self.calls: list[tuple[str, str, dict]] = []
        self.states = list(states)
        self.create_status = create_status

    def post(self, url, **kw):
        self.calls.append(("POST", url, kw))
        assert kw["headers"]["Authorization"] == f"Bearer {TOKEN}"
        if url == f"{API}/sites":
            return FakeResponse(self.create_status, {"id": "site-1", "ssl_url": "https://x-abc.netlify.app"})
        assert url.endswith("/deploys") and kw["headers"]["Content-Type"] == "application/zip"
        assert kw["data"] == b"ZIP"
        return FakeResponse(200, {"id": "dep-1", "state": "uploading"})

    def get(self, url, **kw):
        self.calls.append(("GET", url, kw))
        state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        return FakeResponse(200, {"id": "dep-1", "state": state, "ssl_url": "https://x-abc.netlify.app",
                                  "error_message": "build exploded" if state == "error" else None})

    def delete(self, url, **kw):
        self.calls.append(("DELETE", url, kw))
        return FakeResponse(204, "")


@pytest.fixture
def fake(monkeypatch):
    f = FakeNetlify()
    monkeypatch.setattr(requests, "post", f.post)
    monkeypatch.setattr(requests, "get", f.get)
    monkeypatch.setattr(requests, "delete", f.delete)
    monkeypatch.setattr(netlify.time, "sleep", lambda s: None)
    return f


def test_publish_first_time_creates_site_then_deploys(fake):
    info = publish(NetlifyClient(TOKEN), b"ZIP", None, "sales-overview")
    assert info == {"site_id": "site-1", "url": "https://x-abc.netlify.app", "deploy_id": "dep-1"}
    methods = [(m, u.replace(API, "")) for m, u, _ in fake.calls]
    assert methods[:2] == [("POST", "/sites"), ("POST", "/sites/site-1/deploys")]
    assert all(m == "GET" and u == "/deploys/dep-1" for m, u in methods[2:]) and len(methods) == 4
    name = fake.calls[0][2]["json"]["name"]
    assert name.startswith("sales-overview-") and len(name) == len("sales-overview-") + 6


def test_publish_again_reuses_site(fake):
    info = publish(NetlifyClient(TOKEN), b"ZIP", "site-1", "sales-overview")
    assert info["site_id"] == "site-1"
    assert not any(u == f"{API}/sites" for _, u, _ in fake.calls)


def test_deploy_error_state_raises(fake):
    fake.states = ["error"]
    with pytest.raises(PublishError, match="build exploded"):
        publish(NetlifyClient(TOKEN), b"ZIP", "site-1", "x")


def test_wait_ready_times_out(fake, monkeypatch):
    fake.states = ["processing"]
    clock = iter([0, 1, 200, 201])
    monkeypatch.setattr(netlify.time, "monotonic", lambda: next(clock))
    with pytest.raises(PublishError, match="did not become ready"):
        NetlifyClient(TOKEN).wait_ready("dep-1", timeout_s=100)


def test_http_error_message_has_status_but_not_token(fake):
    fake.create_status = 401
    with pytest.raises(PublishError) as exc:
        NetlifyClient(TOKEN).create_site("x")
    assert str(exc.value).startswith("Netlify 401:") and TOKEN not in str(exc.value)


def test_network_error_is_publish_error(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("no route")
    monkeypatch.setattr(requests, "post", boom)
    with pytest.raises(PublishError, match="Netlify request failed"):
        NetlifyClient(TOKEN).create_site("x")


def test_delete_site(fake):
    NetlifyClient(TOKEN).delete_site("site-1")
    assert fake.calls[-1][:2] == ("DELETE", f"{API}/sites/site-1")


def test_site_name_suffix_and_fallback():
    assert site_name("").startswith("dashboard-")
    assert len(site_name("a" * 60).split("-")[0]) == 40
