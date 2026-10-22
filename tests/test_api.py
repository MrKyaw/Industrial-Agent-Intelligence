"""Tests for the HTTP API and dashboard."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from iai.api import ApiConfig, ServerHandle
from iai.app import Application


@pytest.fixture
def server(app):
    config = ApiConfig(host="127.0.0.1", port=0, rate_limit_per_minute=1000)
    with ServerHandle(app, config) as handle:
        yield handle


def test_healthz(server):
    with urllib.request.urlopen(server.url + "/healthz") as response:
        payload = json.loads(response.read())
    assert payload["status"] == "ok"


def test_meta_and_tools(server):
    with urllib.request.urlopen(server.url + "/v1/meta") as response:
        meta = json.loads(response.read())
    assert "tools" in meta
    with urllib.request.urlopen(server.url + "/v1/tools") as response:
        tools = json.loads(response.read())
    assert len(tools["tools"]) >= 1


def test_ask(server):
    body = json.dumps({"question": "How efficient is Chiller 1 right now?", "site_id": "site.dc-lon-01", "role": "operator"}).encode()
    request = urllib.request.Request(server.url + "/v1/ask", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        payload = json.loads(response.read())
    assert payload["status"] == "answered"
    assert payload["guardrails"]["passed"] is True


def test_ask_validation(server):
    body = json.dumps({"question": "x", "site_id": "nope", "role": "operator"}).encode()
    request = urllib.request.Request(server.url + "/v1/ask", data=body, headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(request)
    assert exc_info.value.code == 400


def test_dashboard(server):
    with urllib.request.urlopen(server.url + "/") as response:
        html = response.read().decode()
    assert "Industrial Agent Intelligence" in html


def test_token_required():
    from iai.app import Application

    app = Application.load()
    config = ApiConfig(host="127.0.0.1", port=0, token="secret")
    with ServerHandle(app, config) as handle:
        request = urllib.request.Request(handle.url + "/v1/meta")
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(request)
        assert exc_info.value.code == 401
        authed = urllib.request.Request(handle.url + "/v1/meta", headers={"Authorization": "Bearer secret"})
        with urllib.request.urlopen(authed) as response:
            assert response.status == 200


def test_rate_limit():
    app = Application.load()
    config = ApiConfig(host="127.0.0.1", port=0, rate_limit_per_minute=1)
    with ServerHandle(app, config) as handle:
        body = json.dumps({"question": "How efficient is Chiller 1 right now?", "site_id": "site.dc-lon-01", "role": "operator"}).encode()
        first = urllib.request.Request(handle.url + "/v1/ask", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(first):
            pass
        second = urllib.request.Request(handle.url + "/v1/ask", data=body, headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(second)
        assert exc_info.value.code == 429
