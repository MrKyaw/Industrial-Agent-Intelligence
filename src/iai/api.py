"""Read-mostly HTTP API and self-contained dashboard.

The API is intentionally small: health, metadata, tool catalogue, ask, and the
latest eval scorecard. It uses the standard library only and refuses to start on a
non-loopback address without an explicit bearer token.
"""

from __future__ import annotations

import hmac
import json
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .app import Application
from .evals import EvalReport
from .report import render_html
from .util import format_timestamp, now_utc

MAX_BODY_BYTES = 64 * 1024


class ApiError(ValueError):
    """Raised for invalid API configuration."""


@dataclass
class ApiConfig:
    host: str = "127.0.0.1"
    port: int = 8080
    token: str | None = None
    max_body_bytes: int = MAX_BODY_BYTES
    rate_limit_per_minute: int = 120
    verbose: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.port <= 65535:
            raise ApiError("port must be within [0, 65535]")
        if self.max_body_bytes < 128:
            raise ApiError("max_body_bytes must be at least 128")
        if self.rate_limit_per_minute <= 0:
            raise ApiError("rate_limit_per_minute must be positive")
        if self.host not in {"127.0.0.1", "localhost", "::1"} and not self.token:
            raise ApiError("a bearer token is required when binding to a non-loopback address")


class RateLimiter:
    def __init__(self, limit: int, *, window_s: float = 60.0) -> None:
        if limit <= 0 or window_s <= 0:
            raise ApiError("rate limiter requires positive limit and window")
        self.limit = limit
        self.window_s = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> tuple[bool, int]:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            cutoff = now - self.window_s
            while hits and hits[0] < cutoff:
                hits.popleft()
            if len(hits) >= self.limit:
                return False, 0
            hits.append(now)
            return True, self.limit - len(hits)


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


class IAIRequestHandler(BaseHTTPRequestHandler):
    server_version = "IndustrialAgentIntelligence/1.0"
    protocol_version = "HTTP/1.1"

    @property
    def app(self) -> Application:
        return self.server.app  # type: ignore[attr-defined]

    @property
    def config(self) -> ApiConfig:
        return self.server.config  # type: ignore[attr-defined]

    @property
    def limiter(self) -> RateLimiter:
        return self.server.limiter  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        if self.config.verbose:
            super().log_message(format, *args)

    # ------------------------------------------------------------------
    # HTTP methods
    # ------------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if not self._rate_limited():
            return
        if parsed.path == "/healthz":
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "agent_version": self.app.config.agent_version,
                    "as_of": format_timestamp(self.app.as_of),
                },
            )
            return
        if not self._authorized():
            return
        if parsed.path == "/":
            self._send_html(HTTPStatus.OK, self._dashboard())
            return
        if parsed.path == "/v1/meta":
            self._send_json(HTTPStatus.OK, self.app.status())
            return
        if parsed.path == "/v1/tools":
            role = parse_qs(parsed.query).get("role", ["viewer"])[0]
            try:
                payload = self.app.tool_manifest(role)
            except Exception as exc:  # noqa: BLE001
                self._send_error(HTTPStatus.BAD_REQUEST, "invalid_role", str(exc))
                return
            self._send_json(HTTPStatus.OK, payload)
            return
        if parsed.path == "/v1/evals/latest":
            report_path = self._report_path()
            if report_path is None:
                self._send_error(HTTPStatus.NOT_FOUND, "no_report", "No evaluation report has been generated.")
                return
            try:
                payload = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "bad_report", "The report could not be read.")
                return
            self._send_json(HTTPStatus.OK, payload)
            return
        self._send_error(HTTPStatus.NOT_FOUND, "not_found", "Unknown endpoint.")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if not self._rate_limited():
            return
        if not self._authorized():
            return
        if parsed.path != "/v1/ask":
            self._send_error(HTTPStatus.NOT_FOUND, "not_found", "Unknown endpoint.")
            return
        try:
            payload = self._read_json()
        except ApiError as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_body", str(exc))
            return
        question = payload.get("question")
        site_id = payload.get("site_id")
        role = payload.get("role", "operator")
        if not isinstance(question, str) or not question.strip():
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_question", "question must be a non-empty string.")
            return
        if len(question) > 2000:
            self._send_error(HTTPStatus.BAD_REQUEST, "question_too_long", "question must be at most 2000 characters.")
            return
        if not isinstance(site_id, str) or site_id not in self.app.site_ids():
            self._send_error(HTTPStatus.BAD_REQUEST, "unknown_site", "site_id must be a known site.")
            return
        if not isinstance(role, str):
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_role", "role must be a string.")
            return
        context = payload.get("context", {})
        if not isinstance(context, dict):
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_context", "context must be an object.")
            return
        from .models import AgentRequest

        try:
            request = AgentRequest(
                question=question,
                site_id=site_id,
                role=role,
                intent=str(payload.get("intent", "auto")),
                context=context,
            )
            if self.app.runtime is None:
                raise RuntimeError("runtime not ready")
            answer = self.app.runtime.run(request)
        except (ValueError, RuntimeError) as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, "agent_error", str(exc))
            return
        except Exception:  # noqa: BLE001 - never leak a traceback
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error", "The agent failed.")
            return
        self._send_json(HTTPStatus.OK, answer.to_dict())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rate_limited(self) -> bool:
        key = self.client_address[0] if self.client_address else "unknown"
        allowed, remaining = self.limiter.allow(key)
        if not allowed:
            self._send_error(HTTPStatus.TOO_MANY_REQUESTS, "rate_limited", "Too many requests.")
            return False
        self._rate_limit_remaining = remaining
        return True

    def _authorized(self) -> bool:
        if not self.config.token:
            return True
        header = self.headers.get("Authorization", "")
        expected = f"Bearer {self.config.token}"
        if not hmac.compare_digest(header, expected):
            self._send_error(HTTPStatus.UNAUTHORIZED, "unauthorized", "A valid bearer token is required.")
            return False
        return True

    def _read_json(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ApiError("Content-Length is required")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ApiError("Content-Length must be an integer") from exc
        if length < 0:
            raise ApiError("Content-Length must be non-negative")
        if length > self.config.max_body_bytes:
            raise ApiError(f"body exceeds {self.config.max_body_bytes} bytes")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ApiError(f"invalid JSON: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ApiError("JSON body must be an object")
        return payload

    def _report_path(self) -> Path | None:
        candidate = self.app.data_dir.parent / "reports" / "evals" / "latest.json"
        return candidate if candidate.exists() else None

    def _dashboard(self) -> str:
        report_path = self._report_path()
        if report_path is not None:
            try:
                report = EvalReport.from_dict(json.loads(report_path.read_text(encoding="utf-8")))
                return render_html(report)
            except (OSError, ValueError, KeyError):
                pass
        status = self.app.status()
        return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Industrial Agent Intelligence</title>
<style>
body{{margin:0;background:#0b1220;color:#e8eefc;font:16px/1.6 ui-sans-serif,system-ui,sans-serif}}
main{{max-width:820px;margin:0 auto;padding:56px 24px}}code{{color:#5eead4}}
.card{{background:#111b2e;border:1px solid #243552;border-radius:14px;padding:22px;margin:18px 0}}
h1{{letter-spacing:-.02em}}.muted{{color:#93a4c3}}
</style></head><body><main>
<h1>Industrial Agent Intelligence</h1>
<p class="muted">Deterministic reasoning, tool policy, guardrails, and behavioural evals for industrial agents.</p>
<div class="card"><h2>Runtime online</h2>
<p>Dataset <code>{status['dataset']}</code> · as of <code>{status['as_of']}</code> · {status['tools']} tools ·
{status['knowledge_documents']} knowledge documents.</p>
<p class="muted">No scorecard has been generated yet. Run <code>python -m iai evals run</code>.</p></div>
<div class="card"><h2>API</h2>
<p><code>GET /healthz</code> · <code>GET /v1/meta</code> · <code>GET /v1/tools</code> ·
<code>POST /v1/ask</code> · <code>GET /v1/evals/latest</code></p></div>
</main></body></html>"""

    def _send_json(self, status: HTTPStatus, payload: Any) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        remaining = getattr(self, "_rate_limit_remaining", None)
        if remaining is not None:
            self.send_header("X-RateLimit-Remaining", str(remaining))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, status: HTTPStatus, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: HTTPStatus, code: str, message: str) -> None:
        self._send_json(status, {"error": {"code": code, "message": message}})


def create_server(app: Application, config: ApiConfig) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((config.host, config.port), IAIRequestHandler)
    server.daemon_threads = True
    server.app = app  # type: ignore[attr-defined]
    server.config = config  # type: ignore[attr-defined]
    server.limiter = RateLimiter(config.rate_limit_per_minute)  # type: ignore[attr-defined]
    return server


class ServerHandle:
    """Context manager for running the API in a background thread."""

    def __init__(self, app: Application, config: ApiConfig) -> None:
        self.server = create_server(app, config)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "ServerHandle":
        self.thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    @property
    def port(self) -> int:
        return int(self.server.server_address[1])

    @property
    def url(self) -> str:
        host = self.server.server_address[0]
        if ":" in str(host):
            host = f"[{host}]"
        return f"http://{host}:{self.port}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def serve(app: Application, config: ApiConfig) -> None:
    server = create_server(app, config)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
