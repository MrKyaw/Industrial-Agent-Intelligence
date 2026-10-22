"""Tool contracts, policy enforcement, execution, and tamper-evident auditing.

The policy engine is deliberately conservative: read tools are cheap, write tools
are rate-limited, and physical-control tools are disabled unless an operator
explicitly enables them *and* supplies a confirmation token.
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable

from .models import ToolCall, ToolResult
from .util import format_timestamp, now_utc, sha256_text, stable_json

ROLE_LEVEL = {"viewer": 0, "operator": 1, "engineer": 2, "admin": 3}
RISK_LEVEL = {"read": 0, "low_write": 1, "high_write": 2, "physical_control": 3}
RISK_LABEL = {value: key for key, value in RISK_LEVEL.items()}

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")


class ToolError(ValueError):
    """Raised for invalid tool definitions or invocation arguments."""


class PolicyError(ValueError):
    """Raised for invalid policy configuration or context."""


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    risk: str
    required_role: str = "viewer"
    requires_confirmation: bool = False
    timeout_s: float = 10.0
    tags: list[str] = field(default_factory=list)
    handler: Callable[[dict[str, Any], "PolicyContext"], dict[str, Any]] | None = None
    precondition: Callable[[dict[str, Any], "PolicyContext"], str | None] | None = None

    def __post_init__(self) -> None:
        if not _NAME_RE.match(self.name):
            raise ToolError(f"invalid tool name: {self.name!r}")
        if not self.description or len(self.description) > 500:
            raise ToolError(f"{self.name}: description must be 1-500 characters")
        if not isinstance(self.input_schema, dict) or self.input_schema.get("type") != "object":
            raise ToolError(f"{self.name}: input_schema must be an object schema")
        if self.risk not in RISK_LEVEL:
            raise ToolError(f"{self.name}: invalid risk level {self.risk!r}")
        if self.required_role not in ROLE_LEVEL:
            raise ToolError(f"{self.name}: invalid role {self.required_role!r}")
        if self.timeout_s <= 0:
            raise ToolError(f"{self.name}: timeout_s must be positive")
        if self.handler is not None and not callable(self.handler):
            raise ToolError(f"{self.name}: handler must be callable")
        if self.precondition is not None and not callable(self.precondition):
            raise ToolError(f"{self.name}: precondition must be callable")

    @property
    def mutating(self) -> bool:
        return RISK_LEVEL[self.risk] >= RISK_LEVEL["low_write"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "risk": self.risk,
            "required_role": self.required_role,
            "requires_confirmation": self.requires_confirmation,
            "timeout_s": self.timeout_s,
            "tags": list(self.tags),
        }


class ToolRegistry:
    """Typed, discoverable tool catalogue."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ToolError(f"duplicate tool: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolError(f"unknown tool: {name}") from exc

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [self._tools[name] for name in self.names()]

    def for_role(self, role: str) -> list[ToolSpec]:
        if role not in ROLE_LEVEL:
            raise ToolError(f"unknown role: {role}")
        return [spec for spec in self.specs() if ROLE_LEVEL[spec.required_role] <= ROLE_LEVEL[role]]

    def to_dict(self) -> dict[str, Any]:
        return {"tools": [spec.to_dict() for spec in self.specs()]}

    def to_mcp(self) -> dict[str, Any]:
        """MCP-shaped tool descriptors (name, description, JSON input schema)."""
        return {
            "tools": [
                {
                    "name": spec.name,
                    "description": spec.description,
                    "inputSchema": spec.input_schema,
                }
                for spec in self.specs()
            ]
        }

    def __len__(self) -> int:
        return len(self._tools)


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return False


def validate_arguments(schema: dict[str, Any], arguments: Any, *, path: str = "$") -> list[str]:
    """Validate a JSON-Schema subset used by tool contracts."""
    errors: list[str] = []
    if schema.get("type") == "object":
        if not isinstance(arguments, dict):
            return [f"{path}: expected object"]
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, dict) or not isinstance(required, list):
            return [f"{path}: malformed object schema"]
        for key in required:
            if key not in arguments:
                errors.append(f"{path}.{key}: required")
        if schema.get("additionalProperties", True) is False:
            for key in sorted(set(arguments) - set(properties)):
                errors.append(f"{path}.{key}: additional property not allowed")
        for key, value in arguments.items():
            if key in properties:
                errors.extend(validate_arguments(properties[key], value, path=f"{path}.{key}"))
        return errors
    expected = schema.get("type")
    if expected is not None and not _matches_type(arguments, expected):
        return [f"{path}: expected {expected}"]
    if "enum" in schema:
        if not isinstance(schema["enum"], list):
            return [f"{path}: malformed enum"]
        if arguments not in schema["enum"]:
            errors.append(f"{path}: must be one of {schema['enum']}")
    if isinstance(arguments, (int, float)) and not isinstance(arguments, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, (int, float)) and arguments < minimum:
            errors.append(f"{path}: must be >= {minimum}")
        if isinstance(maximum, (int, float)) and arguments > maximum:
            errors.append(f"{path}: must be <= {maximum}")
    if isinstance(arguments, str):
        min_length = schema.get("minLength")
        max_length = schema.get("maxLength")
        pattern = schema.get("pattern")
        if isinstance(min_length, int) and len(arguments) < min_length:
            errors.append(f"{path}: must be at least {min_length} characters")
        if isinstance(max_length, int) and len(arguments) > max_length:
            errors.append(f"{path}: must be at most {max_length} characters")
        if isinstance(pattern, str) and not re.search(pattern, arguments):
            errors.append(f"{path}: must match {pattern}")
    if isinstance(arguments, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(arguments):
            errors.extend(validate_arguments(schema["items"], item, path=f"{path}[{index}]"))
    return errors


@dataclass
class PolicyContext:
    role: str
    site_id: str
    session_id: str = "local"
    confirmations: frozenset[str] = frozenset()
    dry_run: bool = True
    now: datetime = field(default_factory=now_utc)
    actor: str = "agent"

    def __post_init__(self) -> None:
        if self.role not in ROLE_LEVEL:
            raise PolicyError(f"unknown role: {self.role}")
        if not self.site_id:
            raise PolicyError("site_id is required")
        if not self.session_id:
            raise PolicyError("session_id is required")


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    code: str
    reason: str
    risk: str
    requires_confirmation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "code": self.code,
            "reason": self.reason,
            "risk": self.risk,
            "requires_confirmation": self.requires_confirmation,
        }


class PolicyEngine:
    """Role, schema, confirmation, and rate-limit checks for tool calls."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        allow_physical_control: bool = False,
        max_writes_per_hour: int = 10,
        clock: Callable[[], datetime] = now_utc,
    ) -> None:
        if max_writes_per_hour <= 0:
            raise PolicyError("max_writes_per_hour must be positive")
        self.registry = registry
        self.allow_physical_control = allow_physical_control
        self.max_writes_per_hour = max_writes_per_hour
        self._clock = clock
        self._writes: dict[str, list[datetime]] = defaultdict(list)

    def _write_count(self, session_id: str, now: datetime) -> int:
        cutoff = now - timedelta(hours=1)
        recent = [stamp for stamp in self._writes[session_id] if stamp >= cutoff]
        self._writes[session_id] = recent
        return len(recent)

    def decide(self, call: ToolCall, context: PolicyContext) -> PolicyDecision:
        if not self.registry.has(call.name):
            return PolicyDecision(False, "unknown_tool", f"Tool {call.name!r} is not registered.", "unknown")
        spec = self.registry.get(call.name)
        if ROLE_LEVEL[context.role] < ROLE_LEVEL[spec.required_role]:
            return PolicyDecision(
                False,
                "role_denied",
                f"Tool {spec.name!r} requires role {spec.required_role!r}; caller is {context.role!r}.",
                spec.risk,
            )
        errors = validate_arguments(spec.input_schema, call.arguments)
        if errors:
            return PolicyDecision(
                False,
                "invalid_arguments",
                "Invalid arguments: " + "; ".join(errors[:4]),
                spec.risk,
            )
        if spec.precondition is not None:
            failure = spec.precondition(call.arguments, context)
            if failure:
                return PolicyDecision(False, "interlock_failed", failure, spec.risk)
        confirmation_required = spec.requires_confirmation or spec.risk == "physical_control"
        if spec.risk == "physical_control":
            if not self.allow_physical_control:
                return PolicyDecision(
                    False,
                    "physical_control_disabled",
                    "Physical control is disabled by policy; use a proposal and human approval.",
                    spec.risk,
                    requires_confirmation=True,
                )
            if "approve_physical_control" not in context.confirmations:
                return PolicyDecision(
                    False,
                    "confirmation_required",
                    "Physical control requires an explicit approval token.",
                    spec.risk,
                    requires_confirmation=True,
                )
        if spec.mutating:
            now = self._clock()
            if self._write_count(context.session_id, now) >= self.max_writes_per_hour:
                return PolicyDecision(
                    False,
                    "rate_limited",
                    f"Session exceeded {self.max_writes_per_hour} writes per hour.",
                    spec.risk,
                )
        return PolicyDecision(
            True,
            "allowed",
            f"Allowed {spec.name!r} at risk level {spec.risk!r}.",
            spec.risk,
            requires_confirmation=confirmation_required,
        )

    def record_write(self, session_id: str) -> None:
        self._writes[session_id].append(self._clock())


@dataclass(frozen=True)
class AuditEntry:
    seq: int
    at: str
    actor: str
    action: str
    payload: dict[str, Any]
    prev_hash: str
    entry_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "at": self.at,
            "actor": self.actor,
            "action": self.action,
            "payload": self.payload,
            "prev_hash": self.prev_hash,
            "entry_hash": self.entry_hash,
        }


class AuditLog:
    """Append-only SHA-256 hash chain for tool calls and decisions."""

    GENESIS = "0" * 64

    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []

    def append(self, actor: str, action: str, payload: dict[str, Any]) -> AuditEntry:
        if not actor or not action:
            raise ToolError("audit actor and action are required")
        prev_hash = self._entries[-1].entry_hash if self._entries else self.GENESIS
        body = {
            "seq": len(self._entries) + 1,
            "at": format_timestamp(now_utc()),
            "actor": actor,
            "action": action,
            "payload": payload,
        }
        entry_hash = sha256_text(prev_hash + stable_json(body))
        entry = AuditEntry(prev_hash=prev_hash, entry_hash=entry_hash, **body)
        self._entries.append(entry)
        return entry

    def verify(self) -> bool:
        prev_hash = self.GENESIS
        for index, entry in enumerate(self._entries, start=1):
            if entry.seq != index or entry.prev_hash != prev_hash:
                return False
            body = {
                "seq": entry.seq,
                "at": entry.at,
                "actor": entry.actor,
                "action": entry.action,
                "payload": entry.payload,
            }
            if sha256_text(prev_hash + stable_json(body)) != entry.entry_hash:
                return False
            prev_hash = entry.entry_hash
        return True

    @property
    def entries(self) -> list[AuditEntry]:
        return list(self._entries)

    def to_jsonl(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for entry in self._entries:
                handle.write(json.dumps(entry.to_dict(), sort_keys=True) + "\n")

    def __len__(self) -> int:
        return len(self._entries)


class ToolExecutor:
    """Policy-checked execution with an audit trail."""

    def __init__(
        self,
        registry: ToolRegistry,
        policy: PolicyEngine,
        audit: AuditLog | None = None,
    ) -> None:
        self.registry = registry
        self.policy = policy
        self.audit = audit if audit is not None else AuditLog()

    def execute(self, call: ToolCall, context: PolicyContext) -> ToolResult:
        decision = self.policy.decide(call, context)
        self.audit.append(
            context.actor,
            "tool.request",
            {
                "tool": call.name,
                "arguments": call.arguments,
                "site_id": context.site_id,
                "session_id": context.session_id,
                "role": context.role,
                "decision": decision.to_dict(),
            },
        )
        if not decision.allowed:
            return ToolResult(
                call=call,
                ok=False,
                error=decision.reason,
                error_code=decision.code,
            )
        spec = self.registry.get(call.name)
        if spec.handler is None:
            return ToolResult(call=call, ok=False, error="tool has no handler", error_code="no_handler")
        started = time.perf_counter()
        try:
            output = spec.handler(call.arguments, context)
            if not isinstance(output, dict):
                raise ToolError("tool handler must return a dictionary")
        except Exception as exc:  # noqa: BLE001 - sanitized into a tool result
            elapsed = (time.perf_counter() - started) * 1000
            message = str(exc) or exc.__class__.__name__
            self.audit.append(
                context.actor,
                "tool.error",
                {"tool": call.name, "error": message[:500]},
            )
            return ToolResult(
                call=call,
                ok=False,
                error=message[:500],
                error_code="tool_error",
                duration_ms=elapsed,
            )
        elapsed = (time.perf_counter() - started) * 1000
        if spec.mutating:
            self.policy.record_write(context.session_id)
        result = ToolResult(
            call=call,
            ok=True,
            output=output,
            simulated=context.dry_run and spec.mutating,
            duration_ms=elapsed,
        )
        self.audit.append(
            context.actor,
            "tool.result",
            {
                "tool": call.name,
                "ok": True,
                "simulated": result.simulated,
                "duration_ms": round(elapsed, 3),
            },
        )
        return result

    def execute_many(
        self, calls: Iterable[ToolCall], context: PolicyContext
    ) -> list[ToolResult]:
        return [self.execute(call, context) for call in calls]
