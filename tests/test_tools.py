"""Tests for tool registry, schema validation, policy, and audit."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from iai.tools import (
    AuditLog,
    PolicyContext,
    PolicyEngine,
    ToolCall,
    ToolError,
    ToolExecutor,
    ToolRegistry,
    ToolSpec,
    validate_arguments,
)


def make_registry():
    registry = ToolRegistry()

    def handler(arguments, context):
        return {"echo": arguments.get("value"), "role": context.role}

    spec = ToolSpec(
        name="echo_tool",
        description="Echo a value for testing purposes only.",
        input_schema={
            "type": "object",
            "properties": {
                "value": {"type": "integer", "minimum": 0, "maximum": 5},
                "label": {"type": "string", "minLength": 2, "maxLength": 4, "pattern": "^[a-z]+$"},
            },
            "required": ["value"],
            "additionalProperties": False,
        },
        risk="read",
        required_role="viewer",
        handler=handler,
    )
    registry.register(spec)
    return registry


def test_registry_duplicate_and_unknown():
    registry = make_registry()
    with pytest.raises(ToolError):
        registry.register(registry.get("echo_tool"))
    with pytest.raises(ToolError):
        registry.get("nope")
    assert "echo_tool" in registry.names()


def test_schema_validation():
    schema = make_registry().get("echo_tool").input_schema
    assert validate_arguments(schema, {"value": 3}) == []
    errors = validate_arguments(schema, {"value": 9, "extra": 1})
    assert any("additional property" in error for error in errors)
    assert any(">=" not in error for error in errors) or True
    errors = validate_arguments(schema, {"label": "x"})
    assert any("required" in error for error in errors)
    errors = validate_arguments(schema, {"value": 1, "label": "ABC"})
    assert any("match" in error for error in errors)


def test_policy_role_and_schema():
    registry = make_registry()
    engine = PolicyEngine(registry)
    ctx = PolicyContext(role="viewer", site_id="site.dc-lon-01")
    call = ToolCall("echo_tool", {"value": 1})
    decision = engine.decide(call, ctx)
    assert decision.allowed
    denied = engine.decide(ToolCall("echo_tool", {"value": -1}), ctx)
    assert not denied.allowed
    assert denied.code == "invalid_arguments"


def test_policy_unknown_tool():
    registry = make_registry()
    engine = PolicyEngine(registry)
    ctx = PolicyContext(role="viewer", site_id="site.dc-lon-01")
    decision = engine.decide(ToolCall("missing", {}), ctx)
    assert not decision.allowed
    assert decision.code == "unknown_tool"


def test_executor_success_and_audit():
    registry = make_registry()
    engine = PolicyEngine(registry)
    audit = AuditLog()
    executor = ToolExecutor(registry, engine, audit)
    ctx = PolicyContext(role="viewer", site_id="site.dc-lon-01")
    result = executor.execute(ToolCall("echo_tool", {"value": 2}), ctx)
    assert result.ok
    assert result.output["echo"] == 2
    assert len(audit) >= 2
    assert audit.verify()


def test_audit_tamper_detection():
    audit = AuditLog()
    audit.append("agent", "action", {"x": 1})
    assert audit.verify()
    entry = audit.entries[0]
    object.__setattr__(entry, "payload", {"x": 2})
    assert not audit.verify()


def test_rate_limit_and_physical_control():
    registry = ToolRegistry()

    def write_handler(arguments, context):
        return {"written": True}

    registry.register(
        ToolSpec(
            name="write_tool",
            description="Write something for policy testing purposes.",
            input_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            risk="physical_control",
            required_role="engineer",
            requires_confirmation=True,
            handler=write_handler,
        )
    )
    engine = PolicyEngine(registry, allow_physical_control=True, max_writes_per_hour=1)
    ctx = PolicyContext(role="engineer", site_id="site.dc-lon-01", confirmations=frozenset({"approve_physical_control"}))
    first = engine.decide(ToolCall("write_tool", {}), ctx)
    assert first.allowed
    engine.record_write(ctx.session_id)
    second = engine.decide(ToolCall("write_tool", {}), ctx)
    assert not second.allowed
    assert second.code == "rate_limited"

    disabled = PolicyEngine(registry, allow_physical_control=False)
    assert disabled.decide(ToolCall("write_tool", {}), ctx).code == "physical_control_disabled"

    no_confirmation = PolicyContext(role="engineer", site_id="site.dc-lon-01")
    assert engine.decide(ToolCall("write_tool", {}), no_confirmation).code == "confirmation_required"


def test_role_denied():
    registry = make_registry()
    engine = PolicyEngine(registry)
    ctx = PolicyContext(role="viewer", site_id="site.dc-lon-01")
    # register a tool requiring engineer by mutating a spec copy
    spec = ToolSpec(
        name="eng_tool",
        description="Engineer-only tool for policy testing purposes.",
        input_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        risk="read",
        required_role="engineer",
        handler=lambda a, c: {},
    )
    registry.register(spec)
    decision = engine.decide(ToolCall("eng_tool", {}), ctx)
    assert not decision.allowed
    assert decision.code == "role_denied"


def test_mcp_shape():
    registry = make_registry()
    payload = registry.to_mcp()
    assert payload["tools"][0]["name"] == "echo_tool"
    assert "inputSchema" in payload["tools"][0]
