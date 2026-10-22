"""Tests for application wiring and default tools."""

from __future__ import annotations

import pytest

from iai.app import Application, default_data_dir
from iai.tools import ToolError


def test_application_loads(app):
    status = app.status()
    assert status["tools"] == 10
    assert status["ontology"]["nodes"] > 40
    validation = app.validate()
    assert validation["ontology"]["ok"]


def test_tools_manifest_role_filter(app):
    manifest = app.tool_manifest("viewer")
    names = {tool["name"] for tool in manifest["tools"]}
    assert "write_setpoint" not in names
    assert "propose_setpoint" in names
    engineer = app.tool_manifest("engineer")
    assert "write_setpoint" in {tool["name"] for tool in engineer["tools"]}


def test_tool_schema_validation(app):
    result = app.executor.execute(
        __import__("iai.models", fromlist=["ToolCall"]).ToolCall("get_tag_series", {}),
        __import__("iai.tools", fromlist=["PolicyContext"]).PolicyContext(role="viewer", site_id="site.dc-lon-01"),
    )
    assert result.ok is False
    assert result.error_code == "invalid_arguments"


def test_propose_setpoint_tool(app):
    result = app.executor.execute(
        __import__("iai.models", fromlist=["ToolCall"]).ToolCall(
            "propose_setpoint", {"tag": "setpoint.chiller-1.chw_supply", "value": 6.5}
        ),
        __import__("iai.tools", fromlist=["PolicyContext"]).PolicyContext(role="viewer", site_id="site.dc-lon-01"),
    )
    assert result.ok
    assert result.output["accepted"] is True
    assert result.output["min"] == 5.0


def test_query_ontology_tool(app):
    result = app.executor.execute(
        __import__("iai.models", fromlist=["ToolCall"]).ToolCall("query_ontology", {"node_id": "plant.chiller-1", "hops": 1}),
        __import__("iai.tools", fromlist=["PolicyContext"]).PolicyContext(role="viewer", site_id="site.dc-lon-01"),
    )
    assert result.ok
    assert result.output["node"]["id"] == "plant.chiller-1"


def test_search_knowledge_tool(app):
    result = app.executor.execute(
        __import__("iai.models", fromlist=["ToolCall"]).ToolCall("search_knowledge", {"query": "condenser pressure", "limit": 2}),
        __import__("iai.tools", fromlist=["PolicyContext"]).PolicyContext(role="viewer", site_id="site.dc-lon-01"),
    )
    assert result.ok
    assert len(result.output["hits"]) >= 1


def test_audit_export(app):
    assert isinstance(app.audit_export(), list)


def test_default_data_dir():
    path = default_data_dir()
    assert (path / "ontology.json").exists()
