"""Application wiring: data loading, default tools, and the agent runtime.

Keeping assembly in one file makes the dependency graph reviewable: ontology and
telemetry are loaded once, tool handlers are closure-bound to that data, and the
policy engine wraps every invocation.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .alarms import Alarm, AlarmStore, cluster_alarms, summarize_window, triage
from .guardrails import detect_injection
from .models import ToolCall
from .ontology import Ontology, UnknownEntityError, load_ontology
from .prompts import PromptLibrary
from .reasoning import AgentConfig, AgentRuntime
from .retrieval import ContextPacker, HybridRetriever, load_knowledge_dir
from .telemetry import DerivedTagEngine, TagStore, TelemetryError
from .tools import (
    AuditLog,
    PolicyContext,
    PolicyEngine,
    ToolError,
    ToolExecutor,
    ToolRegistry,
    ToolSpec,
)
from .util import fingerprint, format_timestamp, now_utc, parse_timestamp


def default_data_dir() -> Path:
    """Resolve the bundled data directory for source checkouts and installs."""
    override = os.environ.get("IAI_DATA_DIR")
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"IAI_DATA_DIR does not exist: {path}")
        return path
    repository = Path(__file__).resolve().parents[2]
    candidate = repository / "data"
    if candidate.is_dir():
        return candidate
    local = Path.cwd() / "data"
    if local.is_dir():
        return local
    raise FileNotFoundError(
        "Could not locate the data directory. Set IAI_DATA_DIR or run from the repository root."
    )


@dataclass
class AppConfig:
    allow_physical_control: bool = False
    stale_after_s: float = 7200.0
    max_writes_per_hour: int = 10
    max_tool_calls: int = 16
    agent_version: str = "1.0.0"


@dataclass
class Application:
    data_dir: Path
    ontology: Ontology
    telemetry: TagStore
    alarms: AlarmStore
    prompts: PromptLibrary
    retriever: HybridRetriever
    as_of: datetime
    config: AppConfig = field(default_factory=AppConfig)
    registry: ToolRegistry = field(default_factory=ToolRegistry)
    audit: AuditLog = field(default_factory=AuditLog)
    policy: PolicyEngine | None = None
    executor: ToolExecutor | None = None
    runtime: AgentRuntime | None = None
    manifest: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        data_dir: str | Path | None = None,
        *,
        config: AppConfig | None = None,
        clock: Any | None = None,
    ) -> "Application":
        root = Path(data_dir) if data_dir is not None else default_data_dir()
        if not root.is_dir():
            raise FileNotFoundError(f"data directory not found: {root}")
        ontology = load_ontology(root / "ontology.json")
        units = {
            node.id: str(node.attributes["unit"])
            for node in ontology.nodes()
            if isinstance(node.attributes.get("unit"), str)
        }
        telemetry = TagStore.from_dir(root / "telemetry", units)
        alarm_path = root / "alarms.jsonl"
        alarms = AlarmStore.from_jsonl(alarm_path) if alarm_path.exists() else AlarmStore()
        documents = load_knowledge_dir(root / "knowledge")
        retriever = HybridRetriever(documents, ontology)
        prompts = PromptLibrary.load_dir(root / "prompts")
        manifest_path = root / "manifest.json"
        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        as_of = parse_timestamp(manifest.get("as_of", format_timestamp(now_utc())))
        app = cls(
            data_dir=root,
            ontology=ontology,
            telemetry=telemetry,
            alarms=alarms,
            prompts=prompts,
            retriever=retriever,
            as_of=as_of,
            config=config or AppConfig(),
            manifest=manifest,
        )
        app._wire(clock)
        return app

    def _wire(self, clock: Any | None) -> None:
        self.registry = build_default_registry(self)
        self.audit = AuditLog()
        self.policy = PolicyEngine(
            self.registry,
            allow_physical_control=self.config.allow_physical_control,
            max_writes_per_hour=self.config.max_writes_per_hour,
            clock=clock or (lambda: self.as_of),
        )
        self.executor = ToolExecutor(self.registry, self.policy, self.audit)
        self.runtime = AgentRuntime(
            ontology=self.ontology,
            telemetry=self.telemetry,
            alarms=self.alarms,
            retriever=self.retriever,
            executor=self.executor,
            prompts=self.prompts,
            clock=clock or (lambda: self.as_of),
            config=AgentConfig(
                stale_after_s=self.config.stale_after_s,
                max_tool_calls=self.config.max_tool_calls,
                agent_version=self.config.agent_version,
            ),
            audit=self.audit,
        )

    @property
    def now(self) -> datetime:
        return self.as_of

    def site_ids(self) -> list[str]:
        return [site.id for site in self.ontology.sites()]

    def ask(self, question: str, *, site_id: str, role: str = "operator"):
        if self.runtime is None:
            raise RuntimeError("application is not wired")
        return self.runtime.ask(question, site_id=site_id, role=role)

    def status(self) -> dict[str, Any]:
        return {
            "agent_version": self.config.agent_version,
            "dataset": self.manifest.get("dataset", "unknown"),
            "as_of": format_timestamp(self.as_of),
            "sites": self.site_ids(),
            "ontology": self.ontology.summary(),
            "tools": len(self.registry),
            "knowledge_documents": len(self.retriever.documents),
            "alarms": len(self.alarms),
            "allow_physical_control": self.config.allow_physical_control,
        }

    def validate(self) -> dict[str, Any]:
        report = self.ontology.validate()
        return {
            "ontology": report.to_dict(),
            "telemetry_series": len(self.telemetry.names()),
            "knowledge_documents": len(self.retriever.documents),
            "tools": len(self.registry),
            "alarms": len(self.alarms),
        }

    def tool_manifest(self, role: str = "viewer") -> dict[str, Any]:
        return {"tools": [spec.to_dict() for spec in self.registry.for_role(role)]}

    def audit_export(self) -> list[dict[str, Any]]:
        return [entry.to_dict() for entry in self.audit.entries]


# ---------------------------------------------------------------------------
# Default tool handlers
# ---------------------------------------------------------------------------

STALE_AFTER_S = 7200.0
KPI_SUFFIX = {
    "cop": "cop",
    "kw_per_ton": "kw_per_ton",
    "delta_t": "delta_t",
    "approach": "approach_temp",
    "tons": "tons",
}


def _resolve(ontology: Ontology, alias: str, expected: set[str] | None = None) -> Any:
    try:
        node = ontology.resolve(alias)
    except UnknownEntityError as exc:
        raise ToolError(str(exc)) from exc
    if expected is not None and node.type not in expected:
        raise ToolError(f"{node.id} is a {node.type}, expected one of {sorted(expected)}")
    return node


def _require_site(app: Application, site_id: str) -> None:
    if not app.ontology.has(site_id) or app.ontology.get(site_id).type != "site":
        raise ToolError(f"unknown site: {site_id}")


def _series_payload(app: Application, tag_id: str, max_points: int) -> dict[str, Any]:
    if app.runtime is None:
        raise ToolError("runtime is not ready")
    series = app.runtime.derived.evaluate(tag_id)
    node = app.ontology.get(tag_id)
    expected_interval = node.attributes.get("sample_interval_s")
    if not isinstance(expected_interval, (int, float)):
        expected_interval = None
    stale = series.is_stale(app.as_of, STALE_AFTER_S)
    gaps = series.gaps(expected_interval_s=expected_interval)
    anomalies = series.detect_anomalies(window=12, z_threshold=3.5)
    return {
        "tag": tag_id,
        "unit": series.unit,
        "summary": series.summary(),
        "records": series.to_records(max_points),
        "stale": stale,
        "stale_after_s": STALE_AFTER_S,
        "expected_interval_s": expected_interval,
        "gaps": [gap.to_dict() for gap in gaps[:10]],
        "anomalies": [anomaly.to_dict() for anomaly in anomalies[-10:]],
    }


def _get_tag_series(arguments: dict[str, Any], context: PolicyContext) -> dict[str, Any]:
    app: Application = _get_tag_series.app  # type: ignore[attr-defined]
    node = _resolve(app.ontology, str(arguments["tag"]), {"sensor", "derived_tag"})
    return _series_payload(app, node.id, int(arguments.get("max_points", 48)))


def _compute_kpi(arguments: dict[str, Any], context: PolicyContext) -> dict[str, Any]:
    app: Application = _compute_kpi.app  # type: ignore[attr-defined]
    equipment = _resolve(app.ontology, str(arguments["equipment_id"]), {"equipment"})
    kpi = str(arguments["kpi"])
    short = equipment.id.split(".")[-1]
    tag_id = f"derived.{short}.{KPI_SUFFIX[kpi]}"
    if not app.ontology.has(tag_id):
        raise ToolError(f"{kpi} is not configured for {equipment.id}")
    payload = _series_payload(app, tag_id, 48)
    if app.runtime is None:
        raise ToolError("runtime is not ready")
    series = app.runtime.derived.evaluate(tag_id)
    shift = series.detect_level_shift(window=6, min_change_pct=5.0)
    return {
        "kpi": kpi,
        "tag": tag_id,
        "unit": payload["unit"],
        "summary": payload["summary"],
        "anomalies": payload["anomalies"],
        "level_shift": shift.to_dict() if shift else None,
        "stale": payload["stale"],
    }


def _get_alarm_history(arguments: dict[str, Any], context: PolicyContext) -> dict[str, Any]:
    app: Application = _get_alarm_history.app  # type: ignore[attr-defined]
    site_id = str(arguments["site_id"])
    _require_site(app, site_id)
    since_hours = int(arguments.get("since_hours", 24))
    since = app.as_of.timestamp() - since_hours * 3600
    equipment_id = arguments.get("equipment_id")
    if equipment_id:
        equipment_id = _resolve(app.ontology, str(equipment_id), {"equipment"}).id
    selected = [
        alarm
        for alarm in app.alarms.filter(
            site_id=site_id,
            equipment_id=equipment_id,
            active_only=bool(arguments.get("active_only", False)),
        )
        if alarm.raised_at.timestamp() >= since
    ]
    summary = summarize_window(app.alarms.all(), site_id=site_id, now=app.as_of, hours=since_hours)
    if equipment_id:
        summary = {
            **summary,
            "total": len(selected),
            "active": sum(1 for alarm in selected if alarm.active),
            "by_equipment": {equipment_id: len(selected)},
        }
    clusters = cluster_alarms(selected, window_s=1800)
    return {
        "summary": summary,
        "alarms": [alarm.to_dict() for alarm in selected[-50:]],
        "clusters": [cluster.to_dict() for cluster in clusters[:25]],
    }


def _triage_alarms(arguments: dict[str, Any], context: PolicyContext) -> dict[str, Any]:
    app: Application = _triage_alarms.app  # type: ignore[attr-defined]
    site_id = str(arguments["site_id"])
    _require_site(app, site_id)
    window_minutes = int(arguments.get("window_minutes", 30))
    root_equipment = arguments.get("root_cause_equipment")
    if root_equipment:
        root_equipment = _resolve(app.ontology, str(root_equipment), {"equipment"}).id
    anomaly_scores: dict[str, float] = {}
    for equipment in app.ontology.equipment(site_id):
        for sensor in app.ontology.sensors_for(equipment.id):
            if not app.telemetry.has(sensor.id):
                continue
            anomalies = app.telemetry.get(sensor.id).detect_anomalies(window=12)
            if anomalies:
                anomaly_scores[sensor.id] = max(anomaly.score for anomaly in anomalies)
    recent = [
        alarm
        for alarm in app.alarms.filter(site_id=site_id)
        if (app.as_of - alarm.raised_at).total_seconds() <= 86400
    ]
    summary = triage(
        recent,
        site_id=site_id,
        ontology=app.ontology,
        now=app.as_of,
        window_s=window_minutes * 60,
        root_cause_equipment=root_equipment,
        anomaly_scores=anomaly_scores,
    )
    return summary.to_dict()


def _search_knowledge(arguments: dict[str, Any], context: PolicyContext) -> dict[str, Any]:
    app: Application = _search_knowledge.app  # type: ignore[attr-defined]
    query = str(arguments["query"])
    limit = int(arguments.get("limit", 4))
    hits = app.retriever.search(query, limit=limit, site_id=context.site_id)
    packed = ContextPacker().pack(hits)
    return {
        "query": query,
        "hits": [hit.to_dict() for hit in hits],
        "context": packed.text,
        "context_tokens": packed.tokens,
    }


def _query_ontology(arguments: dict[str, Any], context: PolicyContext) -> dict[str, Any]:
    app: Application = _query_ontology.app  # type: ignore[attr-defined]
    node = _resolve(app.ontology, str(arguments["node_id"]))
    hops = int(arguments.get("hops", 1))
    nodes = app.ontology.subgraph(node.id, hops=hops)
    ids = {item.id for item in nodes}
    edges = [
        edge.to_dict()
        for edge in app.ontology.edges()
        if edge.src in ids and edge.dst in ids
    ]
    return {
        "node": node.to_dict(),
        "description": app.ontology.describe(node.id, hops=hops),
        "nodes": [
            {"id": item.id, "type": item.type, "name": item.name, "description": app.ontology.describe(item.id)}
            for item in nodes
        ],
        "edges": edges,
    }


def _propose_setpoint(arguments: dict[str, Any], context: PolicyContext) -> dict[str, Any]:
    app: Application = _propose_setpoint.app  # type: ignore[attr-defined]
    node = _resolve(app.ontology, str(arguments["tag"]), {"setpoint"})
    value = float(arguments["value"])
    minimum = float(node.attributes["min"])
    maximum = float(node.attributes["max"])
    current = float(node.attributes.get("default", minimum))
    deadband = float(node.attributes.get("deadband", 0.0))
    writable = bool(node.attributes.get("writable", False))
    accepted = writable and minimum <= value <= maximum and abs(value - current) >= deadband
    reason = "The requested value is inside the configured operating envelope."
    if not writable:
        reason = "The setpoint is configured read-only."
    elif not minimum <= value <= maximum:
        reason = f"The requested value is outside the allowed range {minimum}-{maximum}."
    elif abs(value - current) < deadband:
        reason = f"The requested change is inside the {deadband} deadband around the current value."
    proposal_id = "SP-" + fingerprint(
        {"tag": node.id, "value": value, "current": current, "at": format_timestamp(app.as_of)}
    )[:12]
    return {
        "accepted": accepted,
        "proposal_id": proposal_id,
        "tag": node.id,
        "name": node.name,
        "value": value,
        "current": current,
        "min": minimum,
        "max": maximum,
        "deadband": deadband,
        "unit": str(node.attributes.get("unit", "")),
        "reason": reason,
        "requires_confirmation": True,
        "simulated": True,
    }


def _write_setpoint(arguments: dict[str, Any], context: PolicyContext) -> dict[str, Any]:
    app: Application = _write_setpoint.app  # type: ignore[attr-defined]
    node = _resolve(app.ontology, str(arguments["tag"]), {"setpoint"})
    value = float(arguments["value"])
    minimum = float(node.attributes["min"])
    maximum = float(node.attributes["max"])
    if not minimum <= value <= maximum:
        raise ToolError(f"value {value} is outside {minimum}-{maximum}")
    if arguments["confirmation_token"] != arguments["proposal_id"]:
        raise ToolError("confirmation token does not match the proposal id")
    return {
        "applied": not context.dry_run,
        "simulated": context.dry_run,
        "tag": node.id,
        "value": value,
        "proposal_id": arguments["proposal_id"],
        "message": "Simulated write; no controller was contacted."
        if context.dry_run
        else "Write dispatched through the approved control path.",
    }


def _create_work_order(arguments: dict[str, Any], context: PolicyContext) -> dict[str, Any]:
    ticket = "WO-" + fingerprint(
        {"equipment": arguments["equipment_id"], "summary": arguments["summary"]}
    )[:10].upper()
    return {
        "ticket": ticket,
        "status": "simulated" if context.dry_run else "created",
        "simulated": context.dry_run,
        "priority": arguments.get("priority", "medium"),
    }


def _escalate_to_human(arguments: dict[str, Any], context: PolicyContext) -> dict[str, Any]:
    ticket = "ESC-" + fingerprint(
        {"reason": arguments["reason"], "site": context.site_id}
    )[:10].upper()
    return {
        "ticket": ticket,
        "status": "queued",
        "severity": arguments.get("severity", "high"),
        "reason": arguments["reason"],
    }


def build_default_registry(app: Application) -> ToolRegistry:
    """Register the read-first industrial tool catalogue."""
    registry = ToolRegistry()

    def bind(handler: Any, application: Application) -> Any:
        handler.app = application
        return handler

    registry.register(
        ToolSpec(
            name="get_tag_series",
            description="Read a sensor or derived-tag series with freshness, gap, and anomaly metadata.",
            input_schema={
                "type": "object",
                "properties": {
                    "tag": {"type": "string", "minLength": 3},
                    "max_points": {"type": "integer", "minimum": 1, "maximum": 500, "default": 48},
                },
                "required": ["tag"],
                "additionalProperties": False,
            },
            risk="read",
            required_role="viewer",
            tags=["telemetry"],
            handler=bind(_get_tag_series, app),
        )
    )
    registry.register(
        ToolSpec(
            name="compute_kpi",
            description="Compute a configured KPI (COP, kW/ton, delta-T, approach, tons) for an asset.",
            input_schema={
                "type": "object",
                "properties": {
                    "equipment_id": {"type": "string", "minLength": 3},
                    "kpi": {
                        "type": "string",
                        "enum": ["cop", "kw_per_ton", "delta_t", "approach", "tons"],
                    },
                },
                "required": ["equipment_id", "kpi"],
                "additionalProperties": False,
            },
            risk="read",
            required_role="viewer",
            tags=["telemetry", "kpi"],
            handler=bind(_compute_kpi, app),
        )
    )
    registry.register(
        ToolSpec(
            name="get_alarm_history",
            description="Read alarm history for a site or asset, including deduplicated clusters.",
            input_schema={
                "type": "object",
                "properties": {
                    "site_id": {"type": "string", "minLength": 3},
                    "since_hours": {"type": "integer", "minimum": 1, "maximum": 168, "default": 24},
                    "equipment_id": {"type": "string"},
                    "active_only": {"type": "boolean", "default": False},
                },
                "required": ["site_id"],
                "additionalProperties": False,
            },
            risk="read",
            required_role="viewer",
            tags=["alarms"],
            handler=bind(_get_alarm_history, app),
        )
    )
    registry.register(
        ToolSpec(
            name="triage_alarms",
            description="Cluster, rank, and explain alarms, optionally ranking related root-cause candidates.",
            input_schema={
                "type": "object",
                "properties": {
                    "site_id": {"type": "string", "minLength": 3},
                    "window_minutes": {"type": "integer", "minimum": 5, "maximum": 1440, "default": 30},
                    "root_cause_equipment": {"type": "string"},
                },
                "required": ["site_id"],
                "additionalProperties": False,
            },
            risk="read",
            required_role="viewer",
            tags=["alarms", "triage"],
            handler=bind(_triage_alarms, app),
        )
    )
    registry.register(
        ToolSpec(
            name="search_knowledge",
            description="Hybrid BM25 retrieval over maintenance playbooks, with fenced untrusted context.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 3, "maxLength": 500},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 8, "default": 4},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            risk="read",
            required_role="viewer",
            tags=["knowledge", "rag"],
            handler=bind(_search_knowledge, app),
        )
    )
    registry.register(
        ToolSpec(
            name="query_ontology",
            description="Read an asset, sensor, or setpoint and its typed neighbourhood.",
            input_schema={
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "minLength": 3},
                    "hops": {"type": "integer", "minimum": 0, "maximum": 2, "default": 1},
                },
                "required": ["node_id"],
                "additionalProperties": False,
            },
            risk="read",
            required_role="viewer",
            tags=["ontology"],
            handler=bind(_query_ontology, app),
        )
    )
    registry.register(
        ToolSpec(
            name="propose_setpoint",
            description="Validate a setpoint proposal against bounds and deadband without writing.",
            input_schema={
                "type": "object",
                "properties": {
                    "tag": {"type": "string", "minLength": 3},
                    "value": {"type": "number"},
                },
                "required": ["tag", "value"],
                "additionalProperties": False,
            },
            risk="read",
            required_role="viewer",
            tags=["setpoint", "safety"],
            handler=bind(_propose_setpoint, app),
        )
    )
    registry.register(
        ToolSpec(
            name="write_setpoint",
            description="Apply a confirmed setpoint proposal. Disabled unless physical control is enabled.",
            input_schema={
                "type": "object",
                "properties": {
                    "tag": {"type": "string", "minLength": 3},
                    "value": {"type": "number"},
                    "proposal_id": {"type": "string", "minLength": 6},
                    "confirmation_token": {"type": "string", "minLength": 6},
                },
                "required": ["tag", "value", "proposal_id", "confirmation_token"],
                "additionalProperties": False,
            },
            risk="physical_control",
            required_role="engineer",
            requires_confirmation=True,
            tags=["setpoint", "physical-control"],
            handler=bind(_write_setpoint, app),
        )
    )
    registry.register(
        ToolSpec(
            name="create_work_order",
            description="Create a simulated maintenance work order (no dispatch in demo mode).",
            input_schema={
                "type": "object",
                "properties": {
                    "equipment_id": {"type": "string", "minLength": 3},
                    "summary": {"type": "string", "minLength": 5, "maxLength": 300},
                    "priority": {"type": "string", "enum": ["low", "medium", "high"], "default": "medium"},
                },
                "required": ["equipment_id", "summary"],
                "additionalProperties": False,
            },
            risk="low_write",
            required_role="operator",
            tags=["work-order"],
            handler=bind(_create_work_order, app),
        )
    )
    registry.register(
        ToolSpec(
            name="escalate_to_human",
            description="Queue an escalation to a human operator without changing physical state.",
            input_schema={
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "minLength": 5, "maxLength": 300},
                    "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"], "default": "high"},
                },
                "required": ["reason"],
                "additionalProperties": False,
            },
            risk="read",
            required_role="viewer",
            tags=["escalation"],
            handler=bind(_escalate_to_human, app),
        )
    )
    return registry
