"""Reference industrial agent runtime.

The runtime is a plan-and-synthesise loop, not an opaque LLM wrapper. Each intent
maps to an explicit investigation plan; every tool call is policy-checked, every
claim is linked to evidence, and the finished answer passes through deterministic
guardrails before it is returned.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from .alarms import AlarmStore
from .guardrails import GuardrailContext, detect_injection, evaluate_guardrails
from .models import (
    AgentAnswer,
    AgentRequest,
    Evidence,
    EvidenceLedger,
    ReasoningStep,
    ToolResult,
)
from .ontology import Ontology, OntologyNode
from .prompts import PromptLibrary
from .retrieval import HybridRetriever
from .telemetry import DerivedTagEngine, TagStore
from .tools import AuditLog, PolicyContext, ToolCall, ToolExecutor
from .util import format_timestamp, now_utc, redact_secrets, stable_json

INTENTS = {
    "triage",
    "flood",
    "efficiency",
    "setpoint",
    "data_quality",
    "ambiguous",
    "prompt_injection",
    "unsupported",
}

SETPOINT_TARGET_RE = re.compile(
    r"\b(?:to|at)\s+(-?\d+(?:\.\d+)?)\s*(?:°|deg(?:rees)?)?\s*[cC]\b",
    re.I,
)
TEMPERATURE_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*(?:°|deg(?:rees)?)?\s*[cC]\b", re.I
)

FLOOD_RE = re.compile(r"\b(flood|dozens|too many|which.*first|prioriti[sz]e)\b", re.I)
EFFICIENCY_RE = re.compile(
    r"\b(efficien\w*|kw\s*/?\s*ton|kw per ton|\bcop\b|performance|optimi[sz]\w*|energy)\b",
    re.I,
)
DATA_QUALITY_RE = re.compile(
    r"\b(stale|missing|gap|data quality|not updating|sensor.*(?:wrong|bad|drift))\b", re.I
)
SETPOINT_RE = re.compile(r"\b(setpoint|set point|lower|raise|adjust|change)\b", re.I)
TRIAGE_RE = re.compile(
    r"\b(alarm|trip\w*|fault\w*|why|problem|pressure|vibration|condenser|delta|temperature)\b",
    re.I,
)
AMBIGUOUS_RE = re.compile(
    r"\b(fix it|seems off|something (?:is )?wrong|not working|help|what should i do)\b", re.I
)


class AgentError(RuntimeError):
    """Raised when the runtime cannot produce a safe answer."""


@dataclass
class AgentConfig:
    stale_after_s: float = 7200.0
    max_tool_calls: int = 16
    agent_version: str = "1.0.0"

    def __post_init__(self) -> None:
        if self.stale_after_s <= 0:
            raise ValueError("stale_after_s must be positive")
        if self.max_tool_calls <= 0:
            raise ValueError("max_tool_calls must be positive")


class _Session:
    def __init__(
        self,
        runtime: "AgentRuntime",
        request: AgentRequest,
        *,
        now: datetime,
    ) -> None:
        self.runtime = runtime
        self.request = request
        self.now = now
        self.ledger = EvidenceLedger()
        self.trace: list[ReasoningStep] = []
        self.tool_results: list[ToolResult] = []
        self.citations: list[str] = []
        self.proposed_setpoint: dict[str, Any] | None = None
        self.stale_tags: list[str] = []
        self.untrusted_texts: list[str] = []
        self.injection_attempts: list[str] = []
        self.safety_critical = False
        self.context = PolicyContext(
            role=request.role,
            site_id=request.site_id,
            session_id=str(request.context.get("session_id", request.request_id)),
            dry_run=bool(request.context.get("dry_run", True)),
            now=now,
            actor=request.context.get("actor", "agent"),
        )

    def call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        thought: str,
        rationale: str = "",
    ) -> ToolResult:
        if len(self.tool_results) >= self.runtime.config.max_tool_calls:
            result = ToolResult(
                call=ToolCall(name, arguments, rationale),
                ok=False,
                error="tool-call budget exhausted",
                error_code="budget_exhausted",
            )
            self.trace.append(
                ReasoningStep(
                    step=len(self.trace) + 1,
                    thought=thought,
                    action=name,
                    observation="skipped: tool-call budget exhausted",
                )
            )
            return result
        call = ToolCall(name=name, arguments=arguments, rationale=rationale)
        result = self.runtime.executor.execute(call, self.context)
        self.tool_results.append(result)
        observation = self._observation(result)
        self.trace.append(
            ReasoningStep(
                step=len(self.trace) + 1,
                thought=thought,
                action=f"{name}({stable_json(arguments)})",
                observation=observation,
            )
        )
        return result

    @staticmethod
    def _observation(result: ToolResult) -> str:
        if not result.ok:
            return f"error[{result.error_code}]: {result.error or 'unknown error'}"
        summary = redact_secrets(stable_json(result.output))
        return summary[:400] + ("..." if len(summary) > 400 else "")

    def evidence(
        self,
        kind: str,
        source: str,
        statement: str,
        *,
        value: float | str | None = None,
        unit: str | None = None,
        timestamp: str | None = None,
        confidence: float = 1.0,
    ) -> Evidence:
        item = self.ledger.add(
            kind,
            source,
            statement,
            value=value,
            unit=unit,
            timestamp=timestamp,
            confidence=confidence,
        )
        if self.trace and item.id not in self.trace[-1].evidence_ids:
            self.trace[-1].evidence_ids.append(item.id)
        return item

    def cite(self, *items: Evidence | None) -> None:
        for item in items:
            if item is not None and item.id not in self.citations:
                self.citations.append(item.id)


class AgentRuntime:
    """Intent-routed industrial agent with policy-checked tools and guardrails."""

    def __init__(
        self,
        *,
        ontology: Ontology,
        telemetry: TagStore,
        alarms: AlarmStore,
        retriever: HybridRetriever,
        executor: ToolExecutor,
        prompts: PromptLibrary,
        clock: Callable[[], datetime] = now_utc,
        config: AgentConfig | None = None,
        audit: AuditLog | None = None,
    ) -> None:
        self.ontology = ontology
        self.telemetry = telemetry
        self.alarms = alarms
        self.retriever = retriever
        self.executor = executor
        self.prompts = prompts
        self.clock = clock
        self.config = config or AgentConfig()
        self.audit = audit or executor.audit
        self.derived = DerivedTagEngine(ontology, telemetry)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, request: AgentRequest) -> AgentAnswer:
        started = time.perf_counter()
        now = self.clock()
        session = _Session(self, request, now=now)
        intent = self.detect_intent(request)
        try:
            answer = self._dispatch(session, intent)
        except Exception as exc:  # noqa: BLE001 - never leak internals to the caller
            self.audit.append(
                "agent",
                "agent.error",
                {"request_id": request.request_id, "error": str(exc)[:500]},
            )
            answer = AgentAnswer(
                request_id=request.request_id,
                status="error",
                summary="I could not complete the investigation safely; no actions were taken.",
                confidence=0.0,
                escalated=False,
                intent=intent,
                agent_version=self.config.agent_version,
            )
        self._apply_disclosures(session, answer)
        answer.intent = intent
        answer.tool_calls = [
            {
                "name": result.call.name,
                "arguments": result.call.arguments,
                "ok": result.ok,
                "error_code": result.error_code,
                "simulated": result.simulated,
                "duration_ms": round(result.duration_ms, 3),
            }
            for result in session.tool_results
        ]
        answer.trace = session.trace
        answer.evidence = session.ledger.to_list()
        if answer.diagnosis and not answer.citations:
            answer.citations = list(session.ledger.ids)
        guardrail_context = GuardrailContext(
            ledger=session.ledger,
            ontology=self.ontology,
            proposed_setpoint=session.proposed_setpoint,
            stale_tags=session.stale_tags,
            untrusted_texts=session.untrusted_texts,
            safety_critical=session.safety_critical,
            executed_writes=[
                {
                    "name": result.call.name,
                    "risk": self._risk(result.call.name),
                    "simulated": result.simulated,
                }
                for result in session.tool_results
                if result.ok
            ],
        )
        verdict = evaluate_guardrails(answer, guardrail_context)
        answer.guardrails = verdict.to_dict()
        if not verdict.passed:
            answer = self._safe_fallback(session, answer, verdict)
            answer.guardrails = verdict.to_dict()
        answer.duration_ms = (time.perf_counter() - started) * 1000
        return answer

    def ask(self, question: str, *, site_id: str, role: str = "operator") -> AgentAnswer:
        return self.run(AgentRequest(question=question, site_id=site_id, role=role))

    # ------------------------------------------------------------------
    # Intent routing
    # ------------------------------------------------------------------

    def detect_intent(self, request: AgentRequest) -> str:
        if request.intent != "auto":
            if request.intent not in INTENTS:
                raise AgentError(f"unsupported intent: {request.intent}")
            return request.intent
        question = request.question
        if detect_injection(question):
            return "prompt_injection"
        if SETPOINT_RE.search(question) and (
            TEMPERATURE_RE.search(question) or self._find_setpoint(question) is not None
        ):
            return "setpoint"
        if FLOOD_RE.search(question):
            return "flood"
        if DATA_QUALITY_RE.search(question):
            return "data_quality"
        if EFFICIENCY_RE.search(question):
            return "efficiency"
        if TRIAGE_RE.search(question):
            return "triage"
        if AMBIGUOUS_RE.search(question) or len(question.split()) < 5:
            return "ambiguous"
        return "unsupported"

    def _dispatch(self, session: _Session, intent: str) -> AgentAnswer:
        handlers: dict[str, Callable[[_Session, AgentRequest], AgentAnswer]] = {
            "triage": self._triage,
            "flood": self._flood,
            "efficiency": self._efficiency,
            "setpoint": self._setpoint,
            "data_quality": self._data_quality,
            "ambiguous": self._clarify,
            "prompt_injection": self._prompt_injection,
            "unsupported": self._clarify,
        }
        return handlers[intent](session, session.request)

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------

    def _resolve_equipment(self, text: str) -> OntologyNode | None:
        node = self.ontology.find_in_text(text, node_type="equipment")
        if node is not None:
            return node
        node = self.ontology.find_in_text(text, node_type="sensor")
        if node is not None:
            return self.ontology.equipment_for_tag(node.id)
        return None

    def _find_setpoint(self, text: str) -> OntologyNode | None:
        node = self.ontology.find_in_text(text, node_type="setpoint")
        if node is not None:
            return node
        equipment = self._resolve_equipment(text)
        if equipment is None:
            return None
        candidates = [
            setpoint
            for setpoint in self.ontology.find(node_type="setpoint")
            if setpoint.attributes.get("equipment_id") == equipment.id
        ]
        return candidates[0] if len(candidates) == 1 else None

    def _sensor(self, equipment_id: str, suffix: str) -> OntologyNode | None:
        for sensor in self.ontology.sensors_for(equipment_id):
            if sensor.id.endswith(suffix):
                return sensor
        return None

    def _site_equipment(self, site_id: str) -> list[OntologyNode]:
        return self.ontology.equipment(site_id)

    def _risk(self, tool_name: str) -> str:
        if not self.executor.registry.has(tool_name):
            return "unknown"
        return self.executor.registry.get(tool_name).risk

    def _wrong_site(self, session: _Session, equipment: OntologyNode) -> AgentAnswer | None:
        if equipment.site_id and equipment.site_id != session.request.site_id:
            return AgentAnswer(
                request_id=session.request.request_id,
                status="refused",
                summary=(
                    "That asset belongs to a different site than the one in this request, "
                    "so I will not mix facility context."
                ),
                recommendations=["Repeat the request with the correct site_id."],
                confidence=0.9,
                escalated=False,
                intent="triage",
                agent_version=self.config.agent_version,
            )
        return None

    # ------------------------------------------------------------------
    # Evidence helpers
    # ------------------------------------------------------------------

    def _series_evidence(
        self,
        session: _Session,
        output: dict[str, Any],
        *,
        label: str | None = None,
    ) -> tuple[Evidence | None, dict[str, Any]]:
        summary = output.get("summary", {})
        tag = output.get("tag", "")
        latest = summary.get("latest")
        unit = output.get("unit", "")
        if latest is None:
            return None, summary
        node = self.ontology.try_resolve(tag)
        name = label or (node.name if node else tag)
        evidence = session.evidence(
            "telemetry",
            tag,
            f"{name} latest reading is {latest} {unit} at {summary.get('latest_at')}.",
            value=float(latest),
            unit=unit,
            timestamp=summary.get("latest_at"),
        )
        if output.get("stale"):
            if tag not in session.stale_tags:
                session.stale_tags.append(tag)
        gaps = output.get("gaps", [])
        if gaps:
            session.evidence(
                "data_quality",
                tag,
                f"{name} has {len(gaps)} gap(s) in the requested window.",
                value=float(len(gaps)),
                unit="gaps",
            )
        anomalies = output.get("anomalies", [])
        if anomalies:
            worst = max(anomalies, key=lambda item: item["score"])
            session.evidence(
                "anomaly",
                tag,
                (
                    f"{name} has {len(anomalies)} robust anomaly signal(s); the strongest is "
                    f"{worst['value']} {unit} against an expected {worst['expected']} {unit}."
                ),
                value=float(worst["value"]),
                unit=unit,
                timestamp=worst.get("timestamp"),
            )
        return evidence, summary

    def _kpi_evidence(
        self, session: _Session, output: dict[str, Any]
    ) -> tuple[Evidence | None, dict[str, Any]]:
        summary = output.get("summary", {})
        latest = summary.get("latest")
        if latest is None:
            return None, summary
        kpi = str(output.get("kpi", ""))
        unit = str(output.get("unit", ""))
        tag = str(output.get("tag", ""))
        evidence = session.evidence(
            "computation",
            tag,
            f"{kpi} is {latest} {unit} at the latest aligned sample.",
            value=float(latest),
            unit=unit,
            timestamp=summary.get("latest_at"),
        )
        anomalies = output.get("anomalies", [])
        if anomalies:
            session.evidence(
                "anomaly",
                tag,
                f"{kpi} has {len(anomalies)} anomaly signal(s) in the recent window.",
                value=float(len(anomalies)),
                unit="anomalies",
            )
        return evidence, summary

    def _knowledge_evidence(self, session: _Session, output: dict[str, Any]) -> Evidence | None:
        hits = output.get("hits", [])
        if not hits:
            return None
        top = hits[0]
        text = str(top.get("text", ""))
        if top.get("trust") == "untrusted":
            session.untrusted_texts.append(text)
            for attempt in detect_injection(text):
                if attempt not in session.injection_attempts:
                    session.injection_attempts.append(attempt)
        snippet = " ".join(text.split())[:180]
        return session.evidence(
            "knowledge",
            str(top.get("chunk_id", top.get("doc_id", "knowledge"))),
            f"{top.get('title', 'Reference')}: {snippet}",
            confidence=0.9 if top.get("trust") == "reference" else 0.4,
        )

    # ------------------------------------------------------------------
    # Intent handlers
    # ------------------------------------------------------------------

    def _clarify(self, session: _Session, request: AgentRequest) -> AgentAnswer:
        equipment = self._resolve_equipment(request.question)
        if EQUIPMENT_HINT_RE.search(request.question) and equipment is None:
            summary = (
                "I could not match that asset label to the facility ontology, so I will not "
                "guess which equipment you mean."
            )
            recommendations = [
                "Paste the exact asset id or the point name from the alarm or trend view.",
                "Include the alarm code and the approximate time of onset.",
            ]
        else:
            summary = (
                "I need a specific asset and symptom before I can investigate safely. "
                "Which equipment, and what changed?"
            )
            recommendations = [
                "Name the equipment (for example, the chiller or air-handling unit id).",
                "Share the alarm code, tag, or the KPI that looks wrong.",
                "State whether you want a diagnosis, an efficiency review, or a setpoint proposal.",
            ]
        return AgentAnswer(
            request_id=request.request_id,
            status="needs_clarification",
            summary=summary,
            recommendations=recommendations,
            confidence=0.2,
            escalated=False,
            intent="ambiguous",
            agent_version=self.config.agent_version,
        )

    def _prompt_injection(self, session: _Session, request: AgentRequest) -> AgentAnswer:
        result = session.call(
            "search_knowledge",
            {"query": "prompt injection untrusted operational text data", "limit": 3},
            thought="Check the security policy before responding to instruction-like text.",
        )
        policy_evidence = self._knowledge_evidence(session, result.output) if result.ok else None
        attempts = detect_injection(request.question)
        evidence = session.evidence(
            "policy",
            "security.prompt_injection",
            "Instruction-like operational text is treated as untrusted data and is never executed.",
            confidence=0.95,
        )
        session.cite(evidence, policy_evidence)
        summary = (
            "The request contains instruction-like text aimed at the control system. "
            "I treated it as untrusted data and did not act on it."
        )
        diagnosis = (
            "Instruction-like text cannot override the tool policy or safety interlocks; "
            "no setpoint tool was called."
        )
        if attempts:
            diagnosis += " The suspicious phrase was quarantined for review."
        return AgentAnswer(
            request_id=request.request_id,
            status="escalated",
            summary=summary,
            diagnosis=diagnosis,
            recommendations=[
                "Verify the author and source of the note before any maintenance action.",
                "Report the note through the security incident channel.",
                "Restate the operational objective without embedded instructions.",
                "Route the note for human review before any maintenance action.",
            ],
            confidence=0.95,
            escalated=True,
            escalation_reason="Potential prompt-injection attempt in operational text requires human review.",
            citations=session.citations,
            intent="prompt_injection",
            agent_version=self.config.agent_version,
        )

    def _triage(self, session: _Session, request: AgentRequest) -> AgentAnswer:
        equipment = self._resolve_equipment(request.question)
        if equipment is None:
            if EQUIPMENT_HINT_RE.search(request.question):
                return self._clarify(session, request)
            return self._flood(session, request)
        wrong_site = self._wrong_site(session, equipment)
        if wrong_site is not None:
            return wrong_site
        site_id = equipment.site_id or request.site_id

        history = session.call(
            "get_alarm_history",
            {"site_id": site_id, "since_hours": 24, "active_only": False},
            thought="Establish the recent alarm context before diagnosing one asset.",
            rationale="Alarm history separates a single fault from a site-wide pattern.",
        )
        triage_result = session.call(
            "triage_alarms",
            {
                "site_id": site_id,
                "window_minutes": 30,
                "root_cause_equipment": equipment.id,
            },
            thought="Cluster and rank alarms, then rank nearby root-cause candidates.",
            rationale="Repeated codes should be deduplicated before dispatch.",
        )
        triage_output = triage_result.output if triage_result.ok else {}
        ranked = triage_output.get("ranked", [])
        top = ranked[0] if ranked else {}
        evidence_items: list[Evidence] = []
        if triage_output:
            evidence_items.append(
                session.evidence(
                    "alarm",
                    f"triage:{site_id}",
                    f"There are {triage_output.get('total_alarms', 0)} alarms in the recent window "
                    f"with a flood index of {triage_output.get('flood_index', 0)}.",
                    value=float(triage_output.get("total_alarms", 0)),
                    unit="alarms",
                )
            )
            if top:
                evidence_items.append(
                    session.evidence(
                        "alarm",
                        f"cluster:{top.get('equipment_id')}:{top.get('code')}",
                        f"Top cluster {top.get('code')} has priority {top.get('priority_score')} "
                        f"and {top.get('count')} occurrence(s).",
                        value=float(top.get("priority_score", 0)),
                        unit="priority",
                    )
                )
        root_causes = triage_output.get("root_causes", [])
        root_evidence = None
        if root_causes:
            candidate = root_causes[0]
            root_evidence = session.evidence(
                "computation",
                f"root-cause:{candidate['equipment_id']}",
                f"Top root-cause candidate is {candidate['name']} with score {candidate['score']}.",
                value=float(candidate["score"]),
                unit="score",
            )

        pressure_node = self._sensor(equipment.id, "condenser_pressure")
        return_node = self._sensor(equipment.id, "chw_return_temp")
        supply_node = self._sensor(equipment.id, "chw_supply_temp")
        pressure_value: float | None = None
        pressure_limit: float | None = None
        delta_value: float | None = None
        cop_value: float | None = None
        approach_value: float | None = None
        pressure_evidence = None

        if pressure_node is not None:
            result = session.call(
                "get_tag_series",
                {"tag": pressure_node.id, "max_points": 48},
                thought="Quantify the condenser-pressure excursion against its alarm limit.",
            )
            if result.ok:
                pressure_evidence, series_summary = self._series_evidence(session, result.output)
                pressure_value = series_summary.get("latest")
                evidence_items.append(pressure_evidence) if pressure_evidence else None
            definition = next(
                (
                    item
                    for item in self.ontology.alarm_definitions(site_id)
                    if item.attributes.get("tag") == pressure_node.id
                ),
                None,
            )
            if definition is not None:
                pressure_limit = float(definition.attributes["threshold"])
                evidence_items.append(
                    session.evidence(
                        "policy",
                        definition.id,
                        f"The high-condenser-pressure alarm threshold is {pressure_limit} "
                        f"{pressure_node.attributes.get('unit', '')}.",
                        value=pressure_limit,
                        unit=str(pressure_node.attributes.get("unit", "")),
                    )
                )

        if return_node is not None and supply_node is not None:
            result = session.call(
                "get_tag_series",
                {"tag": return_node.id, "max_points": 48},
                thought="Check return-water temperature for a low delta-T pattern.",
            )
            if result.ok:
                return_evidence, _ = self._series_evidence(session, result.output)
                if return_evidence:
                    evidence_items.append(return_evidence)
            result = session.call(
                "get_tag_series",
                {"tag": supply_node.id, "max_points": 48},
                thought="Check supply-water temperature for sensor or control drift.",
            )
            if result.ok:
                supply_evidence, _ = self._series_evidence(session, result.output)
                if supply_evidence:
                    evidence_items.append(supply_evidence)
            kpi = session.call(
                "compute_kpi",
                {"equipment_id": equipment.id, "kpi": "delta_t"},
                thought="Compute chilled-water delta-T from aligned telemetry.",
            )
            if kpi.ok:
                delta_evidence, kpi_summary = self._kpi_evidence(session, kpi.output)
                delta_value = kpi_summary.get("latest")
                if delta_evidence:
                    evidence_items.append(delta_evidence)

        cop_result = session.call(
            "compute_kpi",
            {"equipment_id": equipment.id, "kpi": "cop"},
            thought="Compare current efficiency with the expected operating envelope.",
        )
        if cop_result.ok:
            cop_evidence, cop_summary = self._kpi_evidence(session, cop_result.output)
            cop_value = cop_summary.get("latest")
            if cop_evidence:
                evidence_items.append(cop_evidence)
        approach_result = session.call(
            "compute_kpi",
            {"equipment_id": equipment.id, "kpi": "approach"},
            thought="Check the condenser approach proxy for heat-transfer degradation.",
        )
        if approach_result.ok:
            approach_evidence, approach_summary = self._kpi_evidence(session, approach_result.output)
            approach_value = approach_summary.get("latest")
            if approach_evidence:
                evidence_items.append(approach_evidence)

        query = f"{equipment.name} high condenser pressure cooling tower fouling low delta t"
        knowledge = session.call(
            "search_knowledge",
            {"query": query, "limit": 3},
            thought="Retrieve the relevant maintenance playbook before recommending checks.",
        )
        knowledge_evidence = self._knowledge_evidence(session, knowledge.output) if knowledge.ok else None
        if knowledge_evidence:
            evidence_items.append(knowledge_evidence)

        session.cite(*evidence_items)
        name = equipment.name
        over_limit = (
            pressure_value is not None
            and pressure_limit is not None
            and pressure_value > pressure_limit
        )
        low_delta = delta_value is not None and delta_value < 4.0
        if over_limit:
            pieces = [
                f"{name} is above its high-condenser-pressure limit "
                f"({pressure_value} vs {pressure_limit} "
                f"{pressure_node.attributes.get('unit', '') if pressure_node else ''})"
            ]
            if approach_value is not None:
                pieces.append(f"with an elevated condenser approach proxy of {approach_value} degC")
            if cop_value is not None:
                pieces.append(f"and a COP of {cop_value}")
            if low_delta:
                pieces.append(f"alongside a low chilled-water delta-T of {delta_value} degC")
            diagnosis = (
                ", ".join(pieces)
                + ". This coherent pattern is most consistent with degraded condenser heat "
                "rejection (fouling or cooling-tower airflow loss) rather than a pure load change."
            )
            answer_summary = (
                f"{name} is likely limited by condenser heat rejection. Treat the high-pressure "
                "alarm as real and investigate the tower and condenser before resetting it."
            )
            recommendations = [
                "Verify cooling-tower fan operation, condenser-water flow, and air recirculation.",
                "Trend condenser approach and kW/ton through the next operating cycle.",
                "Ask an engineer to inspect the condenser for fouling and confirm transmitter calibration.",
            ]
            confidence = 0.84 if low_delta and cop_value is not None else 0.76
            session.safety_critical = True
            escalated = True
            reason = (
                "High condenser pressure on a high-criticality chiller requires human "
                "acknowledgement before any reset or setpoint change."
            )
            status = "escalated"
        elif low_delta:
            diagnosis = (
                f"{name} shows a low chilled-water delta-T of {delta_value} degC"
                + (f" and a COP of {cop_value}" if cop_value is not None else "")
                + ", which is consistent with low delta-T syndrome (coil fouling, valve "
                "authority, flow imbalance, or sensor calibration)."
            )
            answer_summary = (
                f"{name} is likely suffering from low delta-T syndrome rather than a capacity "
                "shortage; verify flow and coil heat transfer before changing setpoints."
            )
            recommendations = [
                "Check chilled-water valve authority and coil fouling.",
                "Verify flow balance and supply/return sensor calibration.",
                "Review the recent alarm clusters for a related condenser or tower issue.",
            ]
            confidence = 0.72
            escalated = bool(triage_output.get("escalation_recommended", False))
            reason = (
                "; ".join(triage_output.get("escalation_reasons", [])) if escalated else None
            )
            status = "escalated" if escalated else "answered"
        else:
            diagnosis = (
                f"{name} has {triage_output.get('total_alarms', 0)} recent alarm(s) but no single "
                "confirmed telemetry signature. The available evidence does not support a "
                "specific root cause."
            )
            answer_summary = (
                f"{name} needs a human review: alarm context is present, but telemetry does "
                "not confirm a root cause."
            )
            recommendations = [
                "Inspect the raw trend around the first alarm onset.",
                "Verify the sensor calibration and alarm threshold configuration.",
                "Escalate to a controls engineer if the alarm repeats.",
            ]
            confidence = 0.5
            escalated = True
            reason = "Evidence is insufficient for a confident diagnosis."
            status = "escalated"
            session.safety_critical = bool(triage_output.get("escalation_recommended", False))
        if root_evidence is not None and root_causes:
            diagnosis += f" The strongest related asset candidate is {root_causes[0]['name']}."
        return AgentAnswer(
            request_id=request.request_id,
            status=status,
            summary=answer_summary,
            diagnosis=diagnosis,
            recommendations=recommendations,
            confidence=confidence,
            escalated=escalated,
            escalation_reason=reason,
            citations=session.citations,
            intent="triage",
            agent_version=self.config.agent_version,
        )

    def _flood(self, session: _Session, request: AgentRequest) -> AgentAnswer:
        site_id = request.site_id
        history = session.call(
            "get_alarm_history",
            {"site_id": site_id, "since_hours": 24, "active_only": False},
            thought="Measure alarm volume before prioritising individual codes.",
        )
        triage_result = session.call(
            "triage_alarms",
            {"site_id": site_id, "window_minutes": 30},
            thought="Cluster duplicate alarms and compute the flood index.",
        )
        if not triage_result.ok:
            return AgentAnswer(
                request_id=request.request_id,
                status="error",
                summary="Alarm triage is unavailable, so I cannot rank the flood safely.",
                confidence=0.0,
                escalated=True,
                escalation_reason="Alarm triage tool failed.",
                intent="flood",
                agent_version=self.config.agent_version,
            )
        output = triage_result.output
        ranked = output.get("ranked", [])
        top = ranked[0] if ranked else {}
        evidence = session.evidence(
            "alarm",
            f"triage:{site_id}",
            (
                f"The window contains {output.get('total_alarms', 0)} alarms across "
                f"{len(output.get('clusters', []))} clusters."
            ),
            value=float(output.get("total_alarms", 0)),
            unit="alarms",
        )
        flood_evidence = session.evidence(
            "computation",
            f"flood-index:{site_id}",
            f"The flood index is {output.get('flood_index', 0)}.",
            value=float(output.get("flood_index", 0)),
            unit="index",
        )
        session.cite(evidence, flood_evidence)
        if top:
            top_evidence = session.evidence(
                "alarm",
                f"cluster:{top.get('equipment_id')}:{top.get('code')}",
                (
                    f"The top cluster is {top.get('code')} on {top.get('equipment_id')} "
                    f"with priority {top.get('priority_score')} and {top.get('count')} "
                    f"occurrence(s)."
                ),
                value=float(top.get("priority_score", 0)),
                unit="priority",
            )
            session.cite(top_evidence)
        if ranked:
            cluster_evidence = session.evidence(
                "alarm",
                f"clusters:{site_id}",
                f"There are {len(output.get('clusters', []))} deduplicated alarm clusters.",
                value=float(len(output.get("clusters", []))),
                unit="clusters",
            )
            session.cite(cluster_evidence)
        if top:
            diagnosis = (
                f"The recent window contains {output.get('total_alarms', 0)} alarms across "
                f"{len(output.get('clusters', []))} clusters "
                f"(flood index {output.get('flood_index', 0)}). "
                f"The highest-priority cluster is {top.get('code')} on "
                f"{self._asset_label(top.get('equipment_id'))} at "
                f"{top.get('priority_score')}, so repeated low-value codes should be "
                f"deduplicated before dispatch."
            )
        else:
            diagnosis = "No alarm clusters were found in the requested window."
        reasons = output.get("escalation_reasons", [])
        escalated = bool(output.get("escalation_recommended", False))
        checks = output.get("next_checks", [])
        recommendations = [
            "Acknowledge and deduplicate repeated codes before assigning work.",
            *(checks[:3] or ["Verify the top cluster against raw telemetry."]),
            "Escalate any safety-critical or high-pressure condition to the duty engineer.",
        ]
        return AgentAnswer(
            request_id=request.request_id,
            status="escalated" if escalated else "answered",
            summary=(
                "The site is in an alarm flood: deduplicate repeated codes and work the "
                "highest-priority asset first."
                if escalated
                else "Alarm volume is elevated but manageable; work the ranked clusters in order."
            ),
            diagnosis=diagnosis,
            recommendations=recommendations,
            confidence=0.82,
            escalated=escalated,
            escalation_reason="; ".join(reasons) if reasons else None,
            citations=session.citations,
            intent="flood",
            agent_version=self.config.agent_version,
        )

    def _efficiency(self, session: _Session, request: AgentRequest) -> AgentAnswer:
        equipment = self._resolve_equipment(request.question)
        if equipment is None:
            candidates = [item for item in self._site_equipment(request.site_id) if item.subtype == "centrifugal_chiller"]
            equipment = candidates[0] if candidates else None
        if equipment is None:
            return self._clarify(session, request)
        wrong_site = self._wrong_site(session, equipment)
        if wrong_site is not None:
            return wrong_site
        evidence_items: list[Evidence] = []
        values: dict[str, float] = {}
        stale = False
        for kpi in ("cop", "kw_per_ton", "delta_t"):
            result = session.call(
                "compute_kpi",
                {"equipment_id": equipment.id, "kpi": kpi},
                thought=f"Quantify {kpi} from aligned telemetry.",
                rationale="Efficiency claims must be computed, not inferred from labels.",
            )
            if result.ok:
                item, summary = self._kpi_evidence(session, result.output)
                if item:
                    evidence_items.append(item)
                if summary.get("latest") is not None:
                    values[kpi] = float(summary["latest"])
                if result.output.get("stale"):
                    stale = True
                    tag = str(result.output.get("tag", ""))
                    if tag not in session.stale_tags:
                        session.stale_tags.append(tag)
        knowledge = session.call(
            "search_knowledge",
            {"query": "chiller efficiency kw per ton cop condenser fouling", "limit": 3},
            thought="Retrieve the efficiency playbook for context.",
        )
        knowledge_evidence = self._knowledge_evidence(session, knowledge.output) if knowledge.ok else None
        if knowledge_evidence:
            evidence_items.append(knowledge_evidence)
        if not values:
            return AgentAnswer(
                request_id=request.request_id,
                status="refused",
                summary=(
                    f"{equipment.name} has no aligned telemetry for an efficiency calculation, "
                    "so I will not estimate one."
                ),
                recommendations=["Restore telemetry or widen the time window, then re-run the review."],
                confidence=0.8,
                escalated=False,
                intent="efficiency",
                agent_version=self.config.agent_version,
            )
        session.cite(*evidence_items)
        pieces = []
        if "cop" in values:
            pieces.append(f"a COP of {values['cop']}")
        if "kw_per_ton" in values:
            pieces.append(f"{values['kw_per_ton']} kW/ton")
        if "delta_t" in values:
            pieces.append(f"a chilled-water delta-T of {values['delta_t']} degC")
        diagnosis = (
            f"{equipment.name} is currently operating at " + ", ".join(pieces) + "."
        )
        if values.get("kw_per_ton", 0) >= 0.95:
            diagnosis += (
                " That is above the expected efficiency envelope and is consistent with "
                "condenser fouling, low delta-T syndrome, or an unfavorable staging decision."
            )
        else:
            diagnosis += (
                " That is within the expected efficiency envelope; no efficiency alarm is "
                "justified from the current snapshot."
            )
        if stale:
            diagnosis = "The available telemetry is stale, so this assessment is uncertain. " + diagnosis
        return AgentAnswer(
            request_id=request.request_id,
            status="answered",
            summary=(
                f"{equipment.name} efficiency is "
                + ("below the expected envelope." if values.get("kw_per_ton", 0) >= 0.95 else "within the expected envelope.")
            ),
            diagnosis=diagnosis,
            recommendations=[
                "Compare kW/ton against the commissioning baseline at the same load.",
                "Verify condenser cleanliness and cooling-tower airflow.",
                "Review chilled-water delta-T and valve authority before changing setpoints.",
            ],
            confidence=0.55 if stale else 0.8,
            escalated=False,
            citations=session.citations,
            intent="efficiency",
            agent_version=self.config.agent_version,
        )

    def _setpoint(self, session: _Session, request: AgentRequest) -> AgentAnswer:
        setpoint = self._find_setpoint(request.question)
        temperature = SETPOINT_TARGET_RE.search(request.question) or TEMPERATURE_RE.search(
            request.question
        )
        if setpoint is None:
            return self._clarify(session, request)
        wrong_site = self._wrong_site(session, setpoint)
        if wrong_site is not None:
            return wrong_site
        if temperature is None:
            return AgentAnswer(
                request_id=request.request_id,
                status="needs_clarification",
                summary=(
                    f"I can evaluate a change to {setpoint.name}, but I need the target value "
                    "and unit before proposing anything."
                ),
                recommendations=["Restate the request with a target value in degC."],
                confidence=0.2,
                escalated=False,
                intent="setpoint",
                agent_version=self.config.agent_version,
            )
        requested = float(temperature.group(1))
        result = session.call(
            "propose_setpoint",
            {"tag": setpoint.id, "value": requested},
            thought="Validate the requested value against configured operating bounds and deadband.",
            rationale="Setpoint proposals must be checked before a human sees them.",
        )
        if not result.ok:
            return AgentAnswer(
                request_id=request.request_id,
                status="refused",
                summary="The setpoint proposal could not be validated, so I will not suggest a value.",
                recommendations=["Ask an engineer to review the setpoint configuration."],
                confidence=0.7,
                escalated=True,
                escalation_reason="Setpoint validation tool failed.",
                intent="setpoint",
                agent_version=self.config.agent_version,
            )
        output = result.output
        unit = str(output.get("unit", "degC"))
        current = float(output.get("current", 0.0))
        minimum = float(output.get("min", 0.0))
        maximum = float(output.get("max", 0.0))
        deadband = float(output.get("deadband", 0.0))
        bounds_evidence = session.evidence(
            "policy",
            f"{setpoint.id}:min",
            f"The configured minimum for {setpoint.name} is {minimum} {unit}.",
            value=minimum,
            unit=unit,
        )
        current_evidence = session.evidence(
            "policy",
            f"{setpoint.id}:current",
            f"The current value of {setpoint.name} is {current} {unit}.",
            value=current,
            unit=unit,
        )
        maximum_evidence = session.evidence(
            "policy",
            f"{setpoint.id}:max",
            f"The configured maximum for {setpoint.name} is {maximum} {unit}.",
            value=maximum,
            unit=unit,
        )
        deadband_evidence = session.evidence(
            "policy",
            f"{setpoint.id}:deadband",
            f"The configured deadband for {setpoint.name} is {deadband} {unit}.",
            value=deadband,
            unit=unit,
        )
        request_evidence = session.evidence(
            "request",
            request.request_id,
            f"The requested value is {requested} {unit}.",
            value=requested,
            unit=unit,
        )
        session.cite(
            bounds_evidence,
            current_evidence,
            maximum_evidence,
            deadband_evidence,
            request_evidence,
        )
        if output.get("accepted"):
            session.proposed_setpoint = {
                "tag": setpoint.id,
                "value": requested,
                "min": minimum,
                "max": maximum,
                "default": current,
            }
            return AgentAnswer(
                request_id=request.request_id,
                status="escalated",
                summary=(
                    f"A change to {requested} {unit} is inside the operating envelope, but it "
                    "is a proposal only and has not been applied."
                ),
                diagnosis=(
                    f"{setpoint.name} is currently {current} {unit} with an allowed range of "
                    f"{minimum}-{maximum} {unit} and a {deadband} {unit} deadband. The requested "
                    f"value passes the bounds and deadband checks; a physical write still "
                    "requires engineer approval and an explicit confirmation token."
                ),
                recommendations=[
                    "Have an engineer review the proposal against current load and delta-T.",
                    "Apply the change through the approved control path with a confirmation token.",
                    "Verify the resulting delta-T and power draw after the change.",
                ],
                confidence=0.9,
                escalated=True,
                escalation_reason=(
                    "Physical setpoint changes require engineer approval and an explicit "
                    "confirmation token; no write was executed."
                ),
                citations=session.citations,
                intent="setpoint",
                agent_version=self.config.agent_version,
            )
        reason = str(output.get("reason", "The requested value failed validation."))
        return AgentAnswer(
            request_id=request.request_id,
            status="escalated",
            summary=f"I cannot propose {requested} {unit} for {setpoint.name}: {reason}",
            diagnosis=(
                f"The requested value {requested} {unit} is outside the safe configuration "
                f"for {setpoint.name} ({minimum}-{maximum} {unit}, {deadband} {unit} deadband). "
                "No setpoint tool was permitted to apply it."
            ),
            recommendations=[
                "Choose a value inside the configured range and re-submit the proposal.",
                "If the range is wrong, route a setpoint-policy change to an engineer.",
                "Do not bypass interlocks or write directly to the controller.",
            ],
            confidence=0.9,
            escalated=True,
            escalation_reason="Requested setpoint violates configured bounds or deadband.",
            citations=session.citations,
            intent="setpoint",
            agent_version=self.config.agent_version,
        )

    def _data_quality(self, session: _Session, request: AgentRequest) -> AgentAnswer:
        equipment = self._resolve_equipment(request.question)
        if equipment is None:
            return self._clarify(session, request)
        wrong_site = self._wrong_site(session, equipment)
        if wrong_site is not None:
            return wrong_site
        sensors = self.ontology.sensors_for(equipment.id)[:3]
        if not sensors:
            return self._clarify(session, request)
        evidence_items: list[Evidence] = []
        stale_count = 0
        gap_count = 0
        for sensor in sensors:
            result = session.call(
                "get_tag_series",
                {"tag": sensor.id, "max_points": 24},
                thought=f"Check freshness and gaps for {sensor.name}.",
                rationale="Data-quality claims must be measured on the raw series.",
            )
            if not result.ok:
                continue
            output = result.output
            item, summary = self._series_evidence(session, output)
            if item:
                evidence_items.append(item)
            if output.get("stale"):
                stale_count += 1
            gap_count += len(output.get("gaps", []))
        if not evidence_items:
            return AgentAnswer(
                request_id=request.request_id,
                status="refused",
                summary="No readable telemetry is available for that asset.",
                recommendations=["Check the data-collection path and re-run the quality check."],
                confidence=0.7,
                escalated=True,
                escalation_reason="All requested telemetry series are unavailable.",
                intent="data_quality",
                agent_version=self.config.agent_version,
            )
        freshness = session.evidence(
            "data_quality",
            f"freshness:{equipment.id}",
            (
                f"{stale_count} of {len(evidence_items)} checked series are stale; "
                f"{gap_count} gap(s) were detected."
            ),
            value=float(stale_count),
            unit="stale_series",
        )
        checked_evidence = session.evidence(
            "data_quality",
            f"checked:{equipment.id}",
            f"{len(evidence_items)} series were checked for freshness and gaps.",
            value=float(len(evidence_items)),
            unit="series",
        )
        gap_evidence = session.evidence(
            "data_quality",
            f"gaps:{equipment.id}",
            f"{gap_count} telemetry gap(s) were detected across the checked series.",
            value=float(gap_count),
            unit="gaps",
        )
        session.cite(*evidence_items, freshness, checked_evidence, gap_evidence)
        if stale_count:
            diagnosis = (
                f"{equipment.name} telemetry is stale: {stale_count} of "
                f"{len(evidence_items)} checked series are outside the freshness window. "
                "Readings are historical context, not a live diagnosis."
            )
            summary = (
                f"{equipment.name} telemetry is stale, so I cannot support a reliable "
                "diagnosis from it."
            )
            confidence = 0.8
        elif gap_count:
            diagnosis = (
                f"{equipment.name} telemetry is fresh but has {gap_count} gap(s); "
                "interpolated conclusions would be unsafe."
            )
            summary = f"{equipment.name} telemetry has gaps that limit diagnosis."
            confidence = 0.75
        else:
            diagnosis = (
                f"{equipment.name} telemetry is fresh and gap-free across the checked series; "
                "no data-quality fault is evident."
            )
            summary = f"{equipment.name} telemetry passes the freshness and gap checks."
            confidence = 0.85
        return AgentAnswer(
            request_id=request.request_id,
            status="answered",
            summary=summary,
            diagnosis=diagnosis,
            recommendations=[
                "Check the gateway and point-mapping health for stale series.",
                "Do not tune controls against stale or gapped data.",
                "Re-run the check after telemetry recovers and compare the new window.",
            ],
            confidence=confidence,
            escalated=False,
            citations=session.citations,
            intent="data_quality",
            agent_version=self.config.agent_version,
        )

    # ------------------------------------------------------------------
    # Finalisation
    # ------------------------------------------------------------------

    def _apply_disclosures(self, session: _Session, answer: AgentAnswer) -> None:
        text = answer.text.lower()
        disclosures: list[str] = []
        if session.stale_tags and not any(
            word in text for word in ("stale", "data quality", "freshness")
        ):
            disclosures.append(
                "Telemetry for the requested asset is older than the freshness window; "
                "treat the readings as historical context."
            )
        if session.injection_attempts and "untrusted" not in text:
            disclosures.append(
                "Untrusted retrieved text contained instruction-like content; it was treated "
                "as data and ignored."
            )
        if disclosures:
            answer.disclosure = " ".join(disclosures)

    def _safe_fallback(
        self, session: _Session, answer: AgentAnswer, verdict: Any
    ) -> AgentAnswer:
        blocked = [finding.rule for finding in verdict.findings if finding.blocking]
        disclosures = []
        if session.stale_tags:
            disclosures.append("Telemetry used for this request is stale.")
        if session.injection_attempts:
            disclosures.append("Untrusted text contained instruction-like content and was ignored.")
        self.audit.append(
            "guardrails",
            "answer.blocked",
            {"request_id": answer.request_id, "rules": blocked},
        )
        return AgentAnswer(
            request_id=answer.request_id,
            status="escalated",
            summary=(
                "I cannot safely answer that request from the available evidence. "
                "No physical action was taken."
            ),
            recommendations=[
                "Route the request to a human reviewer and verify the underlying telemetry.",
                "Check the guardrail report before reusing this answer pattern.",
            ],
            confidence=0.0,
            escalated=True,
            escalation_reason="Output guardrails blocked the generated answer.",
            citations=[],
            tool_calls=answer.tool_calls,
            trace=answer.trace,
            evidence=answer.evidence,
            intent=answer.intent,
            agent_version=answer.agent_version,
            disclosure=" ".join(disclosures) or None,
        )

    def _asset_label(self, equipment_id: str | None) -> str:
        if not equipment_id:
            return "the affected asset"
        node = self.ontology.try_resolve(equipment_id)
        return node.name if node else "the affected asset"


EQUIPMENT_HINT_RE = re.compile(
    r"\b(chiller|ahu|air.handling|pump|cooling.tower|tower|equipment|asset)\b", re.I
)
