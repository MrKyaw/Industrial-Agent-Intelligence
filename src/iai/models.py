"""Core value objects exchanged by the agent, tools, and evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .util import format_timestamp, now_utc, stable_json, fingerprint


@dataclass
class Evidence:
    """A single inspectable fact used to ground an agent answer.

    Evidence ids are short (``E1``) so prose citations stay readable. The
    ``value`` field is what numeric-grounding checks compare against; the
    ``statement`` is human-readable context and is never treated as a source of
    truth on its own.
    """

    id: str
    kind: str
    source: str
    statement: str
    value: float | str | None = None
    unit: str | None = None
    timestamp: str | None = None
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("evidence source is required")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("evidence confidence must be within [0, 1]")

    @property
    def citation(self) -> str:
        return f"[{self.id}]"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "source": self.source,
            "statement": self.statement,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
        }


class EvidenceLedger:
    """Ordered, deduplicating evidence store with stable generated ids."""

    def __init__(self) -> None:
        self._items: list[Evidence] = []
        self._by_id: dict[str, Evidence] = {}
        self._keys: dict[str, str] = {}

    def add(
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
        key = stable_json(
            {
                "kind": kind,
                "source": source,
                "statement": statement,
                "value": value,
                "unit": unit,
                "timestamp": timestamp,
            }
        )
        if key in self._keys:
            return self._by_id[self._keys[key]]
        evidence = Evidence(
            id=f"E{len(self._items) + 1}",
            kind=kind,
            source=source,
            statement=statement,
            value=value,
            unit=unit,
            timestamp=timestamp,
            confidence=confidence,
        )
        self._items.append(evidence)
        self._by_id[evidence.id] = evidence
        self._keys[key] = evidence.id
        return evidence

    def extend(self, evidence: list[Evidence]) -> None:
        for item in evidence:
            self.add(
                item.kind,
                item.source,
                item.statement,
                value=item.value,
                unit=item.unit,
                timestamp=item.timestamp,
                confidence=item.confidence,
            )

    def get(self, evidence_id: str) -> Evidence:
        try:
            return self._by_id[evidence_id]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"unknown evidence id: {evidence_id}") from exc

    def has(self, evidence_id: str) -> bool:
        return evidence_id in self._by_id

    def numeric_values(self) -> list[float]:
        values: list[float] = []
        for item in self._items:
            if isinstance(item.value, (int, float)) and not isinstance(item.value, bool):
                values.append(float(item.value))
        return values

    def render(self) -> str:
        return "\n".join(
            f"[{item.id}] ({item.kind}) {item.statement}" for item in self._items
        )

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, index: int) -> Evidence:
        return self._items[index]

    @property
    def ids(self) -> list[str]:
        return [item.id for item in self._items]

    def to_list(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self._items]


@dataclass
class ToolCall:
    """A requested tool invocation, including the agent's rationale."""

    name: str
    arguments: dict[str, Any]
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments, "rationale": self.rationale}


@dataclass
class ToolResult:
    """Outcome of a policy-checked tool invocation."""

    call: ToolCall
    ok: bool
    output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    error_code: str | None = None
    simulated: bool = False
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "call": self.call.to_dict(),
            "ok": self.ok,
            "output": self.output,
            "error": self.error,
            "error_code": self.error_code,
            "simulated": self.simulated,
            "duration_ms": round(self.duration_ms, 3),
        }


@dataclass
class ReasoningStep:
    """One inspectable step in the agent trace."""

    step: int
    thought: str
    action: str | None = None
    observation: str | None = None
    evidence_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "thought": self.thought,
            "action": self.action,
            "observation": self.observation,
            "evidence_ids": self.evidence_ids,
        }


@dataclass
class AgentRequest:
    """A user request routed to the agent runtime."""

    question: str
    site_id: str
    role: str = "operator"
    intent: str = "auto"
    request_id: str = ""
    urgency: str = "normal"
    context: dict[str, Any] = field(default_factory=dict)
    requested_at: str = ""

    def __post_init__(self) -> None:
        self.question = self.question.strip()
        if not self.question:
            raise ValueError("question must not be empty")
        if not self.site_id:
            raise ValueError("site_id is required")
        if not self.request_id:
            self.request_id = f"req-{fingerprint((self.question, self.site_id))[:10]}"
        if not self.requested_at:
            self.requested_at = format_timestamp(now_utc())

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentRequest":
        if not isinstance(payload, dict):
            raise ValueError("request payload must be an object")
        question = payload.get("question")
        if not isinstance(question, str):
            raise ValueError("question must be a string")
        site_id = payload.get("site_id")
        if not isinstance(site_id, str):
            raise ValueError("site_id must be a string")
        role = payload.get("role", "operator")
        if not isinstance(role, str):
            raise ValueError("role must be a string")
        context = payload.get("context", {})
        if not isinstance(context, dict):
            raise ValueError("context must be an object")
        return cls(
            question=question,
            site_id=site_id,
            role=role,
            intent=str(payload.get("intent", "auto")),
            request_id=str(payload.get("request_id", "")),
            urgency=str(payload.get("urgency", "normal")),
            context=context,
            requested_at=str(payload.get("requested_at", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "site_id": self.site_id,
            "role": self.role,
            "intent": self.intent,
            "request_id": self.request_id,
            "urgency": self.urgency,
            "context": self.context,
            "requested_at": self.requested_at,
        }


@dataclass
class AgentAnswer:
    """Final, auditable agent response."""

    request_id: str
    status: str
    summary: str
    diagnosis: str = ""
    recommendations: list[str] = field(default_factory=list)
    confidence: float = 0.0
    escalated: bool = False
    escalation_reason: str | None = None
    citations: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    trace: list[ReasoningStep] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    guardrails: dict[str, Any] = field(default_factory=dict)
    intent: str = "unknown"
    agent_version: str = "1.0.0"
    duration_ms: float = 0.0
    disclosure: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {
            "answered",
            "needs_clarification",
            "escalated",
            "refused",
            "error",
        }:
            raise ValueError(f"unsupported answer status: {self.status}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")

    @property
    def text(self) -> str:
        parts = [self.summary, self.diagnosis, *self.recommendations]
        if self.escalation_reason:
            parts.append(self.escalation_reason)
        if self.disclosure:
            parts.append(self.disclosure)
        return "\n".join(part for part in parts if part)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "status": self.status,
            "summary": self.summary,
            "diagnosis": self.diagnosis,
            "recommendations": self.recommendations,
            "confidence": self.confidence,
            "escalated": self.escalated,
            "escalation_reason": self.escalation_reason,
            "citations": self.citations,
            "tool_calls": self.tool_calls,
            "trace": [step.to_dict() for step in self.trace],
            "evidence": self.evidence,
            "guardrails": self.guardrails,
            "intent": self.intent,
            "agent_version": self.agent_version,
            "duration_ms": round(self.duration_ms, 3),
            "disclosure": self.disclosure,
        }
