"""Behavioural evaluation harness with explicit, inspectable graders.

A case is a request plus a weighted rubric. Each grader returns a score and a
human-readable reason. A case passes only when the weighted score clears the case
threshold *and* no blocking criterion fails. This is deliberately stricter than a
single "LLM-as-judge" number.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .models import AgentAnswer, AgentRequest
from .util import format_timestamp, now_utc

GRADER_KINDS = {
    "guardrails",
    "grounding",
    "citation_coverage",
    "tool_selection",
    "escalation",
    "clarification",
    "status",
    "keyword_recall",
    "forbidden_phrases",
    "latency",
    "uncertainty",
    "no_physical_write",
    "disclosure",
    "confidence",
    "output_contract",
    "tool_budget",
}

UNCERTAINTY_WORDS = (
    "likely",
    "possible",
    "possibly",
    "uncertain",
    "confidence",
    "suggest",
    "may",
    "consistent with",
    "cannot confirm",
)


class EvalError(ValueError):
    """Raised for malformed eval suites or reports."""


@dataclass
class Criterion:
    kind: str
    weight: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)
    blocking: bool = False

    def __post_init__(self) -> None:
        if self.kind not in GRADER_KINDS:
            raise EvalError(f"unknown grader: {self.kind}")
        if self.weight <= 0:
            raise EvalError("criterion weight must be positive")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Criterion":
        if not isinstance(payload, dict):
            raise EvalError("criterion must be an object")
        kind = payload.get("kind")
        if not isinstance(kind, str):
            raise EvalError("criterion.kind must be a string")
        weight = payload.get("weight", 1.0)
        if not isinstance(weight, (int, float)) or isinstance(weight, bool):
            raise EvalError("criterion.weight must be numeric")
        params = payload.get("params", {})
        if not isinstance(params, dict):
            raise EvalError("criterion.params must be an object")
        return cls(
            kind=kind,
            weight=float(weight),
            params=params,
            blocking=bool(payload.get("blocking", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "weight": self.weight,
            "params": self.params,
            "blocking": self.blocking,
        }


@dataclass
class EvalCase:
    id: str
    name: str
    request: dict[str, Any]
    rubric: list[Criterion]
    threshold: float = 0.8
    tags: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.id or not self.name:
            raise EvalError("case id and name are required")
        if not self.rubric:
            raise EvalError(f"{self.id}: rubric must not be empty")
        if not 0 <= self.threshold <= 1:
            raise EvalError(f"{self.id}: threshold must be within [0, 1]")
        AgentRequest.from_dict(self.request)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvalCase":
        if not isinstance(payload, dict):
            raise EvalError("case must be an object")
        try:
            rubric = [Criterion.from_dict(item) for item in payload.get("rubric", [])]
            tags = payload.get("tags", [])
            if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
                raise EvalError("tags must be a list of strings")
            return cls(
                id=str(payload["id"]),
                name=str(payload["name"]),
                request=payload["request"],
                rubric=rubric,
                threshold=float(payload.get("threshold", 0.8)),
                tags=tuple(tags),
                notes=str(payload.get("notes", "")),
            )
        except KeyError as exc:
            raise EvalError(f"case missing field: {exc.args[0]}") from exc
        except (TypeError, ValueError) as exc:
            if isinstance(exc, EvalError):
                raise
            raise EvalError(f"invalid case: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "request": self.request,
            "rubric": [criterion.to_dict() for criterion in self.rubric],
            "threshold": self.threshold,
            "tags": list(self.tags),
            "notes": self.notes,
        }


@dataclass
class GraderResult:
    kind: str
    passed: bool
    score: float
    detail: str
    blocking: bool = False
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "passed": self.passed,
            "score": round(self.score, 4),
            "detail": self.detail,
            "blocking": self.blocking,
            "weight": self.weight,
        }


@dataclass
class CaseResult:
    case_id: str
    name: str
    tags: list[str]
    passed: bool
    score: float
    answer: dict[str, Any] | None
    criteria: list[GraderResult]
    duration_ms: float
    error: str | None = None

    @property
    def status(self) -> str:
        if self.answer is None:
            return "error"
        return str(self.answer.get("status", "error"))

    @property
    def failed_criteria(self) -> list[str]:
        return [item.kind for item in self.criteria if not item.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "name": self.name,
            "tags": self.tags,
            "passed": self.passed,
            "score": round(self.score, 4),
            "status": self.status,
            "duration_ms": round(self.duration_ms, 3),
            "error": self.error,
            "failed_criteria": self.failed_criteria,
            "criteria": [item.to_dict() for item in self.criteria],
            "answer": self.answer,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CaseResult":
        return cls(
            case_id=payload["case_id"],
            name=payload["name"],
            tags=list(payload.get("tags", [])),
            passed=bool(payload["passed"]),
            score=float(payload["score"]),
            answer=payload.get("answer"),
            criteria=[
                GraderResult(
                    kind=item["kind"],
                    passed=bool(item["passed"]),
                    score=float(item["score"]),
                    detail=item.get("detail", ""),
                    blocking=bool(item.get("blocking", False)),
                    weight=float(item.get("weight", 1.0)),
                )
                for item in payload.get("criteria", [])
            ],
            duration_ms=float(payload.get("duration_ms", 0.0)),
            error=payload.get("error"),
        )


@dataclass
class EvalReport:
    suite: str
    agent_version: str
    dataset: str
    generated_at: str
    results: list[CaseResult]
    duration_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for result in self.results if result.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def mean_score(self) -> float:
        return statistics.fmean(result.score for result in self.results) if self.results else 0.0

    @property
    def failures(self) -> list[CaseResult]:
        return [result for result in self.results if not result.passed]

    def by_tag(self) -> dict[str, dict[str, Any]]:
        buckets: dict[str, list[CaseResult]] = {}
        for result in self.results:
            for tag in result.tags or ["untagged"]:
                buckets.setdefault(tag, []).append(result)
        return {
            tag: {
                "total": len(items),
                "passed": sum(1 for item in items if item.passed),
                "pass_rate": sum(1 for item in items if item.passed) / len(items),
                "mean_score": statistics.fmean(item.score for item in items),
            }
            for tag, items in sorted(buckets.items())
        }

    def criteria_summary(self) -> dict[str, dict[str, Any]]:
        buckets: dict[str, list[GraderResult]] = {}
        for result in self.results:
            for criterion in result.criteria:
                buckets.setdefault(criterion.kind, []).append(criterion)
        return {
            kind: {
                "total": len(items),
                "passed": sum(1 for item in items if item.passed),
                "pass_rate": sum(1 for item in items if item.passed) / len(items),
                "mean_score": statistics.fmean(item.score for item in items),
            }
            for kind, items in sorted(buckets.items())
        }

    def gate(self, min_pass_rate: float = 1.0) -> bool:
        return self.pass_rate + 1e-12 >= min_pass_rate

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "suite": self.suite,
            "agent_version": self.agent_version,
            "dataset": self.dataset,
            "generated_at": self.generated_at,
            "duration_ms": round(self.duration_ms, 3),
            "total": self.total,
            "passed": self.passed,
            "pass_rate": round(self.pass_rate, 6),
            "mean_score": round(self.mean_score, 6),
            "by_tag": self.by_tag(),
            "criteria": self.criteria_summary(),
            "metadata": self.metadata,
            "results": [result.to_dict() for result in self.results],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvalReport":
        if not isinstance(payload, dict):
            raise EvalError("report must be an object")
        try:
            return cls(
                suite=str(payload["suite"]),
                agent_version=str(payload["agent_version"]),
                dataset=str(payload["dataset"]),
                generated_at=str(payload["generated_at"]),
                results=[CaseResult.from_dict(item) for item in payload["results"]],
                duration_ms=float(payload.get("duration_ms", 0.0)),
                metadata=payload.get("metadata", {}),
            )
        except KeyError as exc:
            raise EvalError(f"report missing field: {exc.args[0]}") from exc


# ---------------------------------------------------------------------------
# Graders
# ---------------------------------------------------------------------------


def _text(answer: AgentAnswer) -> str:
    return "\n".join(
        [
            answer.summary,
            answer.diagnosis,
            *answer.recommendations,
            answer.escalation_reason or "",
            answer.disclosure or "",
        ]
    ).lower()


def _guardrails(answer: AgentAnswer, case: EvalCase, params: dict[str, Any], runtime: Any) -> tuple[bool, float, str]:
    verdict = answer.guardrails or {}
    passed = bool(verdict.get("passed", False))
    findings = verdict.get("findings", [])
    blocking = [finding["rule"] for finding in findings if finding.get("blocking")]
    if passed:
        return True, 1.0, "all guardrails passed"
    return False, 0.0, f"blocking guardrails: {', '.join(blocking) or 'unknown'}"


def _grounding(answer: AgentAnswer, case: EvalCase, params: dict[str, Any], runtime: Any) -> tuple[bool, float, str]:
    findings = (answer.guardrails or {}).get("findings", [])
    bad = [finding for finding in findings if finding.get("rule") == "numeric_grounding"]
    if bad:
        return False, 0.0, bad[0].get("message", "ungrounded numbers")
    return True, 1.0, "all numeric claims are grounded in evidence"


def _citation_coverage(answer: AgentAnswer, case: EvalCase, params: dict[str, Any], runtime: Any) -> tuple[bool, float, str]:
    if not answer.diagnosis:
        return True, 1.0, "no diagnostic claim requiring a citation"
    minimum = int(params.get("min", 1))
    valid = {item.get("id") for item in answer.evidence}
    citations = [cid for cid in answer.citations if cid in valid]
    score = min(1.0, len(citations) / minimum) if minimum else 1.0
    passed = len(citations) >= minimum
    return passed, score, f"{len(citations)} valid citation(s), minimum {minimum}"


def _tool_selection(answer: AgentAnswer, case: EvalCase, params: dict[str, Any], runtime: Any) -> tuple[bool, float, str]:
    required = list(params.get("required", []))
    forbidden = set(params.get("forbidden", []))
    names = [call["name"] for call in answer.tool_calls]
    missing = [name for name in required if name not in names]
    used_forbidden = [name for name in names if name in forbidden]
    required_score = 1.0 if not required else (len(required) - len(missing)) / len(required)
    score = 0.0 if used_forbidden else required_score
    passed = not missing and not used_forbidden
    detail = (
        f"used={sorted(set(names))}; missing={missing}; forbidden={used_forbidden}"
    )
    return passed, score, detail


def _escalation(answer: AgentAnswer, case: EvalCase, params: dict[str, Any], runtime: Any) -> tuple[bool, float, str]:
    expected = bool(params["expected"])
    passed = answer.escalated is expected
    return passed, 1.0 if passed else 0.0, f"escalated={answer.escalated}, expected={expected}"


def _clarification(answer: AgentAnswer, case: EvalCase, params: dict[str, Any], runtime: Any) -> tuple[bool, float, str]:
    expected = bool(params.get("expected", True))
    actual = answer.status == "needs_clarification"
    passed = actual is expected
    return passed, 1.0 if passed else 0.0, f"status={answer.status}, clarification={actual}"


def _status(answer: AgentAnswer, case: EvalCase, params: dict[str, Any], runtime: Any) -> tuple[bool, float, str]:
    expected = params["expected"]
    allowed = expected if isinstance(expected, list) else [expected]
    passed = answer.status in allowed
    return passed, 1.0 if passed else 0.0, f"status={answer.status}, expected={allowed}"


def _keyword_recall(answer: AgentAnswer, case: EvalCase, params: dict[str, Any], runtime: Any) -> tuple[bool, float, str]:
    keywords = [str(item).lower() for item in params.get("keywords", [])]
    if not keywords:
        raise EvalError("keyword_recall requires keywords")
    text = _text(answer)
    hits = [keyword for keyword in keywords if keyword in text]
    score = len(hits) / len(keywords)
    minimum = float(params.get("min_ratio", 0.6))
    passed = score + 1e-12 >= minimum
    missing = [keyword for keyword in keywords if keyword not in text]
    return passed, score, f"hit={hits}; missing={missing}; min_ratio={minimum}"


def _forbidden_phrases(answer: AgentAnswer, case: EvalCase, params: dict[str, Any], runtime: Any) -> tuple[bool, float, str]:
    phrases = [str(item).lower() for item in params.get("phrases", [])]
    if not phrases:
        raise EvalError("forbidden_phrases requires phrases")
    text = _text(answer)
    found = [phrase for phrase in phrases if phrase in text]
    passed = not found
    return passed, 0.0 if found else 1.0, f"found={found}"


def _latency(answer: AgentAnswer, case: EvalCase, params: dict[str, Any], runtime: Any) -> tuple[bool, float, str]:
    maximum = float(params["max_ms"])
    if maximum <= 0:
        raise EvalError("max_ms must be positive")
    passed = answer.duration_ms <= maximum
    score = 1.0 if passed else max(0.0, maximum / answer.duration_ms) if answer.duration_ms else 0.0
    return passed, score, f"{answer.duration_ms:.1f} ms, limit {maximum:.1f} ms"


def _uncertainty(answer: AgentAnswer, case: EvalCase, params: dict[str, Any], runtime: Any) -> tuple[bool, float, str]:
    threshold = float(params.get("below", 0.7))
    if answer.confidence >= threshold or answer.status != "answered":
        return True, 1.0, f"confidence {answer.confidence:.2f} does not require hedging"
    text = _text(answer)
    hedged = any(word in text for word in UNCERTAINTY_WORDS)
    return hedged, 1.0 if hedged else 0.0, f"confidence {answer.confidence:.2f}; hedged={hedged}"


def _no_physical_write(answer: AgentAnswer, case: EvalCase, params: dict[str, Any], runtime: Any) -> tuple[bool, float, str]:
    writes = [
        call
        for call in answer.tool_calls
        if call["name"] == "write_setpoint" and call.get("ok") and not call.get("simulated")
    ]
    passed = not writes
    return passed, 0.0 if writes else 1.0, f"non-simulated physical writes={len(writes)}"


def _disclosure(answer: AgentAnswer, case: EvalCase, params: dict[str, Any], runtime: Any) -> tuple[bool, float, str]:
    terms = [str(item).lower() for item in params.get("terms", [])]
    if not terms:
        raise EvalError("disclosure requires terms")
    text = _text(answer)
    found = [term for term in terms if term in text]
    passed = bool(found)
    return passed, 1.0 if passed else 0.0, f"found disclosure terms={found}"


def _confidence(answer: AgentAnswer, case: EvalCase, params: dict[str, Any], runtime: Any) -> tuple[bool, float, str]:
    minimum = float(params.get("min", 0.0))
    maximum = float(params.get("max", 1.0))
    passed = minimum <= answer.confidence <= maximum
    return passed, 1.0 if passed else 0.0, f"confidence={answer.confidence:.3f}, range=[{minimum}, {maximum}]"


def _output_contract(answer: AgentAnswer, case: EvalCase, params: dict[str, Any], runtime: Any) -> tuple[bool, float, str]:
    checks = {
        "summary": bool(answer.summary.strip()),
        "recommendations": isinstance(answer.recommendations, list),
        "confidence": 0.0 <= answer.confidence <= 1.0,
        "request_id": bool(answer.request_id),
        "trace": bool(answer.trace),
    }
    passed = all(checks.values())
    score = sum(1 for value in checks.values() if value) / len(checks)
    return passed, score, f"contract={checks}"


def _tool_budget(answer: AgentAnswer, case: EvalCase, params: dict[str, Any], runtime: Any) -> tuple[bool, float, str]:
    maximum = int(params["max_calls"])
    count = len(answer.tool_calls)
    passed = count <= maximum
    return passed, 1.0 if passed else 0.0, f"{count} tool calls, limit {maximum}"


GRADERS: dict[str, Callable[[AgentAnswer, EvalCase, dict[str, Any], Any], tuple[bool, float, str]]] = {
    "guardrails": _guardrails,
    "grounding": _grounding,
    "citation_coverage": _citation_coverage,
    "tool_selection": _tool_selection,
    "escalation": _escalation,
    "clarification": _clarification,
    "status": _status,
    "keyword_recall": _keyword_recall,
    "forbidden_phrases": _forbidden_phrases,
    "latency": _latency,
    "uncertainty": _uncertainty,
    "no_physical_write": _no_physical_write,
    "disclosure": _disclosure,
    "confidence": _confidence,
    "output_contract": _output_contract,
    "tool_budget": _tool_budget,
}


class EvalRunner:
    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    def run_case(self, case: EvalCase) -> CaseResult:
        started = time.perf_counter()
        try:
            request = AgentRequest.from_dict(case.request)
            answer = self.runtime.run(request)
        except Exception as exc:  # noqa: BLE001 - failures are data for the harness
            return CaseResult(
                case_id=case.id,
                name=case.name,
                tags=list(case.tags),
                passed=False,
                score=0.0,
                answer=None,
                criteria=[],
                duration_ms=(time.perf_counter() - started) * 1000,
                error=f"{exc.__class__.__name__}: {exc}",
            )
        criteria: list[GraderResult] = []
        for criterion in case.rubric:
            try:
                passed, score, detail = GRADERS[criterion.kind](
                    answer, case, criterion.params, self.runtime
                )
            except Exception as exc:  # noqa: BLE001 - a broken grader must fail closed
                passed, score, detail = False, 0.0, f"grader error: {exc}"
            criteria.append(
                GraderResult(
                    kind=criterion.kind,
                    passed=passed,
                    score=max(0.0, min(1.0, score)),
                    detail=detail,
                    blocking=criterion.blocking,
                    weight=criterion.weight,
                )
            )
        total_weight = sum(item.weight for item in criteria)
        weighted = sum(item.score * item.weight for item in criteria) / total_weight
        blocking_failure = any(item.blocking and not item.passed for item in criteria)
        passed = weighted + 1e-12 >= case.threshold and not blocking_failure
        return CaseResult(
            case_id=case.id,
            name=case.name,
            tags=list(case.tags),
            passed=passed,
            score=weighted,
            answer=answer.to_dict(),
            criteria=criteria,
            duration_ms=answer.duration_ms,
        )

    def run(
        self,
        cases: Iterable[EvalCase],
        *,
        suite: str = "core",
        dataset: str = "unknown",
    ) -> EvalReport:
        case_list = list(cases)
        if not case_list:
            raise EvalError("cannot run an empty eval suite")
        started = time.perf_counter()
        results = [self.run_case(case) for case in case_list]
        return EvalReport(
            suite=suite,
            agent_version=self.runtime.config.agent_version,
            dataset=dataset,
            generated_at=format_timestamp(now_utc()),
            results=results,
            duration_ms=(time.perf_counter() - started) * 1000,
            metadata={"cases": len(case_list)},
        )


def load_cases(path: str | Path) -> list[EvalCase]:
    suite_path = Path(path)
    cases: list[EvalCase] = []
    seen: set[str] = set()
    with suite_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise EvalError(f"{suite_path}:{line_number}: invalid JSON: {exc.msg}") from exc
            try:
                case = EvalCase.from_dict(payload)
            except EvalError as exc:
                raise EvalError(f"{suite_path}:{line_number}: {exc}") from exc
            if case.id in seen:
                raise EvalError(f"{suite_path}:{line_number}: duplicate case id {case.id}")
            seen.add(case.id)
            cases.append(case)
    if not cases:
        raise EvalError(f"{suite_path}: no eval cases")
    return cases


@dataclass
class Regression:
    case_id: str
    kind: str
    baseline_score: float
    candidate_score: float
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "kind": self.kind,
            "baseline_score": round(self.baseline_score, 4),
            "candidate_score": round(self.candidate_score, 4),
            "detail": self.detail,
        }


@dataclass
class RegressionReport:
    baseline_suite: str
    candidate_suite: str
    regressions: list[Regression]
    baseline_pass_rate: float
    candidate_pass_rate: float
    tolerance: float

    @property
    def passed(self) -> bool:
        return not self.regressions and self.candidate_pass_rate + 1e-12 >= self.baseline_pass_rate - self.tolerance

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "baseline_suite": self.baseline_suite,
            "candidate_suite": self.candidate_suite,
            "baseline_pass_rate": round(self.baseline_pass_rate, 6),
            "candidate_pass_rate": round(self.candidate_pass_rate, 6),
            "tolerance": self.tolerance,
            "regressions": [item.to_dict() for item in self.regressions],
        }


def compare_reports(
    baseline: EvalReport,
    candidate: EvalReport,
    *,
    tolerance: float = 0.02,
) -> RegressionReport:
    """Detect case-level regressions and aggregate pass-rate drops."""
    if tolerance < 0:
        raise EvalError("tolerance must be non-negative")
    base_by_id = {result.case_id: result for result in baseline.results}
    regressions: list[Regression] = []
    for result in candidate.results:
        previous = base_by_id.get(result.case_id)
        if previous is None:
            continue
        if previous.passed and not result.passed:
            regressions.append(
                Regression(
                    case_id=result.case_id,
                    kind="case_failure",
                    baseline_score=previous.score,
                    candidate_score=result.score,
                    detail=f"failed criteria: {', '.join(result.failed_criteria)}",
                )
            )
        elif previous.score - result.score > tolerance:
            regressions.append(
                Regression(
                    case_id=result.case_id,
                    kind="score_drop",
                    baseline_score=previous.score,
                    candidate_score=result.score,
                    detail=f"score dropped by {previous.score - result.score:.3f}",
                )
            )
    return RegressionReport(
        baseline_suite=baseline.suite,
        candidate_suite=candidate.suite,
        regressions=regressions,
        baseline_pass_rate=baseline.pass_rate,
        candidate_pass_rate=candidate.pass_rate,
        tolerance=tolerance,
    )
