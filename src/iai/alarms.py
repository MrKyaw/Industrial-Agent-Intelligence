"""Alarm ingestion, deduplication, prioritisation, and root-cause ranking.

Alarm floods are the normal failure mode in building and industrial operations.
This module turns a noisy list into ranked clusters, an explicit flood index, and
candidate root causes that an agent can investigate with tools.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from .ontology import Ontology
from .util import clamp, format_timestamp, now_utc, parse_timestamp

SEVERITY_WEIGHT = {"low": 8, "medium": 18, "high": 32, "critical": 50}
SEVERITY_ORDER = {"low": 1, "medium": 2, "high": 3, "critical": 4}

PLAYBOOKS: dict[str, list[str]] = {
    "high_condenser_pressure": [
        "Verify cooling-tower fan operation and condenser-water flow.",
        "Trend condenser approach temperature to distinguish fouling from airflow loss.",
        "Confirm the condenser-pressure transmitter against a calibrated reference.",
    ],
    "low_cop": [
        "Compare kW/ton with the commissioning baseline at similar load.",
        "Check refrigerant charge, condenser cleanliness, and compressor staging.",
    ],
    "low_delta_t": [
        "Check chilled-water valve authority and coil fouling.",
        "Verify flow balance and supply/return temperature sensor calibration.",
    ],
    "ahu_supply_air_high": [
        "Check the AHU cooling-valve position and supply-air temperature sensor.",
        "Verify zone load, filter differential pressure, and fan operation.",
    ],
    "condenser_fan_high": [
        "Check fan vibration, blade condition, and variable-speed drive limits.",
        "Confirm tower airflow is not recirculating.",
    ],
}


class AlarmError(ValueError):
    """Raised for malformed alarm records or invalid triage requests."""


@dataclass(frozen=True)
class Alarm:
    id: str
    site_id: str
    equipment_id: str
    code: str
    severity: str
    raised_at: datetime
    message: str
    tag: str | None = None
    value: float | None = None
    limit: float | None = None
    resolved_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.severity not in SEVERITY_WEIGHT:
            raise AlarmError(f"unsupported severity: {self.severity}")
        if not self.id or not self.site_id or not self.equipment_id or not self.code:
            raise AlarmError("alarm id, site_id, equipment_id, and code are required")
        if self.resolved_at is not None and self.resolved_at < self.raised_at:
            raise AlarmError("resolved_at must not precede raised_at")
        if self.value is not None and not isinstance(self.value, (int, float)):
            raise AlarmError("alarm value must be numeric")

    @property
    def active(self) -> bool:
        return self.resolved_at is None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Alarm":
        if not isinstance(payload, dict):
            raise AlarmError("alarm must be an object")
        required = ("id", "site_id", "equipment_id", "code", "severity", "raised_at", "message")
        missing = [key for key in required if not payload.get(key)]
        if missing:
            raise AlarmError(f"alarm missing fields: {', '.join(missing)}")
        try:
            raised_at = parse_timestamp(str(payload["raised_at"]))
        except ValueError as exc:
            raise AlarmError(f"invalid raised_at: {exc}") from exc
        resolved_at = None
        if payload.get("resolved_at"):
            try:
                resolved_at = parse_timestamp(str(payload["resolved_at"]))
            except ValueError as exc:
                raise AlarmError(f"invalid resolved_at: {exc}") from exc
        return cls(
            id=str(payload["id"]),
            site_id=str(payload["site_id"]),
            equipment_id=str(payload["equipment_id"]),
            code=str(payload["code"]),
            severity=str(payload["severity"]),
            raised_at=raised_at,
            message=str(payload["message"]),
            tag=payload.get("tag"),
            value=payload.get("value"),
            limit=payload.get("limit"),
            resolved_at=resolved_at,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "site_id": self.site_id,
            "equipment_id": self.equipment_id,
            "code": self.code,
            "severity": self.severity,
            "raised_at": format_timestamp(self.raised_at),
            "message": self.message,
            "tag": self.tag,
            "value": self.value,
            "limit": self.limit,
        }
        if self.resolved_at is not None:
            payload["resolved_at"] = format_timestamp(self.resolved_at)
        return payload


@dataclass
class AlarmCluster:
    equipment_id: str
    code: str
    alarms: list[Alarm]

    @property
    def count(self) -> int:
        return len(self.alarms)

    @property
    def first_raised_at(self) -> datetime:
        return min(alarm.raised_at for alarm in self.alarms)

    @property
    def last_raised_at(self) -> datetime:
        return max(alarm.raised_at for alarm in self.alarms)

    @property
    def severity(self) -> str:
        return max(self.alarms, key=lambda alarm: SEVERITY_ORDER[alarm.severity]).severity

    @property
    def representative(self) -> Alarm:
        return max(self.alarms, key=lambda alarm: (SEVERITY_ORDER[alarm.severity], alarm.raised_at))

    def to_dict(self) -> dict[str, Any]:
        return {
            "equipment_id": self.equipment_id,
            "code": self.code,
            "count": self.count,
            "severity": self.severity,
            "first_raised_at": format_timestamp(self.first_raised_at),
            "last_raised_at": format_timestamp(self.last_raised_at),
            "sample_message": self.representative.message,
            "alarm_ids": [alarm.id for alarm in self.alarms],
        }


class AlarmStore:
    """In-memory alarm history with simple filters."""

    def __init__(self, alarms: Iterable[Alarm] = ()) -> None:
        self._alarms: list[Alarm] = []
        self._ids: set[str] = set()
        for alarm in alarms:
            self.add(alarm)

    def add(self, alarm: Alarm) -> None:
        if alarm.id in self._ids:
            raise AlarmError(f"duplicate alarm id: {alarm.id}")
        self._alarms.append(alarm)
        self._ids.add(alarm.id)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "AlarmStore":
        store = cls()
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise AlarmError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
                try:
                    store.add(Alarm.from_dict(payload))
                except AlarmError as exc:
                    raise AlarmError(f"{path}:{line_number}: {exc}") from exc
        return store

    def all(self) -> list[Alarm]:
        return sorted(self._alarms, key=lambda alarm: alarm.raised_at)

    def __len__(self) -> int:
        return len(self._alarms)

    def filter(
        self,
        *,
        site_id: str | None = None,
        equipment_id: str | None = None,
        since: str | datetime | None = None,
        until: str | datetime | None = None,
        active_only: bool = False,
        severity: str | None = None,
    ) -> list[Alarm]:
        since_dt = parse_timestamp(since) if since is not None else None
        until_dt = parse_timestamp(until) if until is not None else None
        if since_dt and until_dt and until_dt < since_dt:
            raise AlarmError("until must not precede since")
        results = []
        for alarm in self.all():
            if site_id is not None and alarm.site_id != site_id:
                continue
            if equipment_id is not None and alarm.equipment_id != equipment_id:
                continue
            if since_dt is not None and alarm.raised_at < since_dt:
                continue
            if until_dt is not None and alarm.raised_at > until_dt:
                continue
            if active_only and not alarm.active:
                continue
            if severity is not None and alarm.severity != severity:
                continue
            results.append(alarm)
        return results

    def counts_by_equipment(
        self, *, site_id: str | None = None, active_only: bool = True
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for alarm in self.filter(site_id=site_id, active_only=active_only):
            counts[alarm.equipment_id] = counts.get(alarm.equipment_id, 0) + 1
        return dict(sorted(counts.items()))


def cluster_alarms(alarms: Iterable[Alarm], *, window_s: float = 1800) -> list[AlarmCluster]:
    """Group repeated alarms on the same asset/code inside a sliding window."""
    if window_s <= 0:
        raise AlarmError("window_s must be positive")
    grouped: dict[tuple[str, str], list[Alarm]] = {}
    for alarm in sorted(alarms, key=lambda item: item.raised_at):
        grouped.setdefault((alarm.equipment_id, alarm.code), []).append(alarm)
    clusters: list[AlarmCluster] = []
    for (equipment_id, code), items in sorted(grouped.items()):
        current: list[Alarm] = []
        for alarm in items:
            if current and (alarm.raised_at - current[-1].raised_at).total_seconds() > window_s:
                clusters.append(AlarmCluster(equipment_id, code, current))
                current = []
            current.append(alarm)
        if current:
            clusters.append(AlarmCluster(equipment_id, code, current))
    return sorted(clusters, key=lambda cluster: (cluster.first_raised_at, cluster.equipment_id))


def priority_score(
    cluster: AlarmCluster,
    *,
    equipment_criticality: int = 3,
    now: datetime | None = None,
) -> float:
    """Score a cluster on a 0-100 scale using transparent additive factors."""
    if not 1 <= equipment_criticality <= 5:
        raise AlarmError("equipment_criticality must be within [1, 5]")
    base = SEVERITY_WEIGHT[cluster.severity]
    criticality_bonus = equipment_criticality * 4
    repeat_bonus = min(10.0, (cluster.count - 1) * 2.0)
    age_bonus = 0.0
    if now is not None:
        age_minutes = (now - cluster.last_raised_at).total_seconds() / 60.0
        if age_minutes < 0:
            raise AlarmError("cluster timestamp is in the future")
        if age_minutes <= 15:
            age_bonus = 5.0
        elif age_minutes >= 1440:
            age_bonus = -5.0
    return clamp(base + criticality_bonus + repeat_bonus + age_bonus, 0.0, 100.0)


def priority_band(score: float) -> str:
    if score >= 75:
        return "P1"
    if score >= 55:
        return "P2"
    if score >= 35:
        return "P3"
    return "P4"


def flood_index(clusters: Iterable[AlarmCluster]) -> float:
    weighted = sum(SEVERITY_WEIGHT[cluster.severity] * cluster.count for cluster in clusters)
    return clamp(weighted / 300.0, 0.0, 1.0)


def flood_level(index: float) -> str:
    if index >= 0.6:
        return "flood"
    if index >= 0.3:
        return "elevated"
    return "normal"


@dataclass
class RootCauseCandidate:
    equipment_id: str
    name: str
    score: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "equipment_id": self.equipment_id,
            "name": self.name,
            "score": round(self.score, 1),
            "reasons": self.reasons,
        }


@dataclass
class TriageSummary:
    site_id: str
    generated_at: str
    total_alarms: int
    clusters: list[AlarmCluster]
    ranked: list[dict[str, Any]]
    flood_index: float
    flood_level: str
    escalation_recommended: bool
    escalation_reasons: list[str]
    next_checks: list[str]
    root_causes: list[RootCauseCandidate] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "site_id": self.site_id,
            "generated_at": self.generated_at,
            "total_alarms": self.total_alarms,
            "clusters": [cluster.to_dict() for cluster in self.clusters],
            "ranked": self.ranked,
            "flood_index": round(self.flood_index, 3),
            "flood_level": self.flood_level,
            "escalation_recommended": self.escalation_recommended,
            "escalation_reasons": self.escalation_reasons,
            "next_checks": self.next_checks,
            "root_causes": [candidate.to_dict() for candidate in self.root_causes],
        }


def _criticality_map(ontology: Ontology | None, equipment_id: str) -> int:
    if ontology is None or not ontology.has(equipment_id):
        return 3
    criticality = ontology.get(equipment_id).attributes.get("criticality", 3)
    return int(criticality) if isinstance(criticality, int) else 3


def triage(
    alarms: Iterable[Alarm],
    *,
    site_id: str,
    ontology: Ontology | None = None,
    now: datetime | None = None,
    window_s: float = 1800,
    max_ranked: int = 10,
    root_cause_equipment: str | None = None,
    anomaly_scores: dict[str, float] | None = None,
) -> TriageSummary:
    """Cluster, rank, and explain an alarm set for one site."""
    if not site_id:
        raise AlarmError("site_id is required")
    if max_ranked <= 0:
        raise AlarmError("max_ranked must be positive")
    current_time = now or now_utc()
    site_alarms = [alarm for alarm in alarms if alarm.site_id == site_id]
    clusters = cluster_alarms(site_alarms, window_s=window_s)
    scored: list[tuple[float, AlarmCluster]] = []
    for cluster in clusters:
        score = priority_score(
            cluster,
            equipment_criticality=_criticality_map(ontology, cluster.equipment_id),
            now=current_time,
        )
        scored.append((score, cluster))
    scored.sort(key=lambda item: (-item[0], item[1].first_raised_at, item[1].equipment_id))
    ranked = [
        {
            **cluster.to_dict(),
            "priority_score": round(score, 1),
            "priority_band": priority_band(score),
        }
        for score, cluster in scored[:max_ranked]
    ]
    index = flood_index(clusters)
    level = flood_level(index)
    reasons: list[str] = []
    if any(alarm.severity == "critical" and alarm.active for alarm in site_alarms):
        reasons.append("Active critical-severity alarm requires human acknowledgement.")
    if level == "flood":
        reasons.append("Alarm volume is above the flood threshold; deduplicate before dispatch.")
    if ranked and ranked[0]["priority_band"] == "P1":
        reasons.append("At least one cluster is P1 on a high-criticality asset.")
    checks: list[str] = []
    for _, cluster in scored[:3]:
        for check in PLAYBOOKS.get(cluster.code, ["Validate the alarm against raw telemetry and asset state."]):
            if check not in checks:
                checks.append(check)
    root_causes: list[RootCauseCandidate] = []
    if root_cause_equipment:
        root_causes = root_cause_candidates(
            root_cause_equipment,
            ontology=ontology,
            alarms=site_alarms,
            anomaly_scores=anomaly_scores or {},
        )
    return TriageSummary(
        site_id=site_id,
        generated_at=format_timestamp(current_time),
        total_alarms=len(site_alarms),
        clusters=clusters,
        ranked=ranked,
        flood_index=index,
        flood_level=level,
        escalation_recommended=bool(reasons),
        escalation_reasons=reasons,
        next_checks=checks[:6],
        root_causes=root_causes,
    )


def root_cause_candidates(
    equipment_id: str,
    *,
    ontology: Ontology | None,
    alarms: Iterable[Alarm],
    anomaly_scores: dict[str, float],
    limit: int = 3,
) -> list[RootCauseCandidate]:
    """Rank nearby assets using active alarms and sensor-level anomalies.

    The score is a heuristic, not a diagnosis. Each contribution is returned in
    ``reasons`` so a human can audit or reject the ranking.
    """
    if limit <= 0:
        raise AlarmError("limit must be positive")
    alarm_list = list(alarms)
    if ontology is None:
        active = [alarm for alarm in alarm_list if alarm.equipment_id == equipment_id and alarm.active]
        reasons = [f"{len(active)} active alarm(s) on the asset."] if active else []
        score = clamp(len(active) * 8.0, 0.0, 100.0)
        return [RootCauseCandidate(equipment_id, equipment_id, score, reasons)]

    if not ontology.has(equipment_id):
        raise AlarmError(f"unknown equipment: {equipment_id}")
    related = {
        node.id: node
        for node in ontology.subgraph(equipment_id, hops=1)
        if node.type == "equipment"
    }
    related[equipment_id] = ontology.get(equipment_id)
    candidates: list[RootCauseCandidate] = []
    for candidate_id, node in related.items():
        score = 0.0
        reasons: list[str] = []
        active = [
            alarm for alarm in alarm_list if alarm.equipment_id == candidate_id and alarm.active
        ]
        if active:
            alarm_points = min(35.0, len(active) * 9.0)
            score += alarm_points
            reasons.append(
                f"{len(active)} active alarm(s): {', '.join(sorted({alarm.code for alarm in active}))}."
            )
            if any(alarm.severity in {"high", "critical"} for alarm in active):
                score += 10.0
                reasons.append("At least one high-severity alarm is active.")
        sensor_points = 0.0
        for sensor in ontology.sensors_for(candidate_id):
            anomaly = anomaly_scores.get(sensor.id, 0.0)
            if anomaly > 0:
                sensor_points += min(15.0, anomaly * 2.0)
                reasons.append(f"Anomaly detected on {sensor.name} ({sensor.id}).")
        score += min(25.0, sensor_points)
        if candidate_id == equipment_id:
            score += 5.0
            reasons.append("Asset under investigation is included as a candidate.")
        if not reasons:
            reasons.append("Related asset with no active alarm or anomaly; low priority.")
        candidates.append(
            RootCauseCandidate(
                equipment_id=candidate_id,
                name=node.name,
                score=clamp(score, 0.0, 100.0),
                reasons=reasons,
            )
        )
    candidates.sort(key=lambda candidate: (-candidate.score, candidate.equipment_id))
    return candidates[:limit]


def summarize_window(
    alarms: Iterable[Alarm], *, site_id: str, now: datetime | None = None, hours: int = 24
) -> dict[str, Any]:
    """Compact alarm history summary for tool output."""
    if hours <= 0:
        raise AlarmError("hours must be positive")
    current = now or now_utc()
    since = current - timedelta(hours=hours)
    selected = [
        alarm
        for alarm in alarms
        if alarm.site_id == site_id and alarm.raised_at >= since
    ]
    return {
        "site_id": site_id,
        "window_hours": hours,
        "total": len(selected),
        "active": sum(1 for alarm in selected if alarm.active),
        "by_severity": {
            severity: sum(1 for alarm in selected if alarm.severity == severity)
            for severity in SEVERITY_WEIGHT
        },
        "by_equipment": {
            equipment_id: sum(1 for alarm in selected if alarm.equipment_id == equipment_id)
            for equipment_id in sorted({alarm.equipment_id for alarm in selected})
        },
    }
