"""Tests for alarm ingestion, clustering, triage, and root cause ranking."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from iai.alarms import (
    Alarm,
    AlarmError,
    AlarmStore,
    cluster_alarms,
    priority_score,
    triage,
    root_cause_candidates,
    summarize_window,
)


def make_alarm(
    alarm_id="a1",
    site="site.dc-lon-01",
    equipment="plant.chiller-1",
    code="high_condenser_pressure",
    severity="critical",
    raised_at=None,
    resolved_at=None,
):
    raised_at = raised_at or datetime(2026, 10, 3, 18, 0, tzinfo=timezone.utc)
    return Alarm(
        id=alarm_id,
        site_id=site,
        equipment_id=equipment,
        code=code,
        severity=severity,
        raised_at=raised_at,
        message="test",
        tag="sensor.chiller-1.condenser_pressure_kpa",
        value=1810.0,
        limit=1800.0,
        resolved_at=resolved_at,
    )


def test_alarm_from_dict_and_store(tmp_path):
    store = AlarmStore()
    alarm = make_alarm()
    store.add(alarm)
    assert len(store) == 1
    assert store.filter(site_id="site.dc-lon-01", active_only=True)
    with pytest.raises(AlarmError):
        store.add(alarm)


def test_alarm_jsonl_roundtrip(tmp_path):
    path = tmp_path / "alarms.jsonl"
    path.write_text(
        '{"id":"a1","site_id":"site.dc-lon-01","equipment_id":"plant.chiller-1",'
        '"code":"high_condenser_pressure","severity":"critical",'
        '"raised_at":"2026-10-03T18:00:00Z","message":"msg"}\n'
    )
    store = AlarmStore.from_jsonl(path)
    assert len(store) == 1


def test_cluster_alarms_groups_repeats():
    base = datetime(2026, 10, 3, 18, 0, tzinfo=timezone.utc)
    alarms = [
        make_alarm(f"a{i}", raised_at=base + timedelta(minutes=i * 5)) for i in range(4)
    ]
    clusters = cluster_alarms(alarms, window_s=600)
    assert len(clusters) == 1
    assert clusters[0].count == 4


def test_priority_score_and_band():
    cluster = cluster_alarms([make_alarm()], window_s=600)[0]
    score = priority_score(cluster, equipment_criticality=5)
    assert 0 <= score <= 100
    assert priority_score(cluster, equipment_criticality=5, now=datetime(2026, 10, 3, 18, 5, tzinfo=timezone.utc)) > score - 1


def test_triage_produces_escalation():
    alarms = [make_alarm(f"a{i}", raised_at=datetime(2026, 10, 3, 18, i * 10, tzinfo=timezone.utc)) for i in range(3)]
    from iai.ontology import load_ontology
    from pathlib import Path

    ontology = load_ontology(Path(__file__).resolve().parents[1] / "data" / "ontology.json")
    summary = triage(alarms, site_id="site.dc-lon-01", ontology=ontology, now=datetime(2026, 10, 3, 19, tzinfo=timezone.utc))
    assert summary.total_alarms == 3
    assert summary.escalation_recommended is True
    assert summary.flood_level in {"normal", "elevated", "flood"}


def test_root_cause_candidates(ontology):
    alarms = [make_alarm()]
    from iai.telemetry import TagStore

    candidates = root_cause_candidates(
        "plant.chiller-1",
        ontology=ontology,
        alarms=alarms,
        anomaly_scores={"sensor.chiller-1.condenser_pressure_kpa": 5.0},
        limit=3,
    )
    assert len(candidates) <= 3
    assert len(candidates) >= 1


def test_summarize_window():
    alarms = [make_alarm()]
    result = summarize_window(alarms, site_id="site.dc-lon-01", now=datetime(2026, 10, 3, 19, tzinfo=timezone.utc), hours=24)
    assert result["total"] == 1


def test_alarm_validation_errors():
    with pytest.raises(AlarmError):
        make_alarm(severity="bogus")
    with pytest.raises(AlarmError):
        Alarm.from_dict({"id": "x", "site_id": "s", "equipment_id": "e", "code": "c", "severity": "high", "raised_at": "bad", "message": "m"})
