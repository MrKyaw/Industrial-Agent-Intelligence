"""Tests for telemetry, anomalies, and derived tags."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from iai.telemetry import (
    DerivedTagEngine,
    TagStore,
    TimeSeries,
    safe_eval,
    kw_per_ton,
    cop,
    delta_t,
    approach_temp,
    tons_from_load,
)


def make_series(values):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    samples = [(start + timedelta(hours=i), value) for i, value in enumerate(values)]
    from iai.telemetry import Sample

    return TimeSeries("tag", "unit", [Sample(ts, value) for ts, value in samples])


def test_timeseries_basic_stats():
    series = make_series([1.0, 2.0, 3.0, 4.0])
    assert series.mean() == 2.5
    assert series.minimum() == 1.0
    assert series.maximum() == 4.0
    assert series.latest.value == 4.0
    assert len(series.window("2026-01-01T01:00:00Z", "2026-01-01T02:00:00Z")) == 2


def test_timeseries_rejects_unsorted_and_duplicates():
    from iai.telemetry import Sample

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(Exception):
        TimeSeries("t", "u", [Sample(start, 1.0), Sample(start, 2.0)])
    with pytest.raises(Exception):
        TimeSeries("t", "u", [Sample(start + timedelta(hours=1), 1.0), Sample(start, 2.0)])


def test_timeseries_from_csv(tmp_path):
    path = tmp_path / "tag.csv"
    with path.open("w") as handle:
        handle.write("timestamp,value\n2026-01-01T00:00:00Z,1.5\n2026-01-01T01:00:00Z,\n")
    series = TimeSeries.from_csv(path, unit="degC")
    assert series.unit == "degC"
    assert series.values == [1.5]
    assert series.missing_ratio(3600) == 0.0


def test_timeseries_from_csv_errors(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("timestamp,value\n2026-01-01T00:00:00Z,not-a-number\n")
    with pytest.raises(Exception):
        TimeSeries.from_csv(path)
    missing = tmp_path / "missing.csv"
    with pytest.raises(Exception):
        TimeSeries.from_csv(missing)


def test_gaps_and_staleness():
    series = make_series([1.0, 2.0, 3.0])
    # inject a gap by rebuilding with a 3-hour step
    from iai.telemetry import Sample

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    gappy = TimeSeries(
        "tag",
        "unit",
        [
            Sample(start, 1.0),
            Sample(start + timedelta(hours=3), 2.0),
        ],
    )
    gaps = gappy.gaps(expected_interval_s=3600)
    assert len(gaps) == 1
    assert gaps[0].missing_intervals == 2
    assert gappy.is_stale(datetime(2026, 1, 1, 5, tzinfo=timezone.utc), 3600) is True


def test_anomaly_detection_flags_outlier():
    values = [10.0] * 12 + [100.0]
    series = make_series(values)
    anomalies = series.detect_anomalies(window=6, z_threshold=2.0)
    assert len(anomalies) == 1
    assert anomalies[0].direction == "above"


def test_anomaly_detection_flat_series_no_false_positive():
    series = make_series([5.0] * 20)
    assert series.detect_anomalies(window=6, z_threshold=2.0) == []


def test_level_shift():
    series = make_series([1.0] * 12 + [5.0] * 4)
    shift = series.detect_level_shift(window=4, min_change_pct=50)
    assert shift is not None
    assert shift.direction == "up"


def test_tag_store_and_derived(app):
    assert app.telemetry.has("sensor.chiller-1.power_kw")
    engine = app.runtime.derived
    series = engine.evaluate("derived.chiller-1.cop")
    assert series.latest.value is not None
    assert series.unit == "COP"


def test_safe_eval_basic_and_rejections():
    assert safe_eval("a + b * 2", {"a": 1, "b": 3}) == 7.0
    with pytest.raises(Exception):
        safe_eval("a.__class__", {"a": 1})
    with pytest.raises(Exception):
        safe_eval("import os", {"a": 1})
    with pytest.raises(Exception):
        safe_eval("a / 0", {"a": 1})
    with pytest.raises(Exception):
        safe_eval("max(a, b) + c", {"a": 1, "b": 2})


def test_safe_eval_missing_variable():
    with pytest.raises(Exception):
        safe_eval("a + z", {"a": 1})


def test_derived_cycle(app):
    from iai.ontology import Ontology, OntologyNode

    # exercise engine error path with a bad tag
    with pytest.raises(Exception):
        app.runtime.derived.evaluate("sensor.does-not-exist")


def test_operational_math():
    assert kw_per_ton(350, 350) == 1.0
    assert cop(350, 350) == pytest.approx(3.5168525, rel=1e-6)
    assert delta_t(12.0, 7.0) == 5.0
    assert approach_temp(30.0, 7.0) == 23.0
    assert tons_from_load(70, 500) == 350.0
    with pytest.raises(Exception):
        kw_per_ton(10, 0)
    with pytest.raises(Exception):
        cop(350, 0)


def test_to_records_and_summary():
    series = make_series(list(range(100)))
    records = series.to_records(10)
    assert len(records) <= 10
    summary = series.summary()
    assert summary["valid_samples"] == 100
