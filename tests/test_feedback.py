"""Tests for feedback storage and drift detection."""

from __future__ import annotations

import pytest

from iai.feedback import FeedbackError, FeedbackStore


def test_feedback_add_list_summary(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.db")
    record = store.add(request_id="req-1", rating=5, agent_version="1.0.0", slice_name="triage", site_id="site.dc-lon-01")
    assert record.rating == 5
    assert len(store) == 1
    assert store.list(limit=10)[0].id == record.id
    summary = store.summary()
    assert summary["total"] == 1
    assert summary["positive_rate"] == 1.0


def test_feedback_validation(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.db")
    with pytest.raises(FeedbackError):
        store.add(request_id="req", rating=6)
    with pytest.raises(FeedbackError):
        store.add(request_id="", rating=4)
    with pytest.raises(FeedbackError):
        store.label(999, "useful")


def test_feedback_label_and_pending(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.db")
    record = store.add(request_id="req-1", rating=2)
    assert store.pending()[0].id == record.id
    labelled = store.label(record.id, "incorrect")
    assert labelled.label == "incorrect"
    assert store.pending() == []


def test_feedback_export(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.db")
    store.add(request_id="req-1", rating=4)
    output = tmp_path / "labels.jsonl"
    assert store.export_jsonl(output) == 1
    assert output.exists()


def test_feedback_drift_insufficient_and_stable(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.db")
    report = store.drift(recent_n=5, baseline_n=5)
    assert report.status == "insufficient_data"
    for i in range(20):
        store.add(request_id=f"r{i}", rating=4 if i % 2 == 0 else 3, created_at=f"2026-01-{i+1:02d}T00:00:00Z")
    report = store.drift(recent_n=5, baseline_n=5)
    assert report.status in {"stable", "regressed", "improved"}
    assert report.recent_n == 5
    assert report.baseline_n == 5
