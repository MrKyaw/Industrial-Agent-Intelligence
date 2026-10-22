"""Tests for the evaluation harness and report rendering."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from iai.app import Application
from iai.evals import EvalCase, EvalRunner, load_cases, compare_reports
from iai.models import AgentAnswer
from iai.report import render_junit, render_markdown, write_report


def test_load_cases(data_dir):
    cases = load_cases(data_dir / "evals" / "core.jsonl")
    assert len(cases) >= 20
    assert all(case.rubric for case in cases)


def test_eval_runner_smoke(app):
    from iai.evals import load_cases

    cases = load_cases(app.data_dir / "evals" / "core.jsonl")
    runner = EvalRunner(app.runtime)
    report = runner.run(cases[:3], suite="smoke", dataset="test")
    assert report.total == 3
    assert 0 <= report.pass_rate <= 1


def test_eval_case_validation():
    with pytest.raises(Exception):
        EvalCase(id="x", name="y", request={"question": "q", "site_id": "site.dc-lon-01"}, rubric=[])
    with pytest.raises(Exception):
        EvalCase(id="x", name="y", request={"question": "q", "site_id": "site.dc-lon-01"}, rubric=[{"kind": "guardrails"}], threshold=2.0)


def test_compare_reports(app, tmp_path):
    from iai.evals import load_cases

    cases = load_cases(app.data_dir / "evals" / "core.jsonl")
    runner = EvalRunner(app.runtime)
    baseline = runner.run(cases[:5], suite="base", dataset="test")
    candidate = runner.run(cases[:5], suite="candidate", dataset="test")
    comparison = compare_reports(baseline, candidate)
    assert comparison.passed


def test_report_rendering(app, tmp_path):
    from iai.evals import load_cases

    cases = load_cases(app.data_dir / "evals" / "core.jsonl")
    runner = EvalRunner(app.runtime)
    report = runner.run(cases[:2], suite="render", dataset="test")
    paths = write_report(report, tmp_path, stem="demo")
    assert paths["markdown"].exists()
    assert paths["html"].exists()
    assert paths["junit"].exists()
    markdown = render_markdown(report)
    assert "render" in markdown
    junit = render_junit(report)
    assert "<testsuite" in junit
