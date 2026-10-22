"""Tests for the CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from iai.cli import main


def test_version():
    assert main(["version"]) == 0


def test_validate():
    assert main(["validate"]) == 0


def test_demo_json():
    code = main(["demo", "--json"])
    assert code == 0


def test_ask_json(tmp_path):
    code = main(["ask", "How efficient is Chiller 1 right now?", "--site", "site.dc-lon-01", "--json"])
    assert code == 0


def test_evals_run_gate(tmp_path):
    out = tmp_path / "reports"
    code = main(["evals", "run", "--suite", "data/evals/core.jsonl", "--out", str(out), "--min-pass-rate", "1.0"])
    assert code == 0
    assert (out / "latest.json").exists()


def test_evals_compare(tmp_path):
    out = tmp_path / "reports"
    main(["evals", "run", "--suite", "data/evals/core.jsonl", "--out", str(out), "--stem", "base"])
    main(["evals", "run", "--suite", "data/evals/core.jsonl", "--out", str(out), "--stem", "candidate"])
    code = main(["evals", "compare", "--baseline", str(out / "base.json"), "--candidate", str(out / "candidate.json")])
    assert code == 0


def test_feedback_roundtrip(tmp_path):
    db = tmp_path / "feedback.db"
    assert main(["feedback", "add", "--db", str(db), "--request-id", "r1", "--rating", "5"]) == 0
    assert main(["feedback", "summary", "--db", str(db)]) == 0
    assert main(["feedback", "list", "--db", str(db)]) == 0


def test_report_render(tmp_path):
    out = tmp_path / "reports"
    main(["evals", "run", "--suite", "data/evals/core.jsonl", "--out", str(out)])
    code = main(["report", str(out / "latest.json"), "--out", str(out), "--stem", "rendered"])
    assert code == 0
    assert (out / "rendered.html").exists()
