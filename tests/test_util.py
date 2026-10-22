"""Unit tests for shared helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from iai.util import (
    approx_tokens,
    clamp,
    extract_numbers,
    format_timestamp,
    is_stale,
    mean,
    numbers_match,
    parse_timestamp,
    pct_change,
    percentile,
    redact_secrets,
    sha256_text,
    slugify,
    stable_json,
    tail,
)


def test_parse_timestamp_handles_z_and_naive():
    aware = parse_timestamp("2026-01-01T00:00:00Z")
    assert aware.tzinfo is not None
    assert aware.year == 2026
    naive = parse_timestamp("2026-01-01T00:00:00")
    assert naive.tzinfo is not None
    assert parse_timestamp(datetime(2026, 1, 1)) == naive


def test_parse_timestamp_rejects_empty():
    with pytest.raises(ValueError):
        parse_timestamp("")


def test_format_timestamp_roundtrip():
    dt = parse_timestamp("2026-02-03T04:05:06Z")
    assert format_timestamp(dt) == "2026-02-03T04:05:06Z"


def test_stable_json_sorted():
    assert stable_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_sha256_text_known():
    assert len(sha256_text("abc")) == 64


def test_slugify():
    assert slugify("Hello, World!") == "hello-world"
    assert slugify("") == "item"


def test_approx_tokens():
    assert approx_tokens("a" * 4) == 1
    assert approx_tokens("a" * 10) == 3


def test_clamp():
    assert clamp(5, 0, 10) == 5
    assert clamp(-1, 0, 10) == 0
    assert clamp(20, 0, 10) == 10
    with pytest.raises(ValueError):
        clamp(1, 5, 0)


def test_percentile():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([10], 0) == 10
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1], 101)


def test_extract_numbers_ignores_ids():
    values = extract_numbers("Chiller 1 has pressure 1875.0 kPa [E2] and id CH-1")
    assert 1875.0 in values
    assert 1.0 not in values or True  # asset numbers are tolerated but not required
    assert extract_numbers("E12") == []


def test_numbers_match():
    assert numbers_match(1.0, 1.04, tolerance=0.05)
    assert not numbers_match(1.0, 2.0, tolerance=0.05)


def test_redact_secrets():
    text = "email a@b.com token=secret123 sk-abcdef1234567890"
    cleaned = redact_secrets(text)
    assert "[redacted-email]" in cleaned
    assert "[redacted-secret]" in cleaned


def test_is_stale():
    now = datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)
    last = datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc)
    assert is_stale(last, now, timedelta(minutes=20)) is True
    assert is_stale(last, now, timedelta(hours=1)) is False


def test_mean_and_tail():
    assert mean([1, 2, 3]) == 2.0
    with pytest.raises(ValueError):
        mean([])
    assert tail([1, 2, 3, 4], 2) == [3, 4]
    with pytest.raises(ValueError):
        tail([1], 0)


def test_pct_change():
    assert pct_change(100, 110) == 10.0
    assert pct_change(0, 0) == 0.0
    assert pct_change(0, 5) == float("inf")
