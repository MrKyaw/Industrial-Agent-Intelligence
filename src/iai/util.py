"""Small, dependency-free helpers shared across the platform.

Everything in this module is deterministic and side-effect free so it can be used
inside evaluation graders, guardrails, and the HTTP layer.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

UTC = timezone.utc

_NUMBER_RE = re.compile(r"(?<![\w:.])-?\d+(?:\.\d+)?")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_SECRET_RES = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{12,}\b"),
    re.compile(r"(?i)\b(bearer|token|password|secret)\s*[:=]\s*[^\s,;]+"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
)


def now_utc() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(tz=UTC)


def parse_timestamp(value: str | datetime) -> datetime:
    """Parse an ISO-8601 timestamp into an aware UTC datetime.

    Naive values are interpreted as UTC. ``Z`` suffixes are accepted.
    """
    if isinstance(value, datetime):
        parsed = value
    else:
        text = value.strip()
        if not text:
            raise ValueError("timestamp must not be empty")
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def format_timestamp(value: datetime) -> str:
    """Format a datetime as a compact ISO-8601 UTC string."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def stable_json(value: Any) -> str:
    """Serialize JSON with sorted keys and no incidental whitespace."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(value: str) -> str:
    """Return the hex SHA-256 digest of a UTF-8 string."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def fingerprint(value: Any) -> str:
    """Return a short stable fingerprint for any JSON-serializable value."""
    return sha256_text(stable_json(value))[:16]


def slugify(value: str) -> str:
    """Convert a label into a URL/filename-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "item"


def approx_tokens(text: str) -> int:
    """Cheap token estimate used only for context-budget packing."""
    return max(1, math.ceil(len(text) / 4))


def clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into the inclusive ``[low, high]`` interval."""
    if low > high:
        raise ValueError("low must not exceed high")
    return max(low, min(high, value))


def percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile for a non-empty sample."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 <= pct <= 100:
        raise ValueError("pct must be between 0 and 100")
    ordered = sorted(values)
    position = (len(ordered) - 1) * (pct / 100)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def extract_numbers(text: str) -> list[float]:
    """Extract numeric literals from prose, ignoring ids and citations.

    The negative look-behind avoids matching digits embedded in identifiers
    such as ``E1`` or ``chiller-1``.
    """
    return [float(match.group(0)) for match in _NUMBER_RE.finditer(text)]


def numbers_match(left: float, right: float, tolerance: float = 0.05) -> bool:
    """Compare two numbers with an absolute tolerance."""
    return abs(left - right) <= tolerance + 1e-12


def redact_secrets(text: str) -> str:
    """Redact e-mail addresses and common credential shapes from text."""
    cleaned = _EMAIL_RE.sub("[redacted-email]", text)
    for pattern in _SECRET_RES:
        cleaned = pattern.sub("[redacted-secret]", cleaned)
    return cleaned


def is_stale(last_seen: datetime, now: datetime, max_age: timedelta) -> bool:
    """Return True when ``last_seen`` is older than ``now - max_age``."""
    return now - last_seen > max_age


def pct_change(baseline: float, candidate: float) -> float:
    """Relative change from baseline, guarding against division by zero."""
    if baseline == 0:
        return 0.0 if candidate == 0 else math.copysign(math.inf, candidate)
    return (candidate - baseline) / abs(baseline) * 100.0


def mean(values: Iterable[float]) -> float:
    """Arithmetic mean that rejects empty input."""
    items = list(values)
    if not items:
        raise ValueError("mean requires at least one value")
    return sum(items) / len(items)


def tail(values: Sequence[float], count: int) -> list[float]:
    """Return the last ``count`` values, tolerating short input."""
    if count <= 0:
        raise ValueError("count must be positive")
    return list(values[-count:])
