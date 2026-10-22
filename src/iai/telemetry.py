"""Telemetry primitives, operational math, anomaly detection, and derived tags.

The implementation is intentionally small and dependency-free, but it models the
parts of industrial analytics that change agent behaviour: gaps, staleness,
drift, low delta-T, and efficiency ratios.
"""

from __future__ import annotations

import ast
import csv
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

from .ontology import Ontology, OntologyError
from .util import format_timestamp, parse_timestamp

TONS_TO_KW = 3.5168525
KELVIN = "degC"


class TelemetryError(ValueError):
    """Raised for malformed telemetry or impossible computations."""


@dataclass(frozen=True)
class Sample:
    timestamp: datetime
    value: float | None


@dataclass(frozen=True)
class Gap:
    start: datetime
    end: datetime
    duration_s: float
    missing_intervals: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": format_timestamp(self.start),
            "end": format_timestamp(self.end),
            "duration_s": round(self.duration_s, 3),
            "missing_intervals": self.missing_intervals,
        }


@dataclass(frozen=True)
class Anomaly:
    timestamp: datetime
    value: float
    expected: float
    score: float
    direction: str
    severity: str

    def __post_init__(self) -> None:
        if self.direction not in {"above", "below"}:
            raise ValueError("direction must be 'above' or 'below'")
        if self.severity not in {"info", "warning", "critical"}:
            raise ValueError("unsupported anomaly severity")

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": format_timestamp(self.timestamp),
            "value": self.value,
            "expected": round(self.expected, 4),
            "score": round(self.score, 3),
            "direction": self.direction,
            "severity": self.severity,
        }


@dataclass(frozen=True)
class LevelShift:
    before: float
    after: float
    change_pct: float
    direction: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "before": round(self.before, 4),
            "after": round(self.after, 4),
            "change_pct": round(self.change_pct, 3),
            "direction": self.direction,
        }


def _require_finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TelemetryError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise TelemetryError(f"{name} must be finite")
    return numeric


class TimeSeries:
    """An immutable-ish, chronologically ordered tag series."""

    def __init__(self, name: str, unit: str, samples: Iterable[Sample]) -> None:
        if not name:
            raise TelemetryError("series name is required")
        ordered = list(samples)
        previous: datetime | None = None
        for sample in ordered:
            if not isinstance(sample.timestamp, datetime):
                raise TelemetryError(f"{name}: sample timestamp must be a datetime")
            if sample.value is not None:
                _require_finite(name, sample.value)
            if previous is not None and sample.timestamp <= previous:
                raise TelemetryError(f"{name}: timestamps must be strictly increasing")
            previous = sample.timestamp
        self.name = name
        self.unit = unit
        self._samples = ordered

    @classmethod
    def from_csv(cls, path: str | Path, *, name: str | None = None, unit: str = "") -> "TimeSeries":
        csv_path = Path(path)
        try:
            with csv_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None or "timestamp" not in reader.fieldnames:
                    raise TelemetryError(f"{csv_path.name}: missing timestamp column")
                if "value" not in reader.fieldnames:
                    raise TelemetryError(f"{csv_path.name}: missing value column")
                samples: list[Sample] = []
                for line_number, row in enumerate(reader, start=2):
                    raw_timestamp = (row.get("timestamp") or "").strip()
                    raw_value = (row.get("value") or "").strip()
                    if not raw_timestamp:
                        if not raw_value:
                            continue
                        raise TelemetryError(f"{csv_path.name}:{line_number}: missing timestamp")
                    try:
                        timestamp = parse_timestamp(raw_timestamp)
                    except ValueError as exc:
                        raise TelemetryError(
                            f"{csv_path.name}:{line_number}: invalid timestamp {raw_timestamp!r}"
                        ) from exc
                    if raw_value in {"", "nan", "NaN", "null", "None"}:
                        value = None
                    else:
                        try:
                            value = float(raw_value)
                        except ValueError as exc:
                            raise TelemetryError(
                                f"{csv_path.name}:{line_number}: invalid value {raw_value!r}"
                            ) from exc
                    samples.append(Sample(timestamp, value))
        except FileNotFoundError as exc:
            raise TelemetryError(f"telemetry file not found: {csv_path}") from exc
        if not samples:
            raise TelemetryError(f"{csv_path.name}: no samples")
        return cls(name or csv_path.stem, unit, samples)

    @property
    def samples(self) -> list[Sample]:
        return list(self._samples)

    @property
    def values(self) -> list[float]:
        return [sample.value for sample in self._samples if sample.value is not None]

    @property
    def timestamps(self) -> list[datetime]:
        return [sample.timestamp for sample in self._samples]

    @property
    def latest(self) -> Sample | None:
        for sample in reversed(self._samples):
            if sample.value is not None:
                return sample
        return None

    @property
    def earliest(self) -> Sample | None:
        for sample in self._samples:
            if sample.value is not None:
                return sample
        return None

    @property
    def last_timestamp(self) -> datetime | None:
        return self._samples[-1].timestamp if self._samples else None

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self):
        return iter(self._samples)

    def window(self, start: str | datetime, end: str | datetime | None = None) -> "TimeSeries":
        start_dt = parse_timestamp(start)
        end_dt = parse_timestamp(end) if end is not None else None
        if end_dt is not None and end_dt <= start_dt:
            raise TelemetryError("window end must be after start")
        samples = [
            sample
            for sample in self._samples
            if sample.timestamp >= start_dt and (end_dt is None or sample.timestamp <= end_dt)
        ]
        return TimeSeries(self.name, self.unit, samples)

    def last_n(self, count: int) -> "TimeSeries":
        if count <= 0:
            raise TelemetryError("count must be positive")
        return TimeSeries(self.name, self.unit, self._samples[-count:])

    def mean(self) -> float:
        return statistics.fmean(self.values)

    def median(self) -> float:
        return statistics.median(self.values)

    def stdev(self) -> float:
        values = self.values
        return statistics.pstdev(values) if len(values) > 1 else 0.0

    def minimum(self) -> float:
        return min(self.values)

    def maximum(self) -> float:
        return max(self.values)

    def missing_ratio(self, expected_interval_s: float | None = None) -> float:
        if expected_interval_s is None:
            expected_interval_s = self.median_interval_s()
        if not self._samples or expected_interval_s <= 0:
            return 0.0
        start, end = self._samples[0].timestamp, self._samples[-1].timestamp
        expected = max(1, int(round((end - start).total_seconds() / expected_interval_s)) + 1)
        return max(0.0, 1.0 - len(self._samples) / expected)

    def median_interval_s(self) -> float:
        if len(self._samples) < 2:
            return 0.0
        deltas = [
            (right.timestamp - left.timestamp).total_seconds()
            for left, right in zip(self._samples, self._samples[1:])
        ]
        return statistics.median(deltas)

    def gaps(
        self,
        expected_interval_s: float | None = None,
        *,
        tolerance: float = 1.5,
    ) -> list[Gap]:
        if expected_interval_s is None:
            expected_interval_s = self.median_interval_s()
        if expected_interval_s <= 0 or len(self._samples) < 2:
            return []
        if tolerance < 1.0:
            raise TelemetryError("tolerance must be at least 1.0")
        gaps: list[Gap] = []
        for left, right in zip(self._samples, self._samples[1:]):
            delta = (right.timestamp - left.timestamp).total_seconds()
            if delta > expected_interval_s * tolerance:
                gaps.append(
                    Gap(
                        start=left.timestamp,
                        end=right.timestamp,
                        duration_s=delta,
                        missing_intervals=max(1, int(round(delta / expected_interval_s)) - 1),
                    )
                )
        return gaps

    def is_stale(self, now: datetime, max_age_s: float) -> bool:
        if max_age_s <= 0:
            raise TelemetryError("max_age_s must be positive")
        last = self.last_timestamp
        if last is None:
            return True
        return (now - last).total_seconds() > max_age_s

    def detect_anomalies(
        self,
        *,
        window: int = 12,
        z_threshold: float = 3.5,
        min_abs_delta: float = 0.0,
    ) -> list[Anomaly]:
        if window < 3:
            raise TelemetryError("anomaly window must be at least 3")
        if z_threshold <= 0:
            raise TelemetryError("z_threshold must be positive")
        values = [(sample.timestamp, sample.value) for sample in self._samples]
        anomalies: list[Anomaly] = []
        for index in range(window, len(values)):
            history = [value for _, value in values[index - window : index] if value is not None]
            timestamp, current = values[index]
            if current is None or len(history) < 3:
                continue
            expected = statistics.median(history)
            deviations = [abs(value - expected) for value in history]
            mad = statistics.median(deviations)
            scale = max(1.4826 * mad, 1e-9)
            delta = current - expected
            if abs(delta) < min_abs_delta:
                continue
            score = abs(delta) / scale
            if score < z_threshold:
                continue
            direction = "above" if delta > 0 else "below"
            severity = "critical" if score >= 8.0 else "warning"
            anomalies.append(
                Anomaly(
                    timestamp=timestamp,
                    value=current,
                    expected=expected,
                    score=score,
                    direction=direction,
                    severity=severity,
                )
            )
        return anomalies

    def detect_level_shift(
        self,
        *,
        window: int = 6,
        min_change_pct: float = 10.0,
    ) -> LevelShift | None:
        if window < 2:
            raise TelemetryError("window must be at least 2")
        values = self.values
        if len(values) < window * 2:
            return None
        before = statistics.fmean(values[-2 * window : -window])
        after = statistics.fmean(values[-window:])
        change = 0.0 if before == 0 else (after - before) / abs(before) * 100.0
        if abs(change) < min_change_pct:
            return None
        return LevelShift(
            before=before,
            after=after,
            change_pct=change,
            direction="up" if change > 0 else "down",
        )

    def to_records(self, max_points: int = 60) -> list[dict[str, Any]]:
        if max_points <= 0:
            raise TelemetryError("max_points must be positive")
        samples = self._samples
        if len(samples) > max_points:
            step = max(1, math.ceil(len(samples) / max_points))
            samples = samples[::step][-max_points:]
        return [
            {
                "timestamp": format_timestamp(sample.timestamp),
                "value": sample.value,
            }
            for sample in samples
        ]

    def summary(self) -> dict[str, Any]:
        values = self.values
        latest = self.latest
        return {
            "tag": self.name,
            "unit": self.unit,
            "samples": len(self._samples),
            "valid_samples": len(values),
            "latest": None if latest is None else latest.value,
            "latest_at": None if latest is None else format_timestamp(latest.timestamp),
            "mean": round(statistics.fmean(values), 4) if values else None,
            "min": round(min(values), 4) if values else None,
            "max": round(max(values), 4) if values else None,
            "stdev": round(self.stdev(), 4) if len(values) > 1 else 0.0,
        }


class TagStore:
    """Named raw telemetry series, loaded from a directory of CSV files."""

    def __init__(self) -> None:
        self._series: dict[str, TimeSeries] = {}

    def add(self, series: TimeSeries) -> None:
        if series.name in self._series:
            raise TelemetryError(f"duplicate series: {series.name}")
        self._series[series.name] = series

    def get(self, name: str) -> TimeSeries:
        try:
            return self._series[name]
        except KeyError as exc:
            raise TelemetryError(f"unknown tag: {name}") from exc

    def has(self, name: str) -> bool:
        return name in self._series

    def names(self) -> list[str]:
        return sorted(self._series)

    def latest(self, name: str) -> Sample | None:
        return self.get(name).latest

    def stale(self, now: datetime, max_age_s: float) -> list[str]:
        return [name for name in self.names() if self._series[name].is_stale(now, max_age_s)]

    @classmethod
    def from_dir(cls, directory: str | Path, units: dict[str, str] | None = None) -> "TagStore":
        path = Path(directory)
        if not path.is_dir():
            raise TelemetryError(f"telemetry directory not found: {path}")
        store = cls()
        for csv_path in sorted(path.glob("*.csv")):
            tag = csv_path.stem
            store.add(TimeSeries.from_csv(csv_path, name=tag, unit=(units or {}).get(tag, "")))
        if not store.names():
            raise TelemetryError(f"no CSV telemetry files in {path}")
        return store


_ALLOWED_FUNCTIONS = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "sqrt": math.sqrt,
    "log": math.log,
}
_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
)


def safe_eval(expression: str, variables: dict[str, float]) -> float:
    """Evaluate a tiny arithmetic expression with no attribute or call escape.

    Only numeric constants, whitelisted names, arithmetic operators, and a small
    set of pure math functions are permitted.
    """
    if not expression.strip():
        raise TelemetryError("expression must not be empty")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise TelemetryError(f"invalid expression: {exc.msg}") from exc
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise TelemetryError(f"disallowed expression element: {type(node).__name__}")
        if isinstance(node, ast.Constant) and (
            isinstance(node.value, bool) or not isinstance(node.value, (int, float))
        ):
            raise TelemetryError("expression constants must be numeric")
        if isinstance(node, ast.Pow):
            exponent = node.right
            if isinstance(exponent, ast.Constant) and abs(float(exponent.value)) > 8:
                raise TelemetryError("exponent is too large")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCTIONS:
                raise TelemetryError("call to a disallowed function")
            if node.keywords:
                raise TelemetryError("keyword arguments are not allowed")
    compiled = compile(tree, "<derived-tag>", "eval")
    missing = sorted(
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id not in variables and node.id not in _ALLOWED_FUNCTIONS
    )
    if missing:
        raise TelemetryError(f"missing variables: {', '.join(missing)}")
    try:
        result = eval(compiled, {"__builtins__": {}}, {**_ALLOWED_FUNCTIONS, **variables})  # noqa: S307
    except (ArithmeticError, ValueError, TypeError, OverflowError) as exc:
        raise TelemetryError(f"expression evaluation failed: {exc}") from exc
    if isinstance(result, bool) or not isinstance(result, (int, float)):
        raise TelemetryError("expression must produce a number")
    if not math.isfinite(float(result)):
        raise TelemetryError("expression produced a non-finite value")
    return float(result)


# ---------------------------------------------------------------------------
# Operational math
# ---------------------------------------------------------------------------


def cooling_load_kw(tons: float) -> float:
    tons = _require_finite("tons", tons)
    if tons <= 0:
        raise TelemetryError("tons must be positive")
    return tons * TONS_TO_KW


def tons_from_load(load_pct: float, design_tons: float) -> float:
    load_pct = _require_finite("load_pct", load_pct)
    design_tons = _require_finite("design_tons", design_tons)
    if not 0 <= load_pct <= 100:
        raise TelemetryError("load_pct must be within [0, 100]")
    if design_tons <= 0:
        raise TelemetryError("design_tons must be positive")
    return design_tons * load_pct / 100.0


def kw_per_ton(power_kw: float, tons: float) -> float:
    power_kw = _require_finite("power_kw", power_kw)
    tons = _require_finite("tons", tons)
    if power_kw < 0:
        raise TelemetryError("power_kw must be non-negative")
    if tons <= 0:
        raise TelemetryError("tons must be positive")
    return power_kw / tons


def cop(tons: float, power_kw: float) -> float:
    power_kw = _require_finite("power_kw", power_kw)
    tons = _require_finite("tons", tons)
    if power_kw <= 0:
        raise TelemetryError("power_kw must be positive")
    if tons <= 0:
        raise TelemetryError("tons must be positive")
    return cooling_load_kw(tons) / power_kw


def delta_t(return_temp: float, supply_temp: float) -> float:
    return _require_finite("return_temp", return_temp) - _require_finite(
        "supply_temp", supply_temp
    )


def approach_temp(condenser_water_temp: float, supply_temp: float) -> float:
    """Condenser approach proxy: leaving condenser water minus CHW supply."""
    return _require_finite("condenser_water_temp", condenser_water_temp) - _require_finite(
        "supply_temp", supply_temp
    )


class DerivedTagEngine:
    """Evaluate ontology-defined derived tags over raw telemetry."""

    def __init__(self, ontology: Ontology, store: TagStore) -> None:
        self.ontology = ontology
        self.store = store
        self._cache: dict[str, TimeSeries] = {}
        self._evaluating: set[str] = set()

    def evaluate(self, tag_id: str) -> TimeSeries:
        if self.store.has(tag_id):
            return self.store.get(tag_id)
        if tag_id in self._cache:
            return self._cache[tag_id]
        if tag_id in self._evaluating:
            raise TelemetryError(f"derived tag cycle detected at {tag_id}")
        try:
            node = self.ontology.get(tag_id)
        except OntologyError as exc:
            raise TelemetryError(f"unknown tag: {tag_id}") from exc
        if node.type != "derived_tag":
            raise TelemetryError(f"{tag_id} is not a raw or derived tag")
        expression = node.attributes.get("expression")
        inputs = node.attributes.get("inputs")
        if not isinstance(expression, str) or not isinstance(inputs, dict) or not inputs:
            raise TelemetryError(f"{tag_id}: invalid derived tag definition")
        self._evaluating.add(tag_id)
        try:
            input_series = {
                variable: self.evaluate(source) for variable, source in sorted(inputs.items())
            }
            common = self._common_timestamps(input_series)
            samples: list[Sample] = []
            value_maps = {
                variable: {sample.timestamp: sample.value for sample in series}
                for variable, series in input_series.items()
            }
            for timestamp in common:
                variables = {
                    variable: float(value_maps[variable][timestamp])
                    for variable in value_maps
                }
                try:
                    value = safe_eval(expression, variables)
                except TelemetryError as exc:
                    raise TelemetryError(f"{tag_id}: {exc}") from exc
                samples.append(Sample(timestamp, value))
            if not samples:
                raise TelemetryError(f"{tag_id}: no aligned input samples")
            series = TimeSeries(tag_id, str(node.attributes.get("unit", "")), samples)
        finally:
            self._evaluating.discard(tag_id)
        self._cache[tag_id] = series
        return series

    def _common_timestamps(self, series_map: dict[str, TimeSeries]) -> list[datetime]:
        timestamp_sets = [
            {sample.timestamp for sample in series if sample.value is not None}
            for series in series_map.values()
        ]
        if not timestamp_sets:
            return []
        common = set.intersection(*timestamp_sets)
        return sorted(common)

    def latest(self, tag_id: str) -> Sample | None:
        return self.evaluate(tag_id).latest
