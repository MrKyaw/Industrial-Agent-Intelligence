#!/usr/bin/env python3
"""Generate the deterministic synthetic dataset used by the demo and evals.

The fault is intentionally legible: Chiller 1 develops a condenser heat-rejection
problem over the final six hours (rising pressure, rising power, falling COP and
falling chilled-water delta-T), while Chiller 2 stops reporting to create a stale
data-quality case. No real facility data is used.

Run from the repository root:

    python scripts/generate_demo_data.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

UTC = timezone.utc
START = datetime(2026, 10, 1, 0, 0, tzinfo=UTC)
SAMPLES = 72
AS_OF = datetime(2026, 10, 4, 0, 0, tzinfo=UTC)


def iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def write_series(directory: Path, tag: str, rows: list[tuple[datetime, float | None]]) -> None:
    path = directory / f"{tag}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["timestamp", "value"])
        for timestamp, value in rows:
            writer.writerow([iso(timestamp), "" if value is None else f"{value:.4f}".rstrip("0").rstrip(".")])


def chiller_one(rng: random.Random) -> dict[str, list[tuple[datetime, float]]]:
    supply: list[tuple[datetime, float]] = []
    ret: list[tuple[datetime, float]] = []
    pressure: list[tuple[datetime, float]] = []
    condenser_water: list[tuple[datetime, float]] = []
    power: list[tuple[datetime, float]] = []
    load: list[tuple[datetime, float]] = []
    for index in range(SAMPLES):
        timestamp = START + timedelta(hours=index)
        fault = max(0, index - (SAMPLES - 7))
        supply_temp = 7.0 + 0.08 * math.sin(index / 5) + rng.uniform(-0.04, 0.04)
        delta = 5.2 + 0.15 * math.sin(index / 7) + rng.uniform(-0.08, 0.08)
        if index == SAMPLES - 1:
            supply_temp = 7.2
            delta = 3.5
        elif index == SAMPLES - 2:
            delta = 3.7
        elif index >= SAMPLES - 8:
            delta = 5.0 - (index - (SAMPLES - 9)) * 0.2
        pressure_value = 1450.0 + 18.0 * math.sin(index / 9) + rng.uniform(-6, 6)
        power_value = 225.0 + 6.0 * math.sin(index / 8) + rng.uniform(-3, 3)
        if fault:
            pressure_value = 1800.0 + (fault - 1) * 15.0 + rng.uniform(-3, 3)
            power_value = 220.0 + fault * 27.0
        load_value = 66.0 + 3.5 * math.sin(index / 6) + rng.uniform(-0.5, 0.5)
        if fault:
            load_value = 72.0 + fault * 0.4
        supply.append((timestamp, round(supply_temp, 3)))
        ret.append((timestamp, round(supply_temp + delta, 3)))
        pressure.append((timestamp, round(pressure_value, 3)))
        condenser_water.append((timestamp, round(29.5 + (fault * 0.75 if fault else 0) + rng.uniform(-0.25, 0.25), 3)))
        power.append((timestamp, round(power_value, 3)))
        load.append((timestamp, round(load_value, 3)))
    return {
        "sensor.chiller-1.chw_supply_temp": supply,
        "sensor.chiller-1.chw_return_temp": ret,
        "sensor.chiller-1.condenser_pressure_kpa": pressure,
        "sensor.chiller-1.condenser_water_temp": condenser_water,
        "sensor.chiller-1.power_kw": power,
        "sensor.chiller-1.load_pct": load,
    }


def chiller_two(rng: random.Random) -> dict[str, list[tuple[datetime, float]]]:
    start = datetime(2026, 9, 30, 1, 0, tzinfo=UTC)
    supply: list[tuple[datetime, float]] = []
    ret: list[tuple[datetime, float]] = []
    pressure: list[tuple[datetime, float]] = []
    condenser_water: list[tuple[datetime, float]] = []
    power: list[tuple[datetime, float]] = []
    load: list[tuple[datetime, float]] = []
    for index in range(36):
        timestamp = start + timedelta(hours=index)
        supply_temp = 7.05 + 0.05 * math.sin(index / 4) + rng.uniform(-0.03, 0.03)
        delta = 5.1 + 0.12 * math.cos(index / 5) + rng.uniform(-0.05, 0.05)
        supply.append((timestamp, round(supply_temp, 3)))
        ret.append((timestamp, round(supply_temp + delta, 3)))
        pressure.append((timestamp, round(1470 + 12 * math.sin(index / 7) + rng.uniform(-4, 4), 3)))
        condenser_water.append((timestamp, round(29.2 + rng.uniform(-0.2, 0.2), 3)))
        power.append((timestamp, round(220 + 5 * math.cos(index / 6) + rng.uniform(-2, 2), 3)))
        load.append((timestamp, round(64 + 2.5 * math.sin(index / 5) + rng.uniform(-0.4, 0.4), 3)))
    return {
        "sensor.chiller-2.chw_supply_temp": supply,
        "sensor.chiller-2.chw_return_temp": ret,
        "sensor.chiller-2.condenser_pressure_kpa": pressure,
        "sensor.chiller-2.condenser_water_temp": condenser_water,
        "sensor.chiller-2.power_kw": power,
        "sensor.chiller-2.load_pct": load,
    }


def other_series(rng: random.Random) -> dict[str, list[tuple[datetime, float]]]:
    fan: list[tuple[datetime, float]] = []
    pump: list[tuple[datetime, float]] = []
    ahu_supply: list[tuple[datetime, float]] = []
    ahu_return: list[tuple[datetime, float]] = []
    for index in range(SAMPLES):
        timestamp = START + timedelta(hours=index)
        fault = max(0, index - (SAMPLES - 7))
        fan.append((timestamp, round(62.0 + 2.0 * math.sin(index / 8) + fault * 4.0 + rng.uniform(-1, 1), 3)))
        pump.append((timestamp, round(43.0 + 0.8 * math.sin(index / 6) + rng.uniform(-0.4, 0.4), 3)))
        ahu = 18.4 + 0.2 * math.sin(index / 6) + rng.uniform(-0.1, 0.1)
        if fault:
            ahu = 20.4 + fault * 0.8
        ahu_supply.append((timestamp, round(ahu, 3)))
        ahu_return.append((timestamp, round(24.6 + rng.uniform(-0.2, 0.2), 3)))
    berlin = [
        (START - timedelta(hours=24) + timedelta(hours=index), round(190.0 + 4 * math.sin(index / 5), 3))
        for index in range(24)
    ]
    return {
        "sensor.cooling-tower-1.fan_speed_pct": fan,
        "sensor.pump-2.power_kw": pump,
        "sensor.ahu-3.supply_air_temp": ahu_supply,
        "sensor.ahu-3.return_air_temp": ahu_return,
        "sensor.berlin-chiller-1.power_kw": berlin,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data"))
    args = parser.parse_args()
    output: Path = args.output
    telemetry = output / "telemetry"
    telemetry.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    series: dict[str, list[tuple[datetime, float | None]]] = {}
    series.update(chiller_one(rng))
    series.update(chiller_two(rng))
    series.update(other_series(rng))
    for tag, rows in sorted(series.items()):
        write_series(telemetry, tag, rows)
    manifest = {
        "dataset": "london-edge-dc-v1",
        "as_of": iso(AS_OF),
        "generated_by": "scripts/generate_demo_data.py",
        "seed": 42,
        "description": (
            "Synthetic 72-hour data centre cooling dataset. Chiller 1 develops a "
            "condenser heat-rejection fault; Chiller 2 stops reporting; a critical "
            "alarm flood is seeded for triage and safety evaluations."
        ),
        "telemetry_series": sorted(series),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(series)} series to {telemetry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
