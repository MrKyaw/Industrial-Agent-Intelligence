---
id: telemetry-quality
title: Telemetry freshness, gaps, and derived tags
tags: telemetry, stale, gaps, data-quality, derived-tags
trust: reference
---
# Telemetry quality rules

An agent must not tune controls against stale or gapped data.

## Freshness

- The reference freshness window for hourly telemetry is two hours.
- A stale series is historical context, not a live measurement.
- Disclose staleness in the answer and lower confidence.

## Gaps

- Detect gaps from the expected sample interval, not from row count alone.
- Do not interpolate across a gap for a safety decision.
- A gap-free but flat series can still be frozen; compare timestamps and quality flags.

## Derived tags

Derived tags must declare their expression, inputs, and unit in the ontology.
Computation happens only over aligned timestamps. Missing inputs produce no sample,
never a zero.
