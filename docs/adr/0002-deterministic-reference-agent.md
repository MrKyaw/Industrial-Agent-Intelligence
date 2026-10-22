# ADR 0002 — Deterministic reference agent

## Status

Accepted.

## Context

Behavioural evals must be reproducible and explainable. An LLM in the loop would
make scores noisy and costly.

## Decision

Ship a deterministic, intent-routed reference agent. Keep an optional
OpenAI-compatible planner adapter as a seam, but do not require it.

## Consequences

- Evals are stable and cheap.
- The agent demonstrates reasoning patterns without hiding behind a model.
- Production use still needs an LLM planner and structured-output repair.
