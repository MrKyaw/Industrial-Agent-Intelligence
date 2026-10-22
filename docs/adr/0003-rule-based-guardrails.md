# ADR 0003 — Rule-based guardrails

## Status

Accepted.

## Context

Guardrails must be auditable and must not depend on another model's opinion.

## Decision

Implement guardrails as explicit rules over the answer, evidence ledger, and
context. Block on high-severity findings and replace the answer with a safe
escalation.

## Consequences

- Every failure is explainable.
- Rules can become brittle; eval cases keep them honest.
- Medium and low findings inform scoring without blocking.
