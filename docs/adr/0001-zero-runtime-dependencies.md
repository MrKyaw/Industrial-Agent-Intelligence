# ADR 0001 — Zero runtime dependencies

## Status

Accepted.

## Context

The project must run in a reviewer environment without network access or package
conflicts.

## Decision

Use only the Python standard library at runtime. Optional dev dependencies are
limited to pytest and pytest-cov.

## Consequences

- Easy to install and review.
- More code to maintain (BM25, HTTP server, SQLite wrappers).
- No external framework lock-in.
