# Evaluation

## Suite

`data/evals/core.jsonl` contains 22 cases. Each case is a request plus a weighted
rubric. A case passes when the weighted score is at least the case threshold and no
blocking criterion fails.

## Graders

- `guardrails` — the full guardrail verdict passes.
- `grounding` — no ungrounded numeric claims.
- `citation_coverage` — diagnosis cites enough valid evidence.
- `tool_selection` — required tools used, forbidden tools absent.
- `escalation` / `clarification` / `status` — control-flow expectations.
- `keyword_recall` — domain terms present.
- `forbidden_phrases` — unsafe language absent.
- `latency` — wall-clock budget.
- `uncertainty` — hedging when confidence is low.
- `no_physical_write` — no non-simulated physical write.
- `disclosure` — stale or untrusted data is disclosed.
- `confidence` — confidence within an expected band.
- `output_contract` — required fields present.
- `tool_budget` — tool-call count within budget.

## Regression gating

`iai evals compare` fails when a previously passing case fails or its score drops
beyond the tolerance. CI runs the full suite with a 1.0 pass-rate threshold.

## Adding a case

Append one JSON object per line. Keep rubrics small and blocking only for safety.
Run `python -m iai evals run` and inspect the scorecard before committing.
