# Interview walkthrough

Ten-minute tour of the repository for reviewers.

1. **Problem framing** — `README.md` and `docs/BEHAVIOR_SPEC.md`.
2. **Domain model** — `data/ontology.json`, `src/iai/ontology.py`, `docs/DOMAIN_MODEL.md`.
3. **Telemetry and derived tags** — `src/iai/telemetry.py`, `data/telemetry/`.
4. **Tool policy and audit** — `src/iai/tools.py`, `docs/TOOL_POLICY.md`.
5. **Agent runtime** — `src/iai/reasoning.py`, especially `_triage` and `_setpoint`.
6. **Guardrails** — `src/iai/guardrails.py`.
7. **Retrieval and injection defence** — `src/iai/retrieval.py`, `data/knowledge/field-note-vendor-access.md`.
8. **Evaluation** — `src/iai/evals.py`, `data/evals/core.jsonl`, `reports/evals/latest.html`.
9. **Feedback and drift** — `src/iai/feedback.py`.
10. **Tests** — `tests/`, `docs/TEST_RESULTS.md`.

## Talking points

- Why the agent is not an LLM wrapper: the reference runtime is deterministic and
  inspectable; an LLM planner is an optional adapter.
- Why guardrails are rule-based: every failure is explainable without a second model.
- Why evals are the contract: CI fails on behavioural regressions, not just unit tests.
- Trade-offs: heuristic root-cause ranking, approach proxy, fixed as-of time.
- What I would add next: real LLM planner with structured output repair, online
  telemetry ingestion, and a labelling UI.
