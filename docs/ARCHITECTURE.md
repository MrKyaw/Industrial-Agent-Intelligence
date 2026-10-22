# Architecture

## Components

- `Ontology` — typed knowledge graph of sites, equipment, sensors, derived tags,
  setpoints, and alarm definitions. Loaded from JSON and validated.
- `TagStore` + `DerivedTagEngine` — raw telemetry and safe arithmetic over aligned
  timestamps. Derived tags declare their expression and inputs.
- `AlarmStore` + triage — ingestion, deduplication, clustering, priority scoring,
  flood indexing, and heuristic root-cause ranking.
- `ToolRegistry`, `PolicyEngine`, `ToolExecutor`, `AuditLog` — contracts, role and
  schema checks, rate limits, confirmation tokens, and a SHA-256 hash chain.
- `HybridRetriever` + `ContextPacker` — BM25 retrieval with ontology boosts and
  fenced, budgeted, injection-neutralised context.
- `PromptLibrary` — versioned prompt templates with strict rendering.
- `AgentRuntime` — intent routing, investigation handlers, evidence ledger,
  guardrails, and safe fallback.
- `EvalRunner`, `compare_reports`, report renderers — behavioural evaluation and
  regression gating.
- `FeedbackStore` — SQLite feedback, labels, and drift detection.
- `Application` — wires everything together and exposes default tools.
- `api` / `cli` — read-only HTTP API, dashboard, and command-line interface.

## Request lifecycle

1. `AgentRequest` is validated.
2. `detect_intent` routes the question.
3. The handler executes a bounded investigation plan through `ToolExecutor`.
4. Every observation becomes an `Evidence` item with a stable id.
5. `AgentAnswer` is assembled with citations and a trace.
6. `evaluate_guardrails` runs; a blocked answer is replaced with a safe escalation.
7. Disclosures are appended for stale telemetry or untrusted text.

## Failure modes

- Unknown asset or ambiguous request: clarification, no tools, no writes.
- Missing telemetry: the agent refuses rather than estimates.
- Stale or gapped data: disclosed and confidence reduced.
- Guardrail failure: safe fallback with no physical action.
- Tool failure: recorded in the trace; the answer degrades gracefully.
- Cross-site request: refused to avoid mixing facility context.

## Extension points

- Add a tool by registering a `ToolSpec` with a handler and schema.
- Add a derived tag by extending the ontology and telemetry series.
- Add an eval case by appending one JSON line to `data/evals/core.jsonl`.
- Add a knowledge document by dropping a Markdown file into `data/knowledge/`.
