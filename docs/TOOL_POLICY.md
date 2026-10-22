# Tool policy

## Risk tiers

- `read` — query telemetry, alarms, knowledge, ontology, and proposals.
- `low_write` — create a work order (simulated in demo mode).
- `high_write` — reserved for future use.
- `physical_control` — apply a setpoint. Disabled by default.

## Roles

`viewer` < `operator` < `engineer` < `admin`. A tool requires a minimum role.

## Confirmations

Physical-control tools require the `approve_physical_control` confirmation token in
the policy context. The token is separate from the proposal id.

## Rate limits

Write tools are limited per session per hour (default 10). Physical control is also
gated by `IAI_ENABLE_WRITES=1`.

## Interlocks

Setpoint proposals are checked against configured min, max, and deadband. Values
outside the range or inside the deadband are rejected.

## Audit

Every request, result, and guardrail block is appended to a SHA-256 hash chain.
`AuditLog.verify()` detects tampering.
