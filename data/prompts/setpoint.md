---
id: setpoint
version: 1.0.0
description: Setpoint proposal with bounds and confirmation semantics
variables: question, tag, proposed_value, bounds
---
Operator request: {{ question }}

Setpoint: {{ tag }}

Proposed value: {{ proposed_value }}

Configured bounds: {{ bounds }}

Explain whether the value is inside the operating envelope and deadband. A proposal
is not an action: state clearly that no write occurred and that engineer approval
plus a confirmation token is required. Never suggest bypassing an interlock.
