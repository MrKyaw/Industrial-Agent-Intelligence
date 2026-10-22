---
id: efficiency
version: 1.0.0
description: Chiller efficiency review
variables: question, equipment, evidence
---
Operator question: {{ question }}

Target asset: {{ equipment }}

Evidence ledger:
{{ evidence }}

Assess efficiency using computed COP, kW/ton, and chilled-water delta-T. Separate
load effects from heat-rejection effects. State whether the snapshot is sufficient
for a conclusion. Do not recommend a setpoint change as a first action unless the
evidence specifically supports it.
