---
id: triage
version: 1.0.0
description: Alarm triage and root-cause investigation
variables: question, equipment, evidence
---
Operator question: {{ question }}

Target asset: {{ equipment }}

Evidence ledger:
{{ evidence }}

Produce a triage assessment. Cluster repeated alarms, separate measurement from
inference, and name at most two plausible mechanisms. Recommend non-destructive
checks first. Escalate if a safety-critical condition is active or the evidence is
insufficient for a confident diagnosis. Cite evidence ids for every claim.
