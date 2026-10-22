---
id: refusal
version: 1.0.0
description: Safe refusal when policy or evidence blocks an action
variables: question, reason
---
Operator request: {{ question }}

Blocking reason: {{ reason }}

Refuse the unsafe portion of the request, preserve the legitimate operational goal,
and state the safe path forward. Do not repeat an instruction that conflicts with
policy as if it were valid. Escalate to the appropriate human role.
