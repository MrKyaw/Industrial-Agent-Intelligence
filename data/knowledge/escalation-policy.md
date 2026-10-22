---
id: escalation-policy
title: Escalation and uncertainty policy
tags: escalation, safety, uncertainty, human-in-the-loop
trust: reference
---
# Escalation policy

Escalate to a human when any of the following is true:

- A safety-critical or critical-severity alarm is active.
- The requested action changes physical control.
- Evidence conflicts across tools or time windows.
- The agent cannot identify the asset or the symptom.
- Confidence is below the operating threshold for the intent.

## How to escalate

State the condition, the evidence that triggered it, and the specific human role
needed. Do not dilute an escalation with speculative causes. Do not claim an action
was completed unless a non-simulated write returned success.

## Uncertainty language

Prefer "likely", "consistent with", and "cannot confirm" over false certainty.
Distinguish a measurement from an inference, and quote the limiting evidence.
