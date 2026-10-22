---
id: alarm-flood-triage
title: Alarm flood triage procedure
tags: alarm, flood, deduplication, priority, operations
trust: reference
---
# Alarm flood triage

An alarm flood is a volume problem before it is a diagnosis problem. Repeated
codes on one asset should be collapsed into a cluster with a count and a first-seen
timestamp.

## Ranking rules

1. Safety-critical and critical-severity alarms outrank everything else.
2. Asset criticality is a multiplier, not an afterthought.
3. A repeated code is evidence of persistence, not independent evidence of severity.
4. Older resolved alarms stay in history but must not inflate the live priority.

## Operator sequence

1. Acknowledge the highest-priority cluster.
2. Verify the primary measurement against a second source.
3. Deduplicate the remaining codes before creating work orders.
4. Escalate safety-critical conditions to the duty engineer immediately.
5. Record which cluster was treated as the root cause and why.
