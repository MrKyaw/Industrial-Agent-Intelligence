# Test results

Generated from the repository on the current `main` branch.

## Summary

- Tests: 119 passed, 0 failed.
- Coverage: 79.94% line and branch (threshold 75%).
- Behavioural evals: 22/22 passed, mean score 0.997, 622 ms.
- Python: 3.12.8 (CI matrix 3.10, 3.11, 3.12).

## Coverage by module

| Module | Coverage |
| --- | ---: |
| `__init__.py` | 100% |
| `util.py` | 98% |
| `guardrails.py` | 94% |
| `feedback.py` | 84% |
| `retrieval.py` | 84% |
| `evals.py` | 83% |
| `alarms.py` | 83% |
| `app.py` | 81% |
| `models.py` | 81% |
| `tools.py` | 80% |
| `reasoning.py` | 77% |
| `ontology.py` | 77% |
| `telemetry.py` | 76% |
| `prompts.py` | 75% |
| `api.py` | 73% |
| `cli.py` | 64% |
| `report.py` | 69% |
| **Total** | **79.94%** |

## Reproduce

```bash
make test
make evals
```

Coverage XML is written to `reports/coverage.xml`. The scorecard is written to
`reports/evals/latest.{md,html,json}`.
