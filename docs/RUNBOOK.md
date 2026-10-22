# Runbook

## Local development

```bash
python -m pip install -e ".[dev]"
make validate
make test
make evals
```

## Serving

```bash
python -m iai serve --port 8080
```

Bind to a non-loopback address only with `--token` or `IAI_API_TOKEN`.

## Feedback

```bash
python -m iai feedback add --request-id req-1 --rating 5 --site site.dc-lon-01
python -m iai feedback drift --recent 50 --baseline 200
```

## Regenerating data

```bash
python scripts/generate_demo_data.py
```

## Troubleshooting

- `Could not locate the data directory`: set `IAI_DATA_DIR` or run from the repository root.
- Eval gate failure: inspect `reports/evals/latest.md` for the failing case and grader detail.
- Physical writes disabled: set `IAI_ENABLE_WRITES=1` and run as an engineer with confirmation.
- Stale telemetry: check gateway health and the `sample_interval_s` in the ontology.
