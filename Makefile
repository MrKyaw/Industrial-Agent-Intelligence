PYTHON ?= python3
SRC = src
DATA ?= data
REPORT_DIR ?= reports/evals

.PHONY: install test coverage evals demo serve validate clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	PYTHONPATH=$(SRC) $(PYTHON) -m pytest -q

coverage:
	PYTHONPATH=$(SRC) $(PYTHON) -m pytest --cov=src/iai --cov-report=term-missing --cov-report=xml:reports/coverage.xml

evals:
	PYTHONPATH=$(SRC) $(PYTHON) -m iai evals run --suite $(DATA)/evals/core.jsonl --min-pass-rate 1.0

demo:
	PYTHONPATH=$(SRC) $(PYTHON) -m iai demo

serve:
	PYTHONPATH=$(SRC) $(PYTHON) -m iai serve --port 8080

validate:
	PYTHONPATH=$(SRC) $(PYTHON) -m iai validate

clean:
	rm -rf .pytest_cache .coverage coverage.xml htmlcov reports/feedback reports/local
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
