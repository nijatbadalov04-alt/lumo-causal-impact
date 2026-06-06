# Convenience targets. Windows users without `make` can run the underlying commands
# directly (shown in each recipe). The pipeline itself is pure Python (run_pipeline.py).
PY := .venv/Scripts/python.exe

.PHONY: help setup run run-gpu test lint format smoke all clean

help:
	@echo "setup    install pinned deps into .venv"
	@echo "run      python run_pipeline.py  (full pipeline M1->M5)"
	@echo "run-gpu  run only the GPU deep counterfactual stage"
	@echo "test     pytest"
	@echo "lint     ruff check"
	@echo "smoke    import-test every pipeline stage"
	@echo "all      run + test"

setup:
	$(PY) -m pip install -r requirements.txt
	@echo "GPU: install Blackwell torch separately -> pip install torch --index-url https://download.pytorch.org/whl/cu128"

run:
	$(PY) run_pipeline.py

run-gpu:
	$(PY) run_pipeline.py --only M4.deep_counterfactual

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests

format:
	$(PY) -m ruff format src tests

smoke:
	$(PY) -m pytest tests/test_imports.py

all: run test

clean:
	rm -rf results/figures/*.png results/tables/*.csv results/metrics/*.json
