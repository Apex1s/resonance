.DEFAULT_GOAL := help
PY ?= python3
PORT ?= 8000

.PHONY: help install train train-fast snapshot serve test lint format docker docker-run clean all

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install the package with dev extras
	$(PY) -m pip install -e ".[dev]"

train: ## Full training run: model selection, grid search, calibration, reports
	$(PY) -m resonance.train

train-fast: ## Training without the hyper-parameter search
	$(PY) -m resonance.train --fast

snapshot: ## Rebuild the dashboard's offline snapshot from the trained artefacts
	$(PY) -m resonance.export_snapshot

serve: ## Run the API + dashboard on http://localhost:$(PORT)
	$(PY) -m uvicorn resonance.api.main:app --reload --port $(PORT)

test: ## Run the test suite
	$(PY) -m pytest

lint: ## Static checks
	$(PY) -m ruff check src tests

format: ## Auto-fix lint and formatting
	$(PY) -m ruff check --fix src tests && $(PY) -m ruff format src tests

docker: ## Build the container image
	docker build -t resonance:latest .

docker-run: ## Run the container on http://localhost:$(PORT)
	docker run --rm -p $(PORT):8000 resonance:latest

all: train snapshot test ## Train, rebuild the snapshot, then test

clean: ## Remove caches and generated artefacts
	rm -rf .pytest_cache .ruff_cache **/__pycache__ models/*.joblib models/*.npz models/*.csv reports/metrics.json
