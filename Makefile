# FantasyFootballBench — common tasks.
# Run `make help` for the list.

PYTHON ?= python3
VENV   ?= .venv
BIN    := $(VENV)/bin

.DEFAULT_GOAL := help
.PHONY: help setup test test-all check run draft season figures demo clean

help: ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Create .venv and install dependencies
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements-dev.txt
	@test -f .env || cp .env.example .env
	@echo ""
	@echo "Done. Add your OPENROUTER_API_KEY to .env, then run 'make run'."

test: ## Run the offline test suite (no API calls)
	$(BIN)/pytest -m "not network"

test-all: ## Run every test, including live model connectivity checks
	$(BIN)/pytest

check: ## Verify every model in config.json responds (uses API credits)
	$(BIN)/python main.py check

run: ## Run the full benchmark pipeline (draft + 17-week season)
	bash scripts/run_full_simulation.sh

draft: ## Run only the draft phase
	$(BIN)/python scripts/run_draft.py

season: ## Run only the season phase (requires an existing draft)
	$(BIN)/python scripts/simulate_season.py

figures: ## Regenerate analysis figures for the latest simulation
	$(BIN)/python scripts/generate_blog_figures.py

demo: ## Print sample DataHandler queries (no API calls)
	$(BIN)/python main.py demo

clean: ## Remove caches and build artifacts
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache figures icons
