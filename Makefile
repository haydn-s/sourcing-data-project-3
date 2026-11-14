# Developer commands. Run `make` or `make help` to list them.
# CI calls the same targets, so passing locally means passing in CI.
# Windows: run these from WSL or Git Bash.

COMMITLINT_VERSION := 21.2.4
COMMITLINT := npx --yes -p @commitlint/cli@$(COMMITLINT_VERSION) -p @commitlint/config-conventional@$(COMMITLINT_VERSION) commitlint
FIXERS := ruff-check ruff-format end-of-file-fixer trailing-whitespace mixed-line-ending
FROM ?= origin/main
TO ?= HEAD
VENV := PyDev
VENV_PYTHON := 3.12
CONFIG ?= config.yaml
IMAGE ?= project3
PLATFORM ?= linux/amd64

.DEFAULT_GOAL := help
.PHONY: help venv setup format lint test lint-commits lint-title pr-ready \
	demo-data ingest prepare push data train serve docker-build docker-smoke

help: ## List available commands
	@awk 'BEGIN {FS = ":.*## "} /^##@/ {printf "\n%s\n", substr($$0, 5)} /^[a-zA-Z_-]+:.*## / {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

##@ Setup and checks

venv: ## Create or update the PyDev virtual environment from requirements-pydev.txt
	@if [ ! -d $(VENV) ]; then \
	  if command -v uv >/dev/null 2>&1; then uv venv --seed --python $(VENV_PYTHON) $(VENV); \
	  else python$(VENV_PYTHON) -m venv $(VENV); fi; \
	fi
	@if command -v uv >/dev/null 2>&1; then uv pip install --python $(VENV)/bin/python -r requirements-pydev.txt; \
	else $(VENV)/bin/pip install -r requirements-pydev.txt; fi
	@echo ""
	@echo "Done. Activate it with: source $(VENV)/bin/activate, then run: make setup"

setup: ## Install dependencies and git hooks (once per clone, inside your venv)
	@if [ -z "$$VIRTUAL_ENV" ]; then echo "Warning: no virtualenv is active. Run make venv, then source $(VENV)/bin/activate."; fi
	pip install -r requirements-dev.txt
	@if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
	pre-commit install

format: ## Auto-fix formatting and lint issues in all files
	@for hook in $(FIXERS); do pre-commit run $$hook --all-files || true; done
	@git status --short

lint: ## Run every pre-commit hook on all files (what CI runs)
	pre-commit run --all-files --show-diff-on-failure

test: ## Run the tests (passes with a notice until tests exist)
	@pytest; status=$$?; \
	if [ $$status -eq 5 ]; then echo "No tests collected yet."; exit 0; fi; \
	exit $$status

##@ Data pipeline (data_pipeline/): sources -> processed table in Cloud Storage

ingest: ## Data 1/3: fetch every source in the config and save raw snapshots
	python -m data_pipeline.ingest --config $(CONFIG)

prepare: ## Data 2/3: clean, join, and add features, writing data.local_path
	python -m data_pipeline.prepare --config $(CONFIG)

push: ## Data 3/3: upload the processed data to data.remote_uri
	python -m data_pipeline.push --config $(CONFIG)

data: ## Run all three data stages in order
	@# One after another, even under make -j (prerequisites would run in parallel).
	@$(MAKE) --no-print-directory ingest
	@$(MAKE) --no-print-directory prepare
	@$(MAKE) --no-print-directory push

##@ ML pipeline (ml_pipeline/): processed table -> trained model

demo-data: ## Generate the synthetic demo dataset (data/demo/demo.csv)
	python -m ml_pipeline.demo_data

train: ## Train a model: make train CONFIG=configs/linear.yaml (default: config.yaml)
	python -m ml_pipeline.train --config $(CONFIG)

##@ API and Docker

serve: ## Run the API locally with auto-reload (serves models/latest unless MODEL_URI is set)
	uvicorn api.main:create_app --factory --reload --port 8080

docker-build: ## Build the image for Lambda (linux/amd64, no attestations)
	docker buildx build --platform $(PLATFORM) --provenance=false --sbom=false -t $(IMAGE) --load .

docker-smoke: docker-build ## Build the image, train on demo data inside it, and test the API
	IMAGE=$(IMAGE) sh scripts/docker_smoke_test.sh

##@ Commits and pull requests

lint-commits: ## Lint commit messages in FROM..TO (default: origin/main..HEAD)
	$(COMMITLINT) --from $(FROM) --to $(TO) --verbose

lint-title: ## Lint a PR title: make lint-title PR_TITLE="feat(api): add /predict"
	@if [ -z "$$PR_TITLE" ]; then echo 'Set PR_TITLE, e.g. make lint-title PR_TITLE="feat(api): add /predict"'; exit 1; fi
	@printf '%s\n' "$$PR_TITLE" | $(COMMITLINT) --verbose

pr-ready: ## Check everything CI checks before you open a PR (optional: PR_TITLE="...")
	@if [ "$$(git rev-parse --abbrev-ref HEAD)" = "main" ]; then \
	  echo "You're on main. Create a branch first: git switch -c feat/<task>"; exit 1; fi
	@for hook in pre-commit commit-msg; do \
	  if [ ! -f "$$(git rev-parse --git-path hooks/$$hook)" ]; then echo "Git hooks aren't installed. Run: make setup"; exit 1; fi; \
	done
	@if [ -n "$$(git status --porcelain)" ]; then \
	  echo "Warning: you have uncommitted changes. They won't be in the PR until you commit them."; fi
	@git fetch --quiet origin main
	@behind=$$(git rev-list --count HEAD..origin/main); \
	if [ "$$behind" -gt 0 ]; then echo "Warning: your branch is $$behind commit(s) behind main. Consider: git merge origin/main"; fi
	@$(MAKE) --no-print-directory lint
	@$(MAKE) --no-print-directory test
	@$(MAKE) --no-print-directory lint-commits
	@if [ -n "$$PR_TITLE" ]; then $(MAKE) --no-print-directory lint-title; fi
	@echo ""
	@echo "Ready for a PR. Next: git push -u origin HEAD, then open the PR with a Conventional Commit title."
