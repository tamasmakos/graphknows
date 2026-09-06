# Graphknows — developer command centre
# ──────────────────────────────────────────────────────────────────
# First run:  make setup   (creates .env)
# Start:      make up
# Shell:      make shell SVC=workspace
# ──────────────────────────────────────────────────────────────────
COMPOSE := docker compose
CONTAINER ?= graphknows-workspace
SVC     ?= workspace   # service `make shell`/`logs`/`restart`/`stop`/`build` targets

.PHONY: up down restart stop build rebuild ps logs shell infra-up infra-down setup clean prune \
        test test-integration lint typecheck arch build-wheel smoke bake eval-smoke sweep ci gate help

# ── Lifecycle ──────────────────────────────────────────────────────
up: ## Start all services
	$(COMPOSE) up -d --remove-orphans
	@echo ""
	@$(MAKE) --no-print-directory ps

down: ## Stop and remove containers (data volumes preserved)
	$(COMPOSE) down --remove-orphans

restart: ## Restart one or all services  →  make restart SVC=workspace
	$(COMPOSE) restart $(SVC)

stop: ## Stop (but don't remove) one or all services  →  make stop SVC=workspace
	$(COMPOSE) stop $(SVC)

ps: ## Show running containers and their status
	$(COMPOSE) ps

logs: ## Tail logs — all services or one  →  make logs SVC=workspace
	$(COMPOSE) logs -f --tail=100 $(SVC)

shell: ## Open a shell in a running container  →  make shell SVC=workspace
	$(COMPOSE) exec $(SVC) /bin/bash

# ── Build ──────────────────────────────────────────────────────────
build: ## Build images (cached)  →  make build SVC=workspace
	$(COMPOSE) build $(SVC)

rebuild: ## Force-rebuild images (no cache)  →  make rebuild SVC=workspace
	$(COMPOSE) build --no-cache $(SVC)

# ── Infra-only ────────────────────────────────────────────────────
infra-up: ## Start only the database (ArcadeDB)
	$(COMPOSE) up -d arcadedb

infra-down: ## Stop the database
	$(COMPOSE) stop arcadedb

# ── Setup / cleanup ───────────────────────────────────────────────
clean: ## Stop containers and remove images built from this project
	$(COMPOSE) down --rmi local

prune: ## ⚠ Remove all stopped containers, unused images and networks
	docker system prune -f

setup: ## First-time setup: copy .env.example → .env if missing
	@if [ ! -f .env ]; then \
	  cp .env.example .env; \
	  echo "  Created .env from .env.example — fill in the REQUIRED keys before 'make up'"; \
	else \
	  echo "  .env already exists"; \
	fi

# ── Quality gates (local, no Docker) ──────────────────────────────
test: ## Run the unit test suite (integration tests excluded)
	pytest -q -m "not integration"

test-integration: ## Run integration tests (needs a live ArcadeDB — make infra-up)
	pytest -q -m integration

lint: ## Ruff lint the package + tests
	ruff check graphknows tests

typecheck: ## Mypy the package
	mypy graphknows

arch: ## Enforce module boundaries (import-linter)
	lint-imports

build-wheel: ## Build the wheel and assert it stays small + ships py.typed
	uv build --wheel
	@python -c "import glob,os,zipfile; w=sorted(glob.glob('dist/*.whl'))[-1]; \
	  z=zipfile.ZipFile(w); assert any('py.typed' in n for n in z.namelist()), 'py.typed missing'; \
	  kb=os.path.getsize(w)/1024; assert kb < 1024, f'wheel too large: {kb:.0f} KB'; \
	  print(f'wheel OK: {os.path.basename(w)} ({kb:.0f} KB, py.typed present)')"

smoke: ## Core-only import smoke: import graphknows + client with no heavy deps
	pytest -q tests/api/test_lazy_core.py tests/api/test_public_surface.py

ci: gate build-wheel ## Run the full local CI gate (scripts/gate.sh + the wheel checks)

gate: ## Run the pre-push gate exactly as the hook runs it
	bash scripts/gate.sh

# Report-only on purpose. vulture false-positives on pydantic fields, protocol
# methods and lazy __getattr__, so it is a sweep a human reads, never a gate.
# Its job is the one thing no gate catches: code that is wired to nothing.
sweep: ## Dead-code sweep (report only, run every few weeks)
	docker exec $(CONTAINER) sh -c "cd /app && uvx vulture graphknows \
	  --min-confidence 60"

# ── Model prep ─────────────────────────────────────────────────────
# Run once after `make up` on a fresh model_cache volume, or after adding/
# changing a model in graphknows/settings.py — populates model_cache so the
# first real ingest doesn't pay the multi-GB download inline.
bake: ## Pre-warm the GLiNER/sentence-transformers/spaCy/NLTK model cache
	docker exec $(CONTAINER) sh -c "cd /app && python scripts/bake_models.py"

eval-smoke: ## End-to-end acceptance: one LoCoMo conversation via the MCP client (needs services up)
	python -m evaluation --limit 1

# ── Help ──────────────────────────────────────────────────────────
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' | sort

.DEFAULT_GOAL := help
