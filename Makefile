# Graphknows — developer command centre
# ──────────────────────────────────────────────────────────────────
# First run:  make setup   (creates .env)
# Start:      make up       /   make up then make ps
# Logs:       make logs     /   make logs SVC=web
# Shell:      make shell SVC=web
# ──────────────────────────────────────────────────────────────────
COMPOSE := docker compose
SVC     ?=   # optional: limit logs/restart/stop/shell/build to one service

.PHONY: up down restart stop build rebuild ps logs shell \
        infra-up infra-down observability setup clean prune help

# ── Lifecycle ──────────────────────────────────────────────────────
up: ## Start all services
	$(COMPOSE) up -d --remove-orphans
	@echo ""
	@$(MAKE) --no-print-directory ps

down: ## Stop and remove containers (data volumes preserved)
	$(COMPOSE) down --remove-orphans

restart: ## Restart one or all services  →  make restart SVC=web
	$(COMPOSE) restart $(SVC)

stop: ## Stop (but don't remove) one or all services  →  make stop SVC=web
	$(COMPOSE) stop $(SVC)

# ── Build ──────────────────────────────────────────────────────────
build: ## Build images (cached)  →  make build SVC=web
	$(COMPOSE) build $(SVC)

rebuild: ## Force-rebuild images (no cache)  →  make rebuild SVC=web
	$(COMPOSE) build --no-cache $(SVC)

# ── Observability ─────────────────────────────────────────────────
ps: ## Show running containers and their status
	$(COMPOSE) ps

logs: ## Tail logs — all services or one  →  make logs SVC=web
	$(COMPOSE) logs -f --tail=100 $(SVC)

shell: ## Open a shell in a running container  →  make shell SVC=web
	$(COMPOSE) exec $(SVC) /bin/bash

# ── Infra-only ────────────────────────────────────────────────────
infra-up: ## Start only the database (ArcadeDB)
	$(COMPOSE) up -d arcadedb

infra-down: ## Stop the database
	$(COMPOSE) stop arcadedb

# ── Observability stack ───────────────────────────────────────────
observability: ## Start Langfuse + its Postgres (profile: observability)
	$(COMPOSE) --profile observability up -d langfuse langfuse-db

# ── Setup / cleanup ───────────────────────────────────────────────
setup: ## First-time setup: copy .env.example → .env if missing
	@if [ ! -f .env ]; then \
	  cp .env.example .env; \
	  echo "  Created .env from .env.example — fill in the REQUIRED keys before 'make up'"; \
	else \
	  echo "  .env already exists"; \
	fi

clean: ## Stop containers and remove images built from this project
	$(COMPOSE) down --rmi local

prune: ## ⚠ Remove all stopped containers, unused images and networks
	docker system prune -f

# ── Help ──────────────────────────────────────────────────────────
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' | sort

.DEFAULT_GOAL := help
