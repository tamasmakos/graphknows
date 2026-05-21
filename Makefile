# Graphknows — developer command centre
# ──────────────────────────────────────────────────────────────────
# Production:  make up
# Development: make dev
# Status:      make status
# Logs:        make logs   /   make logs SVC=agent
# Shell:       make shell SVC=agent
# ──────────────────────────────────────────────────────────────────
COMPOSE        := docker compose
COMPOSE_DEV    := docker compose -f docker-compose.yaml -f docker-compose.dev.yaml
SVC            ?=   # optional: set to limit logs/restart/shell to one service

.PHONY: up down dev restart build rebuild ps status logs logs-agent \
        logs-graphknows logs-search logs-frontend shell clean prune help

# ── Lifecycle ──────────────────────────────────────────────────────
up: ## Start all services (production images)
	$(COMPOSE) up -d
	@echo ""
	@$(MAKE) --no-print-directory status

down: ## Stop and remove containers (data volumes preserved)
	$(COMPOSE) down

dev: ## Start all services with source-mount hot-reload
	$(COMPOSE_DEV) up -d
	@echo ""
	@$(MAKE) --no-print-directory status

restart: ## Restart one or all services  →  make restart SVC=agent
	$(COMPOSE) restart $(SVC)

stop: ## Stop (but don't remove) one or all services  →  make stop SVC=search
	$(COMPOSE) stop $(SVC)

# ── Build ──────────────────────────────────────────────────────────
build: ## Build images (cached)  →  make build SVC=agent
	$(COMPOSE) build $(SVC)

rebuild: ## Force-rebuild images (no cache)  →  make rebuild SVC=agent
	$(COMPOSE) build --no-cache $(SVC)

# ── Observability ─────────────────────────────────────────────────
ps: ## Show running containers and their status
	$(COMPOSE) ps

status: ## Health check every service endpoint (infra + app)
	@bash scripts/health.sh

logs: ## Tail logs — all services or specific one  →  make logs SVC=agent
	$(COMPOSE) logs -f --tail=100 $(SVC)

logs-agent: ## Tail agent logs
	$(COMPOSE) logs -f --tail=100 agent

logs-graphknows: ## Tail graphknows logs
	$(COMPOSE) logs -f --tail=100 graphknows

logs-search: ## Tail search logs
	$(COMPOSE) logs -f --tail=100 search

logs-frontend: ## Tail frontend logs
	$(COMPOSE) logs -f --tail=100 frontend

# ── Debug ──────────────────────────────────────────────────────────
shell: ## Open a shell in a running container  →  make shell SVC=agent
	$(COMPOSE) exec $(SVC) /bin/bash

# ── Infra-only ────────────────────────────────────────────────────
infra-up: ## Start only infrastructure (Neo4j, pgvector, Memgraph)
	$(COMPOSE) up -d neo4j pgvector memgraph
	@echo "Waiting for Neo4j..."
	@$(COMPOSE) exec neo4j bash -c "until wget -q --spider http://localhost:7474; do sleep 2; done" 2>/dev/null || true
	@echo "Infrastructure ready."

infra-down: ## Stop only infrastructure containers
	$(COMPOSE) stop neo4j pgvector memgraph

# ── Observability stack ───────────────────────────────────────────
observability: ## Start Langfuse + its Postgres (profile: observability)
	$(COMPOSE) --profile observability up -d langfuse langfuse-db

# ── Cleanup ───────────────────────────────────────────────────────
clean: ## Stop containers and remove images built from this project
	$(COMPOSE) down --rmi local

prune: ## ⚠ Remove all stopped containers, unused images and networks
	docker system prune -f

# ── Local (bare-metal) ────────────────────────────────────────────
start-local: ## Start all services directly on the host (no Docker) — legacy
	@bash start.sh

start-local-no-tunnel: ## Start local services without Cloudflare tunnel — legacy
	@bash start.sh --no-tunnel

dev-local: ## Smart local dev start: checks infra, kills zombies, starts services, waits for health
	@bash scripts/dev-local.sh

dev-local-no-tunnel: ## dev-local without Cloudflare tunnel
	@bash scripts/dev-local.sh --no-tunnel

dev: doctor dev-local ## Full dev startup: preflight → infra → services (recommended)

doctor: ## Pre-flight: validate .env, Python, Docker, infra reachability, port availability
	@bash scripts/preflight.sh

setup: ## First-time setup: copy .env.example → .env if missing, then run doctor
	@if [ ! -f .env ]; then \
	  cp .env.example .env; \
	  echo "  Created .env from .env.example — fill in the REQUIRED keys before running make dev"; \
	  exit 1; \
	else \
	  echo "  .env already exists — running doctor..."; \
	  bash scripts/preflight.sh; \
	fi

logs: ## Tail all service logs in one view
	@tail -f /tmp/graphknows.log /tmp/agent.log /tmp/search.log /tmp/frontend.log 2>/dev/null \
	  || echo "No service logs found yet — run 'make dev' first"

# ── Help ──────────────────────────────────────────────────────────
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' | sort

.DEFAULT_GOAL := help
