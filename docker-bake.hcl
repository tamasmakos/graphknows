# docker-bake.hcl — Docker Buildx Bake configuration
#
# Used by the CD workflow to build all four application images in parallel
# and push them to GHCR with consistent tags.
#
# Usage (CD workflow):
#   TAG=v1.2.3 IMAGE_PREFIX=ghcr.io/owner/graphknows \
#     docker buildx bake --file docker-bake.hcl --push
#
# Usage (local test build):
#   IMAGE_PREFIX=graphknows docker buildx bake --file docker-bake.hcl --load

variable "TAG" {
  default = "dev"
}

variable "IMAGE_PREFIX" {
  default = "graphknows"
}

group "default" {
  targets = ["graphknows", "agent", "search", "frontend"]
}

target "graphknows" {
  context    = "."
  dockerfile = "services/graphknows/Dockerfile"
  tags = [
    "${IMAGE_PREFIX}-graphknows:${TAG}",
    "${IMAGE_PREFIX}-graphknows:latest",
  ]
  labels = {
    "org.opencontainers.image.title"       = "Graphknows HybridRAG"
    "org.opencontainers.image.description" = "Graphknows HybridRAG MCP service (Neo4j + pgvector)"
  }
  platforms = ["linux/amd64"]
}

target "agent" {
  context    = "."
  dockerfile = "services/agent/Dockerfile"
  tags = [
    "${IMAGE_PREFIX}-agent:${TAG}",
    "${IMAGE_PREFIX}-agent:latest",
  ]
  labels = {
    "org.opencontainers.image.title"       = "Graphknows Agent"
    "org.opencontainers.image.description" = "LangGraph StateGraph agent + WebSocket API"
  }
  platforms = ["linux/amd64"]
}

target "search" {
  context    = "."
  dockerfile = "services/search/Dockerfile"
  tags = [
    "${IMAGE_PREFIX}-search:${TAG}",
    "${IMAGE_PREFIX}-search:latest",
  ]
  labels = {
    "org.opencontainers.image.title"       = "Graphknows Search"
    "org.opencontainers.image.description" = "Document search and indexing service"
  }
  platforms = ["linux/amd64"]
}

target "frontend" {
  context    = "apps/web"
  dockerfile = "apps/web/Dockerfile"
  tags = [
    "${IMAGE_PREFIX}-frontend:${TAG}",
    "${IMAGE_PREFIX}-frontend:latest",
  ]
  labels = {
    "org.opencontainers.image.title"       = "Graphknows Frontend"
    "org.opencontainers.image.description" = "WebSocket chat UI"
  }
  platforms = ["linux/amd64"]
}
