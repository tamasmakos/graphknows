#!/usr/bin/env bash
# scripts/generate-types.sh
# Generates TypeScript types from Python Pydantic models (placeholder).
# Replace with actual openapi-typescript codegen once services expose /openapi.json.
set -euo pipefail

GRAPHGEN_URL="${GRAPHGEN_URL:-http://localhost:8020}"
GRAPHRAG_URL="${GRAPHRAG_URL:-http://localhost:8010}"
OUT_DIR="packages/types/src/generated"

mkdir -p "$OUT_DIR"

echo "Generating types from graphgen OpenAPI spec..."
npx --yes openapi-typescript "${GRAPHGEN_URL}/openapi.json" -o "${OUT_DIR}/graphgen.ts" 2>/dev/null || \
  echo "  ⚠  graphgen not reachable — skipping (run services first)"

echo "Generating types from graphrag OpenAPI spec..."
npx --yes openapi-typescript "${GRAPHRAG_URL}/openapi.json" -o "${OUT_DIR}/graphrag.ts" 2>/dev/null || \
  echo "  ⚠  graphrag not reachable — skipping (run services first)"

echo "Done."
