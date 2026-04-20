# ── Stage 1: deps ────────────────────────────────────────────────────────────
FROM node:18-alpine AS deps
RUN corepack enable && corepack prepare pnpm@9.15.4 --activate
WORKDIR /workspace

COPY package.json pnpm-workspace.yaml pnpm-lock.yaml ./
COPY apps/web/package.json ./apps/web/package.json
COPY packages/types/package.json ./packages/types/package.json
COPY packages/ui/package.json ./packages/ui/package.json
COPY packages/tsconfig/package.json ./packages/tsconfig/package.json

RUN pnpm install --frozen-lockfile

# ── Stage 2: builder ─────────────────────────────────────────────────────────
FROM node:18-alpine AS builder
RUN corepack enable && corepack prepare pnpm@9.15.4 --activate
WORKDIR /workspace

COPY --from=deps /workspace/node_modules ./node_modules
COPY --from=deps /workspace/apps/web/node_modules ./apps/web/node_modules
COPY . .

ENV NEXT_TELEMETRY_DISABLED=1

RUN pnpm --filter @graphknows/types build || true
RUN pnpm --filter @graphknows/web build

# ── Stage 3: runner ───────────────────────────────────────────────────────────
FROM node:18-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1

RUN addgroup --system --gid 1001 nodejs && adduser --system --uid 1001 nextjs

COPY --from=builder /workspace/apps/web/public ./public
COPY --from=builder --chown=nextjs:nodejs /workspace/apps/web/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /workspace/apps/web/.next/static ./apps/web/.next/static

USER nextjs
EXPOSE 3000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD wget -qO- http://localhost:3000/api/v1 2>/dev/null || exit 1

CMD ["node", "apps/web/server.js"]
