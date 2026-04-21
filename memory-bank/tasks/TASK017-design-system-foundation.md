# TASK017 — Design System Foundation (shadcn + Kibo UI)

**Status:** Pending  
**Added:** 2026-04-21  
**Updated:** 2026-04-21  
**Phase:** D — UI Rebuild

---

## Original Request
The frontend has no component library. `packages/ui` exports a single `__placeholder = true`. No shadcn/ui is installed. No Kibo UI components are registered. Pages mix Tailwind classes with raw CSS variable `style={{...}}` hacks. The design language doesn't match the Linear/Notion aesthetic specified in the project brief.

This task establishes the design system so that TASK018–021 can build pages using consistent, high-quality components.

---

## Thought Process
The stack decision is:
- **shadcn/ui** as the primitive layer (accessible, unstyled-by-default, Radix-based, copied into source so we own it)
- **Kibo UI** as the AI-specific component layer (AI chat input/message, file dropzone, timeline for reasoning steps, command palette)
- **next-themes** for dark/light mode with zero flash (CSS class strategy)
- **sonner** for toasts
- **Inter** (body) + **JetBrains Mono** (code/citations) via `next/font`

The `packages/ui` package should re-export shadcn primitives so that `apps/web` can import from `@graphknows/ui` rather than direct `@/components/ui`. This keeps the monorepo boundary clean.

### Why shadcn init first?
`shadcn/ui` init writes `components.json`, sets up the `@/components/ui` path alias, and sets the base colour. All subsequent `shadcn add <component>` calls depend on `components.json` existing.

### Colour tokens (Linear-inspired)
```
background:  hsl(0 0% 100%)     / hsl(240 6% 9%)
foreground:  hsl(240 10% 4%)    / hsl(0 0% 98%)
card:        hsl(0 0% 100%)     / hsl(240 6% 11%)
border:      hsl(240 6% 91%)    / hsl(240 6% 17%)
primary:     hsl(243 75% 59%)   / hsl(243 75% 65%)   ← indigo
muted:       hsl(240 5% 96%)    / hsl(240 5% 15%)
```

---

## Implementation Plan

1. **Run shadcn init in `apps/web/`**:
   ```bash
   pnpm dlx shadcn@latest init
   # → style: default, baseColour: stone, cssVariables: yes, rsc: yes
   # → writes apps/web/components.json, apps/web/src/app/globals.css (extends)
   ```

2. **Install shadcn primitives** (add more as TASK018–021 need them):
   ```bash
   pnpm dlx shadcn@latest add button input badge card separator sheet sidebar tooltip hover-card accordion tabs table scroll-area skeleton dialog command
   ```

3. **Install Kibo UI registries**:
   ```bash
   # From https://www.kibo-ui.com/components
   pnpm dlx shadcn@latest add https://www.kibo-ui.com/r/ai-input.json
   pnpm dlx shadcn@latest add https://www.kibo-ui.com/r/ai-message.json
   pnpm dlx shadcn@latest add https://www.kibo-ui.com/r/dropzone.json
   pnpm dlx shadcn@latest add https://www.kibo-ui.com/r/timeline.json
   pnpm dlx shadcn@latest add https://www.kibo-ui.com/r/combobox.json
   ```

4. **Install runtime deps**:
   ```bash
   pnpm --filter web add next-themes sonner @radix-ui/react-hover-card
   ```

5. **Update `packages/ui/src/index.ts`** — re-export primitives:
   ```typescript
   export * from './components/button'
   export * from './components/badge'
   export * from './components/card'
   // etc.
   ```
   Copy compiled shadcn output into `packages/ui/src/components/`.

6. **Update `apps/web/src/app/layout.tsx`**:
   - Add `ThemeProvider` from `next-themes`
   - Add `<Toaster />` from `sonner`
   - Add Inter + JetBrains Mono via `next/font/google`
   - Remove raw CSS variable `style={{...}}` from layout

7. **Update `apps/web/src/app/globals.css`**:
   - Replace hardcoded colour values with the Linear-inspired HSL token set above (dark + light)
   - Add `font-sans` and `font-mono` CSS variables pointing to Next.js font vars

8. **Delete all `style={{ ... }}` occurrences** using CSS vars in component files — replace with Tailwind classes.

9. **Verify**: `pnpm --filter web build` passes with zero TypeScript errors.

---

## Progress Tracking

**Overall Status:** Not Started — 0%

### Subtasks
| ID | Description | Status | Updated | Notes |
|----|-------------|--------|---------|-------|
| 17.1 | shadcn init (stone, cssVariables, rsc) | Not Started | 2026-04-21 | Interactive CLI |
| 17.2 | Add shadcn primitive components (14 listed) | Not Started | 2026-04-21 | |
| 17.3 | Add Kibo UI registries (5 components) | Not Started | 2026-04-21 | |
| 17.4 | Install next-themes + sonner | Not Started | 2026-04-21 | |
| 17.5 | Populate packages/ui re-exports | Not Started | 2026-04-21 | |
| 17.6 | Update layout.tsx (ThemeProvider, Toaster, fonts) | Not Started | 2026-04-21 | |
| 17.7 | Update globals.css (Linear HSL tokens, dark/light) | Not Started | 2026-04-21 | |
| 17.8 | Remove all raw style={{var(--...)}} occurrences | Not Started | 2026-04-21 | grep for style={{ |
| 17.9 | Build verification | Not Started | 2026-04-21 | pnpm --filter web build |

## Progress Log
### 2026-04-21
- Task created. packages/ui confirmed as placeholder. No shadcn, no Kibo UI, no next-themes installed.
