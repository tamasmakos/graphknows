---
description: "Use when writing, reviewing, or refactoring React and TypeScript code. Covers component structure, naming conventions, folder layout, types vs interfaces, hooks ordering, early returns, comments, and GraphQL query patterns."
applyTo: "**/*.{ts,tsx}"
---

# React + TypeScript Style Guide

Follow [react-typescript-style-guide.com](https://react-typescript-style-guide.com/). Prioritize **clarity, predictability, and minimal cognitive overhead**. Optimize for the reader.

---

## Core Philosophy

- **Clarity over flexibility** — uniform structure beats clever abstraction.
- **Encapsulation** — keep a feature's components, hooks, and utils in the same folder.
- **No unnecessary abstraction** — avoid wrapper functions, premature optimizations, and excessive prop drilling.
- **Early returns** — return early to reduce nesting and improve readability.
- **Separation of concerns** — business logic in hooks/utils, not in JSX.

---

## Folder Structure

```
src/
├── common/           # Shared UI components, hooks, utilities
│   ├── components/
│   ├── hooks/
├── config/           # External service integrations ONLY (Apollo, Analytics, etc.)
├── constants/        # App-wide constants/utils used in 2+ features
├── pages/
│   └── profile/      # Feature folder — self-contained
│       ├── common/   # Feature-level shared components
│       ├── hooks/    # Feature-specific queries/mutations
│       ├── Profile.tsx
│       ├── profileUtils.ts
│       ├── types.ts
│       └── index.ts
```

**Rules:**
- Feature-specific hooks, utils, and constants stay inside `pages/featureName/`.
- Move to `common/` or `constants/` only when used by 2+ features.
- Add `index.ts` barrel files to simplify imports — but avoid app-wide barrels (circular deps, tree-shaking issues, SSR problems).
- Prefer named imports over `import * as X`.

---

## Components

**Always use functional components:**
```tsx
export const ProfileHero = ({ onClick, title }: ProfileHeroProps) => (
  <div onClick={onClick}>{title}</div>
)
```

**Component internal order:**
1. Hooks (`useState`, `useNavigate`, custom hooks)
2. Local variables / constants (not functions)
3. `useEffect` hooks
4. Functions (event handlers, derived helpers)
5. Early return for loading/error states
6. Final `return` JSX

```tsx
export const Profile = () => {
  // 1. Hooks
  const { accountHandle } = useParams()
  const { hasError, isLoading, profileData } = useGetProfileQuery(accountHandle)

  // 3. Effects
  useEffect(() => {
    // analytics
  }, [])

  // 4. Functions
  const getProfileAvatar = () => {}

  // 5. Early returns (blank line before each)
  if (isLoading) return <ProfileLoading />

  if (hasError) return <ProfileEmpty />

  // 6. Final return (blank line before)
  return (
    <section>
      <ProfileHero />
    </section>
  )
}
```

**Rules:**
- Keep components under ~150 lines. Split if exceeding.
- Always add a blank line before `return`.
- Early returns for guard clauses — inline, no `{}` block for one-liners.
- No deep JSX nesting — extract sub-components instead.
- Loading states (`ProfileLoading`) must mirror the real component structure with skeleton placeholders.

---

## Naming

| Entity | Convention | Example |
|---|---|---|
| Components, types, interfaces | `PascalCase` | `ProfileHero`, `UserProps` |
| Functions, hooks, variables | `camelCase` | `getProfileName`, `useFlag` |
| Feature-scoped files | `camelCase` | `profileUtils.ts`, `profileConstants.ts` |
| Component files/folders | Match component name | `ProfileHero/ProfileHero.tsx` |

---

## Types & Interfaces

**Use `interface` for component props only:**
```tsx
interface ProfileHeroProps {
  onClick: () => void
  title: string
}

// Extend with interface
interface ProfileAddressProps extends GenericAddress {
  onClick: VoidFunction
}
```

**Use `type` for everything else** (hooks, utilities, GraphQL return types, unions, intersections):
```ts
type UseGetProfileQueryReturn = {
  hasError: ApolloError
  isLoading: boolean
  profileData: Profile
}

// Pick / Omit
type UserInfo = Pick<User, 'id' | 'email'>
type PublicUser = Omit<User, 'password'>

// Intersection
type ProfileWithBase = Profile & Base
```

**Use `Extract<>` when narrowing GraphQL union types:**
```ts
profileData: Extract<GetProfileQueryInProfileQuery['node'], { __typename?: 'Profile' }>
```

---

## Functions & Utilities

- **Flat logic** — early returns over nested `if/else`.
- **No blank line** before an early return at the start of a function.
- **Add a blank line** before an `if` that appears in the middle of a function.
- **No extra blank line** before the final `return`.

```ts
// ✅
const getUserDetails = (user: User | null) => {
  if (!user) return null

  return { id: user.id, name: user.name }
}

// ✅ — blank line before mid-function if
const getProfileName = (profileData: ProfileData) => {
  const { firstName, lastName } = profileData ?? {}

  if (!firstName || !lastName) return 'Guest'

  return `${firstName} ${lastName}`
}
```

Export multiple utilities together at the bottom:
```ts
export { getProfileAvatar, getProfileName }
```

---

## Comments & Documentation

- **Self-documenting names first** — rename before commenting.
- **Only explain "why"**, never "what" the code obviously does.

```ts
// ✅ — explains why the workaround exists
// Safari requires a slight delay for smooth scrolling
setTimeout(() => window.scrollTo(0, 0), 10)

// ❌ — describes what the code does
// Scrolls to the top of the page
window.scrollTo(0, 0)
```

**`/** @todo */` for future work** (JSDoc-compatible):
```ts
/** @todo TMNT-123 Remove once API supports real-time updates */
const getUserPreferences = async (userId: string) => { ... }
```

**`/** @see */` for external references:**
```ts
/**
 * Safari smooth scroll workaround.
 * @see https://stackoverflow.com/q/xxxx
 */
```

Extract complex `useEffect` logic into named functions instead of inline comments.

---

## GraphQL Queries & Mutations

- Place queries/mutations in `hooks/` inside their feature folder.
- **Feature-scoped** operations include `In{FeatureName}` in the operation name.
- **Sitewide** operations (used in 2+ features) go in `src/hooks/` and omit the feature suffix.
- Sort GraphQL fields alphabetically — `id` always first.
- Alias `error`/`loading` to `hasError`/`isLoading` for readability.

```ts
// Feature-scoped query
// src/pages/profile/hooks/useGetProfileQuery.ts

const profileQuery = gql(`
  query GetProfileQueryInProfile($id: ID!) {
    node(id: $id) {
      ... on Profile {
        id
        accountHandle
        displayName
        image
      }
    }
  }
`)

export const useGetProfileQuery = (id: string): UseGetProfileQueryReturn => {
  const { data, error: hasError, loading: isLoading } = useQuery(profileQuery, {
    variables: { id },
  })

  return { hasError, isLoading, profileData: data?.node }
}
```

- Hook return types use `PascalCase`: `UseGetProfileQueryReturn`.
- Mutation files always include `Mutation` in both filename and operation name.

---

## Feature Flags

- Define all flags in `src/config/feature-flags/featureFlags.ts`.
- Access via `useFlag(flagName)` hook — never read from config directly in components.
- Feature flags are **short-lived** — remove once the feature is stable.

```tsx
const isProfileHeroV2Enabled = useFlag('profileHeroV2')

return (
  <section>
    {isProfileHeroV2Enabled ? <ProfileHero /> : <ProfileHeroOld />}
  </section>
)
```

---

## Guiding Principle

> **BE CONSISTENT.** Predictability reduces cognitive load. Follow the structure of the surrounding code.
