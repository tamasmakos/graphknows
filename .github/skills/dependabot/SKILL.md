---
name: dependabot
description: >-
  Comprehensive guide for configuring and managing GitHub Dependabot. Use this skill when
  users ask about creating or optimizing dependabot.yml files, managing Dependabot pull requests,
  configuring dependency update strategies, setting up grouped updates, monorepo patterns,
  multi-ecosystem groups, security update configuration, auto-triage rules, or any GitHub
  Advanced Security (GHAS) supply chain security topic related to Dependabot.
---

# Dependabot Configuration & Management

## Overview

Dependabot is GitHub's built-in dependency management tool with three core capabilities:

1. **Dependabot Alerts** — Notify when dependencies have known vulnerabilities (CVEs)
2. **Dependabot Security Updates** — Auto-create PRs to fix vulnerable dependencies
3. **Dependabot Version Updates** — Auto-create PRs to keep dependencies current

All configuration lives in a **single file**: `.github/dependabot.yml` on the default branch.

## Configuration Workflow

### Step 1: Detect All Ecosystems

Scan the repository for dependency manifests. Look for:

| Ecosystem | YAML Value | Manifest Files |
|---|---|---|
| npm/pnpm/yarn | `npm` | `package.json`, `package-lock.json`, `pnpm-lock.yaml`, `yarn.lock` |
| pip/pipenv/poetry/uv | `pip` | `requirements.txt`, `Pipfile`, `pyproject.toml`, `setup.py` |
| Docker | `docker` | `Dockerfile` |
| Docker Compose | `docker-compose` | `docker-compose.yml` |
| GitHub Actions | `github-actions` | `.github/workflows/*.yml` |
| Go modules | `gomod` | `go.mod` |
| Bundler (Ruby) | `bundler` | `Gemfile` |
| Cargo (Rust) | `cargo` | `Cargo.toml` |
| Composer (PHP) | `composer` | `composer.json` |
| NuGet (.NET) | `nuget` | `*.csproj`, `packages.config` |
| Maven (Java) | `maven` | `pom.xml` |
| Gradle (Java) | `gradle` | `build.gradle` |
| Terraform | `terraform` | `*.tf` |
| Helm | `helm` | `Chart.yaml` |

### Step 2: Map Directory Locations

For each ecosystem, identify where manifests live. Use `directories` (plural) with glob patterns for monorepos:

```yaml
directories:
  - "/"           # root
  - "/apps/*"     # all app subdirs
  - "/packages/*" # all package subdirs
  - "**/*"        # recursive (all subdirs)
```

`directory` (singular) does NOT support globs — use `directories` (plural) for wildcards.

### Step 3: Configure Each Ecosystem Entry

Minimum required config:

```yaml
- package-ecosystem: "npm"
  directory: "/"
  schedule:
    interval: "weekly"
```

## Monorepo Strategies

```yaml
- package-ecosystem: "npm"
  directories:
    - "/"
    - "/apps/*"
    - "/packages/*"
    - "/services/*"
  schedule:
    interval: "weekly"
```

Use `group-by: dependency-name` to create a single PR when the same dependency updates across multiple directories.

## Dependency Grouping

```yaml
groups:
  dev-dependencies:
    dependency-type: "development"
    update-types: ["minor", "patch"]
  production-dependencies:
    dependency-type: "production"
    update-types: ["minor", "patch"]
  security-patches:
    applies-to: security-updates
    patterns: ["*"]
    update-types: ["patch", "minor"]
```

## PR Customization

```yaml
labels:
  - "dependencies"
commit-message:
  prefix: "deps"
  prefix-development: "deps-dev"
  include: "scope"
assignees: ["security-team-lead"]
```

## Schedule Options

```yaml
schedule:
  interval: "weekly"   # daily, weekly, monthly, quarterly, semiannually, yearly, cron
  day: "monday"
  time: "09:00"
  timezone: "America/New_York"
```

## Cooldown Periods

```yaml
cooldown:
  default-days: 5
  semver-major-days: 30
  semver-minor-days: 7
  semver-patch-days: 3
```

## Ignore and Allow Rules

```yaml
ignore:
  - dependency-name: "lodash"
  - dependency-name: "@types/node"
    update-types: ["version-update:semver-patch"]

allow:
  - dependency-type: "production"
```

## Versioning Strategy

| Value | Behavior |
|---|---|
| `auto` | Default — increase for apps, widen for libraries |
| `increase` | Always increase minimum version |
| `lockfile-only` | Only update lockfiles, ignore manifests |
| `widen` | Widen range to include both old and new versions |

## PR Limit

```yaml
open-pull-requests-limit: 10  # default is 5; set to 0 to disable version updates
```

## PR Comment Commands

| Command | Effect |
|---|---|
| `@dependabot rebase` | Rebase the PR |
| `@dependabot recreate` | Recreate the PR from scratch |
| `@dependabot ignore this dependency` | Close and never update this dependency |
| `@dependabot ignore this major version` | Ignore this major version |
| `@dependabot ignore this minor version` | Ignore this minor version |
| `@dependabot ignore this patch version` | Ignore this patch version |
