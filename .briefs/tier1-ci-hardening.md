# Brief: Tier 1 CI Hardening

## Working directory
/root/graphknows

## Goal
Implement Tier 1 CI quality hardening in one commit. No softening anything. Do not
touch files unrelated to this task.

---

## Change 1 — Expand ruff ruleset in ruff.toml

File: `/root/graphknows/ruff.toml`

Current `select` line:
```toml
select = ["E", "F", "I", "D", "C90", "N", "UP"]
```

Replace with:
```toml
select = [
    "E",    # pycodestyle errors
    "F",    # pyflakes
    "I",    # isort
    "D",    # pydocstyle
    "C90",  # mccabe complexity
    "N",    # naming
    "UP",   # pyupgrade
    "B",    # flake8-bugbear — likely bugs, mutable defaults, loop-var capture
    "BLE",  # blind except — catches `except Exception:` without variable binding
    "TRY",  # tryceratops — exception anti-patterns
    "SIM",  # flake8-simplify — boolean traps, redundant conditions
    "ERA",  # eradicate — commented-out dead code
    "RUF",  # Ruff-specific rules
]
```

Also add these ignore entries to the existing `ignore = [...]` list:
```toml
    "TRY003",  # long messages outside exception class — too strict for this codebase
    "TRY301",  # abstract raise to inner function — style choice, not a bug
    "BLE001",  # blind except — we have legitimate broad catches in MCP health checks
    "SIM108",  # ternary — readability preference
    "ERA001",  # commented code — false positives on intentional examples
```

The `**/graphgen/**` already has `"ALL"` in per-file-ignores — leave that alone.

After editing, run:
```bash
cd /root/graphknows && uv run ruff check graphknows/ --config ruff.toml 2>&1 | head -60
```

This will show violations. Do NOT auto-fix them yet — just record what's found.

---

## Change 2 — Fix the two clear silent swallows (not the MCP health probes)

These are genuine bugs — exceptions swallowed with no log, no re-raise:

### 2a. graphknows/container.py lines ~105 and ~110
```python
        except Exception:
            pass
```
These are in the `close()` / teardown method. Replace each bare `pass` with:
```python
        except Exception:
            logger.debug("Cleanup error ignored", exc_info=True)
```
Make sure `logger` is already imported at the top of that file (it likely is — check first).

### 2b. graphknows/retrieval/stores/memgraph.py line ~42
```python
                except Exception:
                    pass
```
Replace with:
```python
                except Exception:
                    logger.debug("Memgraph index creation ignored", exc_info=True)
```
Same — verify `logger` exists before adding the call.

DO NOT touch graphknows/mcp/stdio_server.py exception handlers — those are legitimate
connectivity probes that correctly swallow and set result flags.

DO NOT touch graphknows/ontology/rel_extractor.py — that's a deliberate schema-truncation
fallback, not a silent swallow.

---

## Change 3 — Add pip-audit CI job

File: `/root/graphknows/.github/workflows/ci.yml`

Add a new job AFTER the existing `arch-check` job (at the end of the file, before the final newline):

```yaml
  # ── 7. Dependency vulnerability scan — hard gate ──────────────────────────
  dep-audit:
    name: Dependency audit (pip-audit)
    runs-on: ubuntu-latest
    timeout-minutes: 5

    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          version: "latest"
          enable-cache: true

      - name: Install pip-audit
        run: uv tool install pip-audit

      - name: Audit dependencies for known CVEs
        run: pip-audit --requirement <(uv export --no-dev --format requirements-txt) --progress-spinner off
```

---

## Change 4 — Add gitleaks secret scanning CI job

Add another new job in the same ci.yml file:

```yaml
  # ── 8. Secret scanning — hard gate ────────────────────────────────────────
  secret-scan:
    name: Secret scanning (gitleaks)
    runs-on: ubuntu-latest
    timeout-minutes: 5

    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0    # full history for historical secret detection

      - name: Run gitleaks
        uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

---

## Change 5 — Add coverage floor to CI pytest command

File: `/root/graphknows/.github/workflows/ci.yml`

Find the `uv run pytest` command in the test job. It currently ends with:
```
            --cov-report=term-missing
```

Add after that line (still inside the run block):
```
            --cov-fail-under=55
```

So the full pytest invocation ends with:
```yaml
            --cov-report=xml:coverage.xml \
            --cov-report=term-missing \
            --cov-fail-under=55
```

---

## Acceptance criteria

1. `uv run ruff check graphknows/ --config ruff.toml` runs and reports any new violations
   (violations are EXPECTED — we want to see them, not fix them all now)
2. `container.py` and `memgraph.py` no longer have bare `except Exception: pass`
3. `ci.yml` has `dep-audit` and `secret-scan` jobs appended
4. `ci.yml` pytest command has `--cov-fail-under=55`
5. Run: `cd /root/graphknows && git diff --name-only` — should show ONLY:
   - `ruff.toml`
   - `graphknows/container.py`
   - `graphknows/retrieval/stores/memgraph.py`
   - `.github/workflows/ci.yml`
6. Commit with message:
   ```
   ci: tier 1 hardening — ruff B/TRY/BLE/SIM/ERA/RUF, pip-audit, gitleaks, coverage floor

   - ruff.toml: add B, BLE, TRY, SIM, ERA, RUF rule sets
   - container.py: replace silent except/pass with logger.debug
   - memgraph.py: replace silent except/pass with logger.debug
   - ci.yml: add dep-audit job (pip-audit, hard gate)
   - ci.yml: add secret-scan job (gitleaks, hard gate)
   - ci.yml: add --cov-fail-under=55 coverage floor
   ```
7. Push to main.
8. After committing, run ruff again and print ALL violations found so they can be reviewed:
   ```bash
   cd /root/graphknows && uv run ruff check graphknows/ --config ruff.toml 2>&1
   ```
   Print the full output — do not truncate.

## What NOT to do
- Do NOT auto-fix ruff violations beyond the two explicit silent-swallow fixes
- Do NOT modify any test files
- Do NOT touch graphgen/ production code
- Do NOT add `noqa` suppression lines to hide new violations
- Do NOT soften anything
