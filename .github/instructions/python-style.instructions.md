---
description: "Use when writing, reviewing, or refactoring Python code. Enforces clean Pythonic style, Google Python Style Guide conventions, and the Zen of Python. Covers naming, imports, type annotations, docstrings, exceptions, and idioms."
applyTo: "**/*.py"
---

# Python Style: Pythonic Zen + Google Style Guide

Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) and the Zen of Python (`import this`). Write code that is **readable, explicit, and simple**. Optimize for the reader, not the writer.

---

## Imports

- One import per line (except `from typing import A, B` and `from collections.abc import X, Y`).
- Order: `__future__` → stdlib → third-party → local. Separate groups with a blank line.
- Use `import x` for packages/modules; `from x import y` for specific symbols.
- Always use full package paths — no bare relative imports.
- Never use wildcard imports (`from x import *`).

```python
# Yes
from collections.abc import Mapping, Sequence
import os
import sys

from mypackage.utils import helper
```

---

## Naming

| Entity | Style | Example |
|---|---|---|
| Packages / Modules | `lower_with_under` | `graph_utils` |
| Classes / Exceptions | `CapWords` | `GraphNode`, `ParseError` |
| Functions / Methods / Variables | `lower_with_under` | `fetch_rows`, `user_name` |
| Constants | `CAPS_WITH_UNDER` | `MAX_RETRIES` |
| Protected / internal | leading `_` | `_internal_helper` |

- Use descriptive names. **No abbreviations** unless universally known (`url`, `html`, `id`).
- Avoid type-encoding in names (`id_to_name_dict` → `id_to_name`).
- Never use `__double_leading_and_trailing_underscore__` names.
- Single-char names only for counters (`i`, `j`, `k`), exception handles (`e`), file handles (`f`), or unconstrained private type vars (`_T`).

---

## Type Annotations

Annotate all public APIs. Use modern syntax (Python 3.10+).

```python
# Yes
def fetch(keys: Sequence[str], limit: int | None = None) -> Mapping[str, list[str]]:
    ...

# No
def fetch(keys, limit=None):
    ...
```

- Use `X | None` instead of `Optional[X]`.
- Prefer abstract types in signatures: `Sequence` over `list`, `Mapping` over `dict`.
- Use built-in generics: `list[int]`, `dict[str, int]`, `tuple[int, ...]` — not `typing.List`, `typing.Dict`.
- Never use implicit `Optional` (e.g. `a: str = None` is wrong — write `a: str | None = None`).
- Annotate `self`/`cls` only when needed (e.g. `typing.Self` for chained constructors).

---

## Docstrings

Use `"""triple double-quotes"""`. Required for all public modules, classes, and functions with non-trivial logic or public APIs.

```python
def connect(self, minimum: int) -> int:
    """Connects to the next available port.

    Args:
        minimum: A port value greater or equal to 1024.

    Returns:
        The new minimum port.

    Raises:
        ConnectionError: If no available port is found.
    """
```

- Summary line: ≤80 chars, ends with `.`, `?`, or `!`.
- Use `Args:`, `Returns:` (or `Yields:`), `Raises:` sections for non-trivial functions.
- `@override` methods without behavioral changes do not need a docstring.
- Never describe *what* the code does line-by-line — explain *why* for non-obvious logic.

---

## Exceptions

```python
# Yes — specific, with context
raise ValueError(f'Port must be >= 1024, got {minimum!r}.')

# No — bare except swallows everything
try:
    connect()
except:
    pass

# No — catches too broadly
except Exception:
    pass
```

- Raise built-in exceptions for programming mistakes (`ValueError`, `TypeError`, `RuntimeError`).
- Never use bare `except:` or `except Exception:` unless re-raising or at an isolation boundary.
- Custom exception classes must end in `Error` and inherit from an existing exception.
- Keep `try` blocks minimal — only wrap the line(s) that can raise.
- Use `finally` for cleanup.

---

## Pythonic Idioms

**Boolean checks:**
```python
# Yes
if users:           ...
if not items:       ...
if x is None:       ...
if not x and x is not None:  ...  # distinguish False from None

# No
if len(users) == 0: ...
if users == []:     ...
if x == None:       ...
```

**Iteration:**
```python
# Yes
for key in mapping:         ...
for k, v in mapping.items(): ...

# No
for key in mapping.keys():  ...
```

**Comprehensions** — use for simple cases; never more than one `for` and one optional `if`:
```python
# Yes
results = [f(x) for x in iterable if predicate(x)]

# No — too nested
results = [(x, y) for x in range(10) for y in range(5) if x * y > 10]
```

**Strings:** use f-strings or `%`/`.format()` — never `+` in loops:
```python
# Yes
parts = [f'<td>{name}</td>' for name in names]
output = ''.join(parts)

# No
output = ''
for name in names:
    output += f'<td>{name}</td>'
```

**Resources:** always use `with` statements:
```python
with open('file.txt') as f:
    data = f.read()
```

---

## Functions & Classes

- Functions should be ≤40 lines. If longer, consider splitting.
- No mutable default arguments:

```python
# Yes
def process(items: list[str] | None = None) -> None:
    if items is None:
        items = []

# No
def process(items: list[str] = []) -> None: ...
```

- Avoid mutable global state. Module-level **constants** are fine (`MAX_SIZE = 100`).
- Use `@property` only for trivial, cheap attribute access. Avoid side effects in properties.
- Avoid `staticmethod`. Use module-level functions instead.
- Use `classmethod` only for named constructors or class-scoped operations.
- Avoid Python power features: metaclasses, `__del__`, dynamic code generation, monkey-patching.

---

## Formatting

- **Indent:** 4 spaces. Never tabs.
- **Line length:** 80 chars. Use implicit continuation inside `()`, `[]`, `{}` — never backslash `\`.
- **Blank lines:** 2 between top-level definitions; 1 between methods.
- **No semicolons.** One statement per line.
- No spaces inside brackets: `spam(ham[1], {'eggs': 2})`.
- Spaces around `=` only when a type annotation is also present.
- Imports at top; no top-level executable code outside `if __name__ == '__main__':`.

---

## Guiding Principle

> **BE CONSISTENT.** Match the style of the surrounding code. Readability counts.
