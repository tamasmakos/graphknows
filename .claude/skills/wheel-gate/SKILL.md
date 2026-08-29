---
name: wheel-gate
description: "Build the sdist and wheel, install the wheel in a clean venv, and prove the change ships. The only stage that MEASURES packaging instead of reasoning about it. Used by the process-issues workflow's Package stage; runnable by hand on any worktree."
disable-model-invocation: true
---

The unit suite passing proves the SOURCE TREE works. It does not prove the WHEEL works: tests import the source tree, users import the installed package. This gate builds the artifact and exercises it.

Everything runs INSIDE an ephemeral container mounted on the worktree: the host has neither `build` nor `twine`, and its virtualenv is incomplete. The caller gives you the shell-opening command; inside the container, /tmp is a real Linux tmpdir that dies with the container, so no artifact can ever land in the worktree.

The caller also names the import package (`<import_name>` below) and the reproduction test, if one exists.

## The seven steps

Run them one at a time and record what each actually printed.

1. `uvx --from build pyproject-build --outdir /tmp/dist` -- the image ships `uv`, not `build`. Both an sdist (.tar.gz) and a wheel (.whl) must appear.
2. `uvx twine check /tmp/dist/*` -- metadata is valid and the README renders.
3. `python -m zipfile -l /tmp/dist/*.whl`. Confirm EVERY file this change added is inside it -- new modules, and especially non-.py data files and py.typed. The backend is hatchling: it ships exactly what pyproject tells it to and nothing else. A missing file here is precisely the bug this gate exists to catch.
4. Fresh venv: `python -m venv /tmp/venv`, then `/tmp/venv/bin/pip install /tmp/dist/*.whl` -- ONLY the built wheel. Not the source tree, not editable.
5. `cd /tmp` (OUT of /app), then `/tmp/venv/bin/python -c "import <import_name>; print(<import_name>.__file__)"`. It must resolve inside /tmp/venv/lib/*/site-packages. If it prints a path under /app you are still importing the source tree -- cd out and rerun, the check is worthless otherwise.
6. With a reproduction test: copy it out of the source tree (`cp /app/<test file> /tmp/`, plus /app/tests/conftest.py if it needs fixtures) and run it from /tmp with `/tmp/venv/bin/python -m pytest`. Copying is the point: run it from /app and pytest puts the source tree on sys.path, so you would be testing the code you did not install. Passing from the source tree but failing here means the fix does not ship -- a blocker, record it under problems. Without one: exercise the changed code path directly against the installed package from /tmp, and say exactly what you ran.
7. Exit the container (`--rm` takes /tmp with it), then on the HOST confirm `git status --short` in the worktree shows no dist/, build/ or *.egg-info. hatchling should leave none; delete any that appeared and set worktree_clean accordingly.

## The rules

You are the only stage that MEASURES packaging rather than reasoning about it, so you also fix what you break. If a step fails because the change left the metadata wrong -- a third-party import missing from [project.dependencies], a runtime package sitting in a dev group, a new directory without __init__.py, a data file the backend was never told to ship -- edit pyproject.toml (or add the __init__.py), rebuild from step 1, and record each edit under "fixed".

Before editing pyproject.toml, consult the python-packaging skill beside this one: REFERENCE.md is the authority on dependency specifiers and what a build backend ships; SKILL.md covers PEP 508 grammar and extras vs dependency-groups -- which table a new dependency belongs in.

- Declare every dependency in the correct table with a lower bound.
- NEVER change the version, and never widen what ships just to make a step pass -- if you cannot fix it honestly, leave it and record it under "problems".
- NEVER upload anything: no `twine upload`, no TestPyPI, no PyPI, no git tag, no version edit. This gate builds and inspects only -- publishing belongs to a separate release workflow.
- If a step genuinely cannot run (no network for uvx, docker unavailable), set that field false and say exactly why in notes. Never report a check you did not actually run.
- Anything you find that is NOT caused by this change -- a file that was already missing from the wheel, metadata that was already wrong, a dependency nobody declared -- goes under "discoveries", not "problems". Those get filed as their own GitHub issues: packaging defects are invisible to the test suite, so an unfiled one stays unknown indefinitely.
