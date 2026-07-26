"""Command-line entry points.

Composition roots: a CLI wires the facade (:mod:`graphknows.memory`) to argv and
stdout, so it sits *above* every capability package. That is why this lives here
rather than as ``graphknows/topics/__main__.py`` — a ``__main__`` inside a low
layer makes that layer depend on the facade above it, which the architecture
contract forbids and which ``lint-imports`` rejects.
"""
