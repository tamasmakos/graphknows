"""Extraction — turn chunk text into entities and relations.

``entities/`` and ``relations/`` are the local decoder's stack; ``llm/`` is the
replacement decoder. ``build_decoder(settings)`` picks between them.
"""
