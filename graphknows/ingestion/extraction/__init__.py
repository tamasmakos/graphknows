"""Extraction — turn chunk text into entities and relations.

``entities/`` is shared across modes; ``relations/`` forks the mode
(``llm_free`` deterministic cleaning vs ``llm_assisted`` DSPy/ontology) behind
``relations.build_relation_extractor``.
"""
