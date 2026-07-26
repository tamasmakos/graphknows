"""llm_assisted relation extraction — DSPy constrained by an injectable ontology.

Importing anything here pulls ``dspy``/``rdflib`` (the ``assisted`` extra), so
callers import lazily from within the mode factory.
"""
