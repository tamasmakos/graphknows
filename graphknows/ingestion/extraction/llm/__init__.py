"""The LLM decoder — one structured call per chunk, over a closed per-chunk schema.

``schema`` holds the request the ingest handler builds and the strict response
models the provider is held to; the anchoring, vocabulary and confidence gates
that turn a response into an ``ExtractionResult`` live beside it.
"""
