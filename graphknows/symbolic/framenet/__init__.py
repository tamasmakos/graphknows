"""framenet — FrameNet as pack vocabulary, and nothing else.

:mod:`graphknows.symbolic.framenet.emitter` turns the shipped corpus into
concepts and role predicates a pack can declare (FR-023). The frame plane the
old core wrote — the frame-element mirror, frame-specific vertex and edge
types, frame parsing and the frame-definition matching index — is gone.

No re-exports: consumers import from the specific module, so nothing pulls the
corpus just by touching the package.
"""
