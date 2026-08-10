"""frames — FrameNet extraction and the frame-definition embedding index.

Shared infrastructure, not a channel: :mod:`graphknows.frames.framenet` parses
text (a chunk at flush time, a question at query time) into frames, and
:mod:`graphknows.frames.index` cosine-matches an embedding against the ~1221
FrameNet frame definitions. Both paths use it — ``ingestion`` writes the FRAME
layer, ``channels.frames`` reads it back — which is why it sits below both
rather than inside the plugin folder.

No re-exports: consumers import from the specific module, so nothing pulls
spaCy/NLTK just by touching the package.
"""
