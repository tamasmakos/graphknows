"""frames — FrameNet extraction and the frame-definition embedding index.

Shared infrastructure, not a channel: :mod:`graphknows.symbolic.framenet.framenet` parses
text (a chunk at flush time, a question at query time) into frames, and
:mod:`graphknows.symbolic.framenet.index` cosine-matches an embedding against the ~1221
FrameNet frame definitions. Both paths use it — ``ingestion`` writes the FRAME
layer, ``channels.frames`` reads it back — which is why it sits below both
rather than inside the plugin folder. :mod:`graphknows.symbolic.framenet.fe` is the
frame-element plane: coreness, Requires/Excludes validity, CoreSet fillability
and the semantic-type veto.

No re-exports: consumers import from the specific module, so nothing pulls
spaCy/NLTK just by touching the package.
"""
