"""Blocking keys: the coarse partition candidate generation compares within.

Two entities are only ever compared when they share a block, which is what
keeps identity off a full-table scan (FR-020). The key is deliberately blunt —
a short name prefix and the type the entity is most often read under — so that
a spelling drift ("Acme Corp" / "Acme Corporation") still collides while an
unrelated name never does.
"""

from __future__ import annotations

from collections.abc import Mapping

_PREFIX = 4


def dominant_type(type_histogram: Mapping[str, int]) -> str:
    """The label *type_histogram* was most often observed under, or ``""``.

    Ties break alphabetically: a block key that depended on insertion order
    would move an entity between blocks as observations arrive.
    """
    if not type_histogram:
        return ""
    return min(type_histogram, key=lambda label: (-type_histogram[label], label))


def block_key(name_norm: str, type_histogram: Mapping[str, int]) -> str:
    """The block *name_norm* belongs to under its observed types.

    *name_norm* is the resolver's normalised key, not a name: recomputing the
    normalisation here would be a second, silently diverging normaliser.
    """
    return f"{name_norm[:_PREFIX]}|{dominant_type(type_histogram)}"
