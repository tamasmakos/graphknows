"""The channels every recall runs — fixed, not configured.

There is one set and it is the same for every namespace: the core collectors.
No opt-in flag chooses them, because a flag on a retrieval leg is an A/B
knob wearing a setting's clothes — it forks the fusion by environment, so no
two runs are comparable and no measurement survives the next deploy.

The four baseline collectors (entity, vector, bm25, temporal) are methods on
the retriever spine and are always run; the ontology collector is a
:class:`~graphknows.channels.base.Channel` because it reads structure the
symbolic layer writes.

Frames are not core: FrameNet is a pack's concept+predicate emitter, so its
collector belongs to the pack that emits it. A pack-supplied channel arrives
with the dialogue pack (FR-022) — ``DomainPack`` declares no ``channel()``
yet, so nothing here reads packs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graphknows.channels.base import Channel


def core_channels() -> list[Channel]:
    """Instantiate the core collectors — the same list on every call."""
    from graphknows.channels.ontology import OntologyChannel

    return [OntologyChannel()]


__all__ = ["core_channels"]
