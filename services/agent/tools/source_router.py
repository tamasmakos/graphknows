"""EU source router — maps query keywords/topics to source adapter IDs."""
from __future__ import annotations


_DEFAULT_SOURCES = ["eprs_briefings", "ep_resolutions", "ep_debates"]

_KEYWORD_RULES: list[tuple[set[str], list[str]]] = [
    (
        {"regulation", "directive", "law", "legislation", "celex"},
        ["ep_resolutions", "eprs_briefings"],
    ),
    (
        {"debate", "speech", "mep", "parliament", "plenary"},
        ["ep_debates", "ep_resolutions"],
    ),
    (
        {"disinformation", "fimi", "interference", "manipulation", "propaganda",
         "doppelganger", "hybrid", "influence", "kremlin", "russia"},
        ["eprs_briefings", "ep_resolutions", "ep_debates"],
    ),
    (
        {"election", "vote", "campaign", "democratic"},
        ["eprs_briefings", "ep_resolutions"],
    ),
    (
        {"budget", "finance", "monetary"},
        ["eprs_briefings", "ep_resolutions"],
    ),
    (
        {"commission", "press release", "announcement"},
        ["eprs_briefings"],
    ),
]


def route_sources(query: str, max_sources: int = 3) -> list[str]:
    """Route a query to up to max_sources source IDs via keyword matching."""
    if not query or not query.strip():
        return _DEFAULT_SOURCES[:max_sources]

    q = query.lower()
    selected: list[str] = []

    for keywords, source_ids in _KEYWORD_RULES:
        if any(keyword in q for keyword in keywords):
            for source_id in source_ids:
                if source_id not in selected:
                    selected.append(source_id)

    if not selected:
        selected = list(_DEFAULT_SOURCES)

    return selected[:max_sources]
