"""Shared helpers for writing entity nodes + edges + relation edges to the STM store.

_STMIngestHandler supplies the write pattern; the edge_writer callable is the
seam — callers supply a lambda that calls the appropriate edge method for the
edge type they need (chunk vs. turn).
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Any

# spaCy NER labels that are numeric/temporal noise as graph entities. Dates and
# times are captured separately as canonical TEMPORAL nodes; numbers/percents/
# money saturate the entity graph and pollute community/topic membership without
# ever serving as useful retrieval discriminators.
_NON_GRAPH_ENTITY_TYPES = frozenset(
    {"DATE", "TIME", "CARDINAL", "ORDINAL", "PERCENT", "MONEY", "QUANTITY"}
)

# GLiNER often leaves date/time/number spans UNTYPED (type=None), so the type
# allowlist above misses them and they leak in as ENTITY nodes (e.g. "23 July
# 2023", "10:31 am", "2023") — corpus-polluting noise that also corrupts
# community detection. This pattern catches them by SHAPE. It requires a digit
# alongside any month word, so real person/place names ("May", "June") survive.
_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
_TIME = r"\d{1,2}(?::\d{2})?\s*[ap]\.?m\.?"
_TEMPORAL_NAME_RE = re.compile(
    r"^("
    r"[\W\d]+"  # bare numbers / punctuation / "10:31" / "100" / "4"
    rf"|{_TIME}(?:\s+on\s+.{{0,40}})?"  # "10:31 am" / "1:56 pm on 8 May, 2023"
    rf"|\d{{1,4}}\s+{_MONTH}(?:,?\s+\d{{1,4}})?"  # "23 July 2023" / "8 May, 2023"
    # month-first, with optional day, an optional "- <month> <day>" range end,
    # and an optional trailing year: "July 2023" / "December 15, 2023" /
    # "November 1 - November 15, 2023" / "January 16 - February 15, 2024".
    # RUF001 suppressed: the en/em dashes are the literal characters real date
    # ranges are written with — replacing them with a hyphen breaks the match.
    rf"|{_MONTH}\s+\d{{1,4}}(?:\s*[-–—]\s*(?:{_MONTH}\s+)?\d{{1,4}})?(?:,?\s+\d{{2,4}})?"  # noqa: RUF001
    r"|\d{4}"  # bare year
    r")$",
    re.IGNORECASE,
)

# Relative-date shapes carry no digit, so they need their own branch rather than
# a loosened digit rule (which would kill the person names "May"/"June").
# Weekday names — unlike month names — are not person names, so bare weekdays
# are safe to drop.
_WEEKDAY = r"(?:mon|tues?|wed(?:nes)?|thu(?:rs)?|fri|satur|sun)(?:day)?s?"
_RELATIVE_DATE_RE = re.compile(
    rf"^(?:(?:last|next|this|every)\s+)?"
    rf"(?:{_WEEKDAY}|week(?:end)?|month|year|morning|afternoon|evening|night|"
    rf"yesterday|today|tomorrow)$",
    re.IGNORECASE,
)


def _is_temporal_or_numeric_name(name: str) -> bool:
    stripped = (name or "").strip()
    return bool(_TEMPORAL_NAME_RE.match(stripped)) or bool(_RELATIVE_DATE_RE.match(stripped))


# Closed-class function words (articles, prepositions, conjunctions, pronouns,
# auxiliaries, determiners, wh-words). GLiNER over-generates these on any text —
# a census of two live graphs found "the"/"and"/"of"/"be"/"these" minted as
# ENTITY nodes. They are never entities by grammar. Content words that merely
# happen to be generic ("project", "database") are deliberately NOT listed:
# retrieval already de-weights high-frequency entities (_discriminative_entities),
# so this set is only the closed class, where a drop can never lose a real entity.
_STOPWORD_NAMES = frozenset(
    """
    a an the this that these those such
    and or but nor so yet
    of to in on at by for from with without into onto over under about
    is are was were be been being am
    have has had do does did done
    would shall should can could might must ought
    it its he she him her they them we us you i me my your his their our
    as if then than there here not no yes
    what which who whom whose when where why how
    all any some each every both either neither few many much more most other
    """.split()
)

# Code, markup, URLs: GLiNER on technical text mints spans like
# "http://localhost:5000" or "Flask application: ```python from flask import".
_CODE_MARKUP_RE = re.compile(r"```|https?://|www\.|://|[{}<>=|]|::|\(\)|\[\]")
# A proper entity is a noun phrase, not a sentence or a heading — cap the words.
_MAX_ENTITY_WORDS = 8


def _is_stopword_name(name: str) -> bool:
    stripped = name.strip()
    # Acronym-shaped tokens (all-caps, 2-3 chars) collide with function words
    # once casefolded — "US"/"IT"/"UK"/"OR" vs "us"/"it"/... — so keep them.
    if stripped.isupper() and 2 <= len(stripped) <= 3:
        return False
    return stripped.casefold() in _STOPWORD_NAMES


def _is_code_or_clause(name: str) -> bool:
    stripped = name.strip()
    return bool(_CODE_MARKUP_RE.search(stripped)) or len(stripped.split()) > _MAX_ENTITY_WORDS


def _is_degenerate_name(name: str) -> bool:
    """Single characters, punctuation-only, and 1-2 char lowercase fragments.

    "db"/"id"/"io"/"to" are noise; an uppercase short form ("AI", "US") is a
    real acronym and kept, so the length-2 drop requires the token be lowercase.
    """
    stripped = name.strip()
    if len(stripped) <= 1:
        return True
    if not any(c.isalpha() for c in stripped):  # pure punctuation / digits / symbols
        return True
    return len(stripped) <= 2 and stripped == stripped.lower()


def is_graph_entity_name(name: str, ent_type: str = "") -> bool:
    """Whether ``name`` belongs in the ENTITY graph — the ONE junk predicate.

    False for empty, stopword, degenerate (single/short/punctuation), code/markup,
    over-long clause, and temporal/numeric names (by type or by shape). Exported
    so every minting path applies the same rule: ``graph_entities`` for the
    entity/MENTIONS path, and the relation gate in ``_write_extracted`` for
    relation endpoints, which previously bypassed all filtering and minted date
    ENTITY nodes (which then received a community_id and fragmented the topics).
    """
    stripped = (name or "").strip()
    if not stripped:
        return False
    if (ent_type or "").upper() in _NON_GRAPH_ENTITY_TYPES:
        return False
    if _is_stopword_name(stripped):
        return False
    if _is_degenerate_name(stripped):
        return False
    if _is_code_or_clause(stripped):
        return False
    return not _is_temporal_or_numeric_name(stripped)


def graph_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop numeric/temporal spans that should not become ENTITY nodes.

    Filters by spaCy/GLiNER type AND by name shape — the latter catches the
    common case where the extractor leaves a date/time/number untyped.
    """
    return [e for e in entities if is_graph_entity_name(e.get("name", ""), e.get("type") or "")]


async def write_entity_nodes_and_edges(
    stm: Any,
    session_id: str,
    entities: list[dict[str, Any]],
    edge_writer: Callable[[dict[str, Any]], Awaitable[Any]],
) -> int:
    """Write entity nodes and their source edge. Returns the entity count.

    Args:
        stm: ArcadeDBSTMStore (or ISessionDb) for graph writes.
        session_id: Session namespace.
        entities: List of entity dicts with "name", optional "type" and "score".
        edge_writer: Async callable(ent_dict) that writes the edge linking the
            entity to its source node (chunk or turn).  Callers supply this to
            avoid coupling the helper to a specific edge method.
    """
    entity_count = 0
    for ent in graph_entities(entities):
        await stm.write_entity_node(
            session_id=session_id,
            entity_name=ent["name"],
            entity_type=ent.get("type", "ENTITY"),
            confidence=ent.get("score", 1.0),
        )
        await edge_writer(ent)
        entity_count += 1
    return entity_count


async def write_relation_edges(
    stm: Any,
    session_id: str,
    relations: list[dict[str, Any]],
) -> None:
    """Write relation edges from extraction results.

    Args:
        stm: ArcadeDBSTMStore (or ISessionDb) for graph writes.
        session_id: Session namespace.
        relations: List of relation dicts with "head", "relation", "tail", optional "score".
    """
    for rel in relations:
        # Same junk gate the golden write path applies: a temporal endpoint
        # would mint an unfiltered ENTITY node.
        if not is_graph_entity_name(rel["head"], rel.get("head_type", "")) or (
            not is_graph_entity_name(rel["tail"], rel.get("tail_type", ""))
        ):
            continue
        await stm.write_relation_edge(
            session_id=session_id,
            subject_name=rel["head"],
            relation_name=rel["relation"],
            object_name=rel["tail"],
            confidence=rel.get("score", 0.5),
        )
