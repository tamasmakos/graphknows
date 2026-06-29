"""Semantic-memory distillation phase for the LTM promotion pipeline.

Produces canonical SEMANTIC nodes — deduplicated, entity-resolved, reasoning-
oriented fact/preference records derived from raw episodic STM chunks.  These
are distinct from CHUNK/TURN (raw evidence) and TOPIC (community summaries).

Distillation strategy
---------------------
Each ``(subject_entity, predicate, object_value)`` triplet extracted from
relation_edges *or* synthesised from chunk text is mapped to a SEMANTIC node.
Deduplication is by ``sem_key = sha1(subject.lower():predicate.upper():object.lower())``.
Entity aliases from EntityResolution are resolved to canonical names before
keying so that e.g. "eu" and "EU" share the same node.

In ``llm_free`` mode the triplets come exclusively from the relation_edges
already present in stm_data (extracted without LLM during STM ingest).

In ``llm_assisted`` mode the distiller also calls the single configured LLM
(``deepseek/deepseek-v4-flash`` via OpenRouter) to synthesise additional
fact/preference statements from chunk text, then merges them with the
structural triplets.

Both modes produce the same SEMANTIC node schema and go through the same
deduplication/resolution path.

SEMANTIC node schema
--------------------
``SEMANTIC`` vertex in ArcadeDB / conceptual dict structure::

    {
        "id":          str   # sha1-derived stable key
        "subject":     str   # canonical entity name
        "predicate":   str   # relation label / slot name (UPPER_SNAKE)
        "object":      str   # value, may be another entity name or literal
        "confidence":  float # aggregated across source edges
        "memory_class": str  # "fact" or "preference"
        "source_session_ids": list[str]
        "created_at":  str   # ISO-8601
        "updated_at":  str   # ISO-8601
    }

The ArcadeDB store (ltm_store) provides ``upsert_semantic_facts`` to write
these.  flush.py calls this module as a post-merge, pre-PageRank phase so the
SEMANTIC nodes exist when GDS centrality runs.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import UTC, datetime
from typing import Any

log = logging.getLogger("graphknows.memory.ltm.semantic")

# Predicates considered "preference" rather than "fact".
_PREFERENCE_PREDICATES: frozenset[str] = frozenset(
    {
        "PREFERS",
        "LIKES",
        "DISLIKES",
        "WANTS",
        "AVOIDS",
        "INTERESTED_IN",
        "HOBBY",
        "GOAL",
        "VALUES",
    }
)

# Maximum number of chunks sent to the LLM per distillation call.
_MAX_LLM_CHUNKS = 8


def _sem_key(subject: str, predicate: str, obj: str) -> str:
    """Stable dedup key for a (subject, predicate, object) triplet."""
    raw = f"{subject.strip().lower()}:{predicate.strip().upper()}:{obj.strip().lower()}"
    return hashlib.sha1(raw.encode(), usedforsecurity=False).hexdigest()


def _resolve_name(name: str, alias_to_canon: dict[str, str]) -> str:
    """Map an entity name to its canonical form using alias table."""
    return alias_to_canon.get(name.strip().casefold(), name.strip())


def _memory_class(predicate: str) -> str:
    """Classify a predicate as 'fact' or 'preference'."""
    return "preference" if predicate.strip().upper() in _PREFERENCE_PREDICATES else "fact"


def _triplets_from_relation_edges(
    relation_edges: list[dict[str, Any]],
    alias_to_canon: dict[str, str],
) -> list[dict[str, Any]]:
    """Extract (subject, predicate, object) triplets from existing relation_edges.

    These come from GLiNER / spaCy extraction during STM ingest — no LLM needed.
    """
    seen: dict[str, dict[str, Any]] = {}
    for edge in relation_edges:
        src = str(edge.get("source") or "").strip()
        rel = str(edge.get("relation") or "").strip().upper()
        tgt = str(edge.get("target") or "").strip()
        if not (src and rel and tgt):
            continue
        src = _resolve_name(src, alias_to_canon)
        tgt = _resolve_name(tgt, alias_to_canon)
        key = _sem_key(src, rel, tgt)
        if key in seen:
            # Accumulate confidence (average later).
            seen[key]["_conf_sum"] = seen[key].get("_conf_sum", seen[key]["confidence"]) + float(
                edge.get("confidence", edge.get("weight", 1.0)) or 1.0
            )
            seen[key]["_conf_count"] = seen[key].get("_conf_count", 1) + 1
        else:
            seen[key] = {
                "id": key,
                "subject": src,
                "predicate": rel,
                "object": tgt,
                "confidence": float(edge.get("confidence", edge.get("weight", 1.0)) or 1.0),
                "memory_class": _memory_class(rel),
                "_conf_sum": float(edge.get("confidence", edge.get("weight", 1.0)) or 1.0),
                "_conf_count": 1,
            }
    # Finalise averaged confidence.
    for item in seen.values():
        count = item.pop("_conf_count", 1)
        total = item.pop("_conf_sum", item["confidence"])
        item["confidence"] = round(total / max(count, 1), 4)
    return list(seen.values())


def _build_alias_map(entity_aliases: dict[str, list[str]]) -> dict[str, str]:
    """Build casefold(alias) → canonical_name lookup from PromotionAudit.entity_aliases."""
    alias_to_canon: dict[str, str] = {}
    for canon, aliases in entity_aliases.items():
        for alias in aliases:
            alias_to_canon[alias.casefold()] = canon
    return alias_to_canon


# ---------------------------------------------------------------------------
# LLM distillation (llm_assisted only)
# ---------------------------------------------------------------------------


async def _llm_distill_chunks(
    chunks: list[dict[str, Any]],
    openrouter_key: str,
    llm_model: str,
    alias_to_canon: dict[str, str],
) -> list[dict[str, Any]]:
    """Call the configured LLM to extract fact/preference triplets from chunk text.

    Uses a minimal prompt — no DSPy overhead needed here. Returns triplets in
    the same format as _triplets_from_relation_edges.  Falls back to empty list
    on any error so that the promotion pipeline remains non-fatal.
    """
    try:
        import httpx
    except ImportError:
        log.warning("httpx not available; skipping LLM semantic distillation")
        return []

    from graphknows.config.llm import normalize_openrouter_model

    model = normalize_openrouter_model(llm_model)
    # Strip openrouter/ prefix — OpenRouter API doesn't want it in the model field.
    if model.startswith("openrouter/"):
        model = model[len("openrouter/"):]

    # Collect up to _MAX_LLM_CHUNKS longest chunks.
    sorted_chunks = sorted(chunks, key=lambda c: len(c.get("text", "")), reverse=True)
    selected = sorted_chunks[:_MAX_LLM_CHUNKS]
    combined_text = "\n\n---\n\n".join(c.get("text", "") for c in selected if c.get("text"))
    if not combined_text.strip():
        return []

    system_prompt = (
        "Extract factual statements and preferences from the text as JSON lines. "
        "Each line: {\"subject\": str, \"predicate\": str, \"object\": str, \"type\": \"fact\"|\"preference\"}. "
        "Use UPPER_SNAKE_CASE predicates. Output only valid JSON lines, no commentary."
    )
    user_prompt = f"Text:\n{combined_text[:6000]}\n\nExtract triplets:"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {openrouter_key}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
        content = data["choices"][0]["message"]["content"]
    except Exception as exc:
        log.warning("LLM semantic distillation API call failed: %s", exc)
        return []

    import json

    triplets: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for line in content.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        subj = str(item.get("subject") or "").strip()
        pred = str(item.get("predicate") or "").strip().upper()
        obj = str(item.get("object") or "").strip()
        mem_type = str(item.get("type") or "fact").strip()
        if not (subj and pred and obj):
            continue
        subj = _resolve_name(subj, alias_to_canon)
        obj_resolved = _resolve_name(obj, alias_to_canon)
        key = _sem_key(subj, pred, obj_resolved)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        triplets.append(
            {
                "id": key,
                "subject": subj,
                "predicate": pred,
                "object": obj_resolved,
                "confidence": 0.8,  # LLM-extracted; moderate default confidence
                "memory_class": "preference" if mem_type == "preference" else "fact",
            }
        )
    log.debug("LLM semantic distillation: %d triplets extracted from %d chunks", len(triplets), len(selected))
    return triplets


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class SemanticDistiller:
    """Distil raw STM data into canonical SEMANTIC fact/preference nodes.

    This is a promotion pipeline phase — call ``distil`` after entity resolution
    and LTM graph write, before PageRank/GDS.

    Args:
        openrouter_key: API key for LLM-assisted mode. Empty string = llm_free.
        llm_model:      Model identifier (e.g. ``deepseek/deepseek-v4-flash``).
                        Only used when openrouter_key is non-empty.
    """

    def __init__(self, openrouter_key: str = "", llm_model: str = "") -> None:
        self._openrouter_key = (openrouter_key or "").strip()
        self._llm_model = llm_model or ""

    @property
    def llm_enabled(self) -> bool:
        """True when operating in llm_assisted mode."""
        return bool(self._openrouter_key)

    async def distil(
        self,
        stm_data: dict[str, Any],
        session_id: str,
        entity_aliases: dict[str, list[str]] | None = None,
    ) -> list[dict[str, Any]]:
        """Return a deduplicated list of SEMANTIC fact/preference dicts.

        Args:
            stm_data:       The STM subgraph dict (output of read_session_subgraph).
            session_id:     Current session identifier — stored on each SEMANTIC node.
            entity_aliases: Optional alias map from PromotionAudit.entity_aliases.

        Returns:
            List of SEMANTIC node dicts ready for ``ltm_store.upsert_semantic_facts``.
        """
        alias_to_canon = _build_alias_map(entity_aliases or {})
        now = datetime.now(UTC).isoformat()

        # Phase 1: structural triplets from existing relation_edges (both modes).
        structural = _triplets_from_relation_edges(
            stm_data.get("relation_edges", []), alias_to_canon
        )
        log.debug(
            "SemanticDistiller.distil session=%s: %d structural triplets",
            session_id,
            len(structural),
        )

        # Phase 2: LLM-synthesised triplets (llm_assisted only).
        llm_triplets: list[dict[str, Any]] = []
        if self.llm_enabled:
            chunks = stm_data.get("chunks", [])
            if chunks:
                try:
                    llm_triplets = await _llm_distill_chunks(
                        chunks, self._openrouter_key, self._llm_model, alias_to_canon
                    )
                except Exception as exc:
                    log.warning(
                        "SemanticDistiller.distil session=%s: LLM distillation failed, "
                        "falling back to structural only: %s",
                        session_id,
                        exc,
                    )
                    llm_triplets = []
                log.debug(
                    "SemanticDistiller.distil session=%s: %d LLM triplets",
                    session_id,
                    len(llm_triplets),
                )

        # Merge: structural facts take precedence (higher/known confidence).
        merged: dict[str, dict[str, Any]] = {item["id"]: item for item in structural}
        for item in llm_triplets:
            key = item["id"]
            if key not in merged:
                merged[key] = item
            else:
                # LLM corroborates structural — boost confidence slightly.
                existing_conf = merged[key]["confidence"]
                merged[key]["confidence"] = round(min(1.0, existing_conf * 1.05), 4)

        # Annotate with session provenance and timestamps.
        result: list[dict[str, Any]] = []
        for item in merged.values():
            result.append(
                {
                    **item,
                    "source_session_ids": [session_id],
                    "created_at": now,
                    "updated_at": now,
                }
            )

        log.info(
            "SemanticDistiller.distil session=%s: %d SEMANTIC facts (structural=%d llm=%d mode=%s)",
            session_id,
            len(result),
            len(structural),
            len(llm_triplets),
            "llm_assisted" if self.llm_enabled else "llm_free",
        )
        return result
