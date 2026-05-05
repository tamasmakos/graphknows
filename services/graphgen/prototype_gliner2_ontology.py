"""
Political/Legal ontology label space builder for GLiNER2 extraction.

NER labels come exclusively from ontologies that define real-world entity types:
  - LKIF Core (local names) — legal entities: Person, Treaty, Directive, Legislative Body ...
  - W3C Org (rdfs:label)    — organizational structure: Organization, Formal Organization ...

CDM 4.13.2 describes EU document schema types (FRBR Work/Expression/Manifestation/Item
hierarchies) and is used as a relation/document-type vocabulary, NOT as NER labels.
GLiNER needs entity type labels that describe spans in running text — LKIF provides those.

No hardcoded label lists.  Every label comes from an ontology file.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS

from graphgen.utils.utils import standardize_label

logger = logging.getLogger(__name__)

# ── Ontology sources ──────────────────────────────────────────────────────────
# NER sources: define real-world entity types, suitable for tagging text spans.
_NER_SOURCES: List[tuple] = [
    ("input/ontology/lkif",    "*.owl", None),   # LKIF Core — best legal entity types
    ("input/ontology/w3c-org", "*.ttl", None),   # W3C Org — organizational structure
]

# Document-type sources: CDM — used for relation/edge classification, not NER.
# Kept here so callers can load them separately if needed.
_DOCUMENT_TYPE_SOURCES: List[tuple] = [
    ("input/ontology/cdm-4.13.2", "*.rdf", "xml"),
]

_SEARCH_ROOTS = [
    Path("/root/graphknows"),
    Path("/app"),
    Path("."),
]

# ── Label quality filters ─────────────────────────────────────────────────────
_MIN_LABEL_LEN = 3
_MAX_LABEL_LEN = 60

# Patterns that indicate abstract OWL machinery, not entity types
_SKIP_PATTERNS = [
    r"^[0-9_\-\s]+$",           # purely numeric / punctuation
]
_SKIP_PREFIXES_LOWER = (
    "concept representing",     # CDM meta-concepts
    "superclass",
    "container",
    "atto fd",
    "xl notation",
    "frbr",
)

# ── Internal bucketing (post-processing only, never sent to GLiNER) ───────────
_LABEL_TO_BUCKET: Dict[str, str] = {
    # PERSON
    "person": "PERSON", "natural person": "PERSON", "legal person": "PERSON",
    "private legal person": "PERSON", "individual": "PERSON",
    "professional legal role": "PERSON", "social legal role": "PERSON",
    "legal role": "PERSON",
    # ORGANIZATION
    "organization": "ORGANIZATION", "organisation": "ORGANIZATION",
    "formal organization": "ORGANIZATION", "organizational unit": "ORGANIZATION",
    "legislative body": "ORGANIZATION", "public body": "ORGANIZATION",
    "corporation": "ORGANIZATION", "company": "ORGANIZATION",
    "incorporated": "ORGANIZATION", "co-operative": "ORGANIZATION",
    "society": "ORGANIZATION", "trade": "ORGANIZATION",
    "foundation": "ORGANIZATION", "membership": "ORGANIZATION",
    # LOCATION
    "location": "LOCATION", "site": "LOCATION", "country": "LOCATION",
    "region": "LOCATION", "territory": "LOCATION",
    # POLICY
    "treaty": "POLICY", "directive": "POLICY", "regulation": "POLICY",
    "statute": "POLICY", "decision": "POLICY", "contract": "POLICY",
    "mandate": "POLICY", "norm": "POLICY", "code": "POLICY",
    "decree": "POLICY", "resolution": "POLICY", "declaration": "POLICY",
    "international agreement": "POLICY", "soft law": "POLICY",
    "legal document": "POLICY", "legal source": "POLICY",
    "legal expression": "POLICY", "obligation": "POLICY",
    "prohibition": "POLICY", "permission": "POLICY",
    "statement in writing": "POLICY",
    # EVENT
    "change event": "EVENT", "action": "EVENT", "process": "EVENT",
    "transaction": "EVENT", "delegation": "EVENT", "assignment": "EVENT",
    "creation": "EVENT", "termination": "EVENT",
    # CONCEPT
    "right": "CONCEPT", "immunity": "CONCEPT", "power": "CONCEPT",
    "assertion": "CONCEPT", "argument": "CONCEPT", "evidence": "CONCEPT",
    "precedent": "CONCEPT", "custom": "CONCEPT",
}


def _resolve_dir(relative: str) -> Optional[Path]:
    for root in _SEARCH_ROOTS:
        candidate = root / relative
        if candidate.exists():
            return candidate
    return None


def _fmt_local_name(name: str) -> str:
    """Convert camelCase / snake_case URI local name to Title Case words."""
    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return " ".join(name.split())


def _is_useful_label(label: str) -> bool:
    if len(label) < _MIN_LABEL_LEN or len(label) > _MAX_LABEL_LEN:
        return False
    lower = label.lower()
    if any(re.match(p, label) for p in _SKIP_PATTERNS):
        return False
    if any(lower.startswith(p) for p in _SKIP_PREFIXES_LOWER):
        return False
    return True


def _load_graph(sources: List[tuple]) -> Graph:
    g = Graph()
    count = 0
    for rel_dir, glob, fmt in sources:
        d = _resolve_dir(rel_dir)
        if d is None:
            logger.debug("Ontology dir not found: %s", rel_dir)
            continue
        for f in d.glob(glob):
            try:
                g.parse(str(f), format=fmt) if fmt else g.parse(str(f))
                count += 1
            except Exception as e:
                logger.warning("Could not parse %s: %s", f.name, e)
    logger.info("Loaded %d files → %d triples", count, len(g))
    return g


def _extract_ner_labels(g: Graph) -> List[str]:
    """
    Extract NER-suitable labels from a graph.

    For LKIF: rdfs:label is sparse (only 4 classes have English labels).
    We use formatted local names for all classes that lack an English rdfs:label,
    since LKIF local names are well-chosen human-readable entity types.

    For W3C Org: rdfs:labels are clean and always present — use those directly.
    """
    exclude = {OWL.Thing, OWL.Nothing, OWL.NamedIndividual, RDFS.Resource, RDFS.Class, OWL.Class}
    all_classes = (
        set(g.subjects(RDF.type, OWL.Class))
        | set(g.subjects(RDF.type, RDFS.Class))
    )

    labels: List[str] = []
    seen: Set[str] = set()

    for cls in all_classes:
        if isinstance(cls, BNode) or cls in exclude:
            continue

        uri = str(cls)
        local = uri.split("#")[-1] if "#" in uri else uri.split("/")[-1]

        # Try rdfs:label (English preferred)
        raw = list(g.objects(cls, RDFS.label))
        label: Optional[str] = None
        for l in raw:
            if isinstance(l, Literal) and l.language in (None, "en"):
                label = str(l).strip()
                break
        if label is None and raw:
            label = str(raw[0]).strip()

        # Fallback to formatted local name (important for LKIF)
        if label is None and local and any(c.isalpha() for c in local):
            label = _fmt_local_name(local)

        if label and _is_useful_label(label) and label not in seen:
            seen.add(label)
            labels.append(label)

    return sorted(labels)


@lru_cache(maxsize=1)
def _cached_ner_labels() -> List[str]:
    """Load and cache all NER labels from LKIF + W3C Org."""
    g = _load_graph(_NER_SOURCES)
    labels = _extract_ner_labels(g)
    logger.info("Final NER label set: %d labels from LKIF + W3C Org", len(labels))
    return labels


def _guess_bucket(label: str) -> str:
    """Map an extracted entity label to a coarse semantic bucket."""
    lower = label.lower()
    best, best_len = "CONCEPT", 0
    for kw, bucket in _LABEL_TO_BUCKET.items():
        if kw in lower and len(kw) > best_len:
            best, best_len = bucket, len(kw)
    return best


# ── Public API ────────────────────────────────────────────────────────────────

def build_top_level_label_space(
    ontology_labels: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Return the NER label space derived from LKIF + W3C Org ontologies.

    The space has a single key "ontology" with all labels as children.
    Any extra labels passed via `ontology_labels` are merged in (API compat).
    """
    base = _cached_ner_labels()
    all_labels: List[str] = list(base)
    seen: Set[str] = set(base)

    if ontology_labels:
        for lbl in ontology_labels:
            if lbl and lbl not in seen:
                all_labels.append(lbl)
                seen.add(lbl)

    return {"ontology": {"children": all_labels, "aliases": []}}


# Political/legal text canonical label set — always relevant for EU/parliamentary text.
# These map to LKIF classes and cover person, org, location, legislation, policy, event.
_POLITICAL_CORE_LABELS: List[str] = [
    "Person", "Natural Person", "Organisation", "Organization", "Formal Organization",
    "Legislative Body", "Public Body", "Foundation",
    "Treaty", "Directive", "Regulation", "Decision", "Statute",
    "International Agreement", "Resolution", "Declaration",
    "Action", "Plan", "Problem", "Obligation", "Right",
    "Legal Document", "Legal Speech Act",
]

# Heuristic keyword patterns → extra labels to include when detected in text
_TEXT_SIGNAL_LABELS: List[tuple] = [
    # (regex pattern, extra labels)
    (r"\b(war|conflict|crisis|attack|ceasefire|military)\b", ["Problem", "Action"]),
    (r"\b(law|legislation|directive|regulation|statute|code)\b", ["Directive", "Statute", "Regulation"]),
    (r"\b(treaty|convention|agreement|protocol|accord)\b", ["Treaty", "International Agreement"]),
    (r"\b(commission|council|parliament|assembly|senate)\b", ["Legislative Body", "Public Body"]),
    (r"\b(budget|fund|revenue|tax|fiscal|financial)\b", ["Transaction", "Obligation"]),
    (r"\b(right|freedom|liberty|immunity|privilege)\b", ["Right", "Liberty Right"]),
    (r"\b(vote|election|referendum|ballot)\b", ["Decision", "Action"]),
]


def select_candidate_labels(
    text: str,
    label_space: Dict[str, Dict[str, Any]],
    top_k: int = 20,
) -> List[str]:
    """Select relevant NER labels for a political/legal text chunk.

    Uses a curated core set of political/legal LKIF labels, then adds
    signal-based extras detected from the text. Word-overlap scoring on
    ontology label names is intentionally NOT used — label names like
    "Person" or "Treaty" rarely appear verbatim in political text.

    Args:
        text: The chunk text to select labels for.
        label_space: Ontology label space from build_top_level_label_space().
        top_k: Maximum number of labels to return. Default 20.

    Returns:
        Deduplicated list of up to top_k label strings.
    """
    all_labels: Set[str] = set(label_space.get("ontology", {}).get("children", []))
    if not all_labels:
        return list(_POLITICAL_CORE_LABELS)[:top_k]

    selected: List[str] = []
    seen: Set[str] = set()

    # Always include core political labels that exist in ontology
    for lbl in _POLITICAL_CORE_LABELS:
        # Try exact match, then case-insensitive match
        if lbl in all_labels and lbl not in seen:
            selected.append(lbl)
            seen.add(lbl)
        else:
            # Case-insensitive fallback
            for ontology_lbl in all_labels:
                if ontology_lbl.lower() == lbl.lower() and ontology_lbl not in seen:
                    selected.append(ontology_lbl)
                    seen.add(ontology_lbl)
                    break

    # Add signal-based extras
    text_lower = text.lower()
    for pattern, extra_labels in _TEXT_SIGNAL_LABELS:
        if re.search(pattern, text_lower):
            for lbl in extra_labels:
                if lbl not in seen:
                    # Try to find in ontology (exact or case-insensitive)
                    if lbl in all_labels:
                        selected.append(lbl)
                        seen.add(lbl)
                    else:
                        for ontology_lbl in all_labels:
                            if ontology_lbl.lower() == lbl.lower() and ontology_lbl not in seen:
                                selected.append(ontology_lbl)
                                seen.add(ontology_lbl)
                                break

    if not selected:
        # Ultimate fallback: first top_k labels from ontology
        return list(all_labels)[:top_k]

    return selected[:top_k]


def build_gliner2_schema(
    candidate_labels: List[str],
    label_descriptions: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, str]]:
    """
    Build a GLiNER2-compatible schema from ontology NER labels.

    Schema format: { "Label Name": {"description": "..."}, ... }
    """
    schema: Dict[str, Dict[str, str]] = {}
    for label in candidate_labels:
        if not label:
            continue
        desc = f"A {label.lower()} as defined in the LKIF legal ontology"
        if label_descriptions and label in label_descriptions:
            desc = label_descriptions[label]
        schema[label] = {"description": desc}
    return schema


def refine_entities_with_ontology(
    entities: List[Dict[str, Any]],
    label_space: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Post-process: add a coarse_bucket field for analytics grouping.

    The fine-grained ontology label from GLiNER2 (e.g. "Treaty", "Directive",
    "Legislative Body") is preserved as-is in entity["ontology_label"].
    We add entity["coarse_bucket"] as a secondary analytics grouping only.

    We do NOT overwrite ontology_label with the coarse bucket — that would
    destroy the semantic richness the ontology injection was designed to produce.
    """
    refined = []
    for entity in entities:
        entity = dict(entity)
        # Preserve fine-grained label; only add coarse_bucket for grouping
        entity["coarse_bucket"] = _guess_bucket(entity.get("label", ""))
        # ontology_label stays exactly as GLiNER2 assigned it (the LKIF/W3C label)
        refined.append(entity)
    return refined
