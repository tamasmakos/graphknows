"""Ontology-guided label space and entity refinement for the graphgen pipeline.

This module owns the political-domain label taxonomy, GLiNER2 schema builder,
candidate-label selector, and entity-refinement logic.  It is the single source
of truth for how raw GLiNER2 outputs are mapped to the project's canonical
entity types.

Replaces the old ``graphgen.prototype_gliner2_ontology`` research prototype.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx

from graphknows.graphgen.utils import standardize_label

# TYPE_CHECKING guard avoids circular import at runtime
if TYPE_CHECKING:
    from graphknows.ingestion.domain.taxonomy import TaxonomyConfig

# ---------------------------------------------------------------------------
# Label taxonomy
# ---------------------------------------------------------------------------

DEFAULT_GROUP_RULES: dict[str, list[str]] = {
    "PERSON": ["PERSON", "POLITICIAN", "COMMISSIONER", "PRESIDENT", "CHAIRPERSON"],
    "ORGANIZATION": [
        "ORGANIZATION",
        "POLITICAL_PARTY",
        "INSTITUTION",
        "AGENCY",
        "COMMISSION",
        "PARLIAMENT",
        "COUNCIL",
        "BANK",
    ],
    "LOCATION": ["LOCATION", "COUNTRY", "CITY", "REGION", "PLACE", "CAPITAL"],
    "POLICY": ["POLICY", "REGULATION", "LAW", "DIRECTIVE", "TREATY", "FRAMEWORK"],
    "EVENT": ["EVENT", "SUMMIT", "MEETING", "ELECTION", "CRISIS", "WAR", "DEBATE"],
}

TOP_LEVEL_ALIASES: dict[str, list[str]] = {
    "PERSON": ["person", "president", "commissioner", "leader", "prime minister", "speaker"],
    "ORGANIZATION": [
        "organization",
        "commission",
        "parliament",
        "council",
        "party",
        "agency",
        "bank",
        "kremlin",
        "army",
        "military",
        "union",
    ],
    "LOCATION": [
        "location",
        "country",
        "city",
        "region",
        "brussels",
        "france",
        "germany",
        "ukraine",
        "mediterranean",
        "europe",
    ],
    "POLICY": [
        "policy",
        "regulation",
        "law",
        "directive",
        "framework",
        "migration",
        "climate",
        "sanctions",
        "aid",
        "health",
        "security",
        "energy",
        "defence",
        "defense",
    ],
    "EVENT": ["event", "summit", "meeting", "election", "crisis", "war", "debate", "conference"],
}

# Fallback candidate-label hints used when no taxonomy is provided
DEFAULT_CANDIDATE_HINTS: dict[str, list[str]] = {
    "PERSON": ["prime minister", "president", "speaker", "leader"],
    "ORGANIZATION": [
        "kremlin",
        "army",
        "military",
        "parliament",
        "commission",
        "bank",
        "union",
    ],
    "LOCATION": ["ukraine", "italy", "mediterranean", "europe", "eurozone"],
    "POLICY": [
        "sanctions",
        "aid",
        "migration",
        "climate",
        "health",
        "security",
        "energy",
        "policy",
    ],
    "EVENT": ["invasion", "conference", "debate", "crisis", "war"],
}

# Hard overrides that bypass the normal child-alias scoring
POLITICAL_TYPE_OVERRIDES: dict[str, str] = {
    "KREMLIN": "ORGANIZATION",
    "RUSSIAN_ARMY": "ORGANIZATION",
    "SANCTIONS": "POLICY_INSTRUMENT",
    "AID": "POLICY_INSTRUMENT",
    "ENERGY_DEPENDENCE": "POLICY_INSTRUMENT",
    "SECURITY_AND_DEFENCE_UNION": "POLICY_INSTRUMENT",
    "MIGRATION": "POLICY_INSTRUMENT",
    "CLIMATE": "POLICY_INSTRUMENT",
    "HEALTH": "POLICY_INSTRUMENT",
    "SECURITY": "POLICY_INSTRUMENT",
    "POLICY_CHANGES": "POLICY_INSTRUMENT",
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_top_level_label_space(
    ontology_labels: list[str],
    taxonomy: "TaxonomyConfig | None" = None,
) -> dict[str, dict[str, Any]]:
    """Build a hierarchical label-space dict from a flat list of entity labels.

    When *taxonomy* is provided all group rules and aliases are read from it;
    otherwise the module-level ``DEFAULT_GROUP_RULES`` / ``TOP_LEVEL_ALIASES``
    constants are used as a fallback.

    Returns a mapping of ``{TOP_LABEL: {aliases, children, child_aliases}}``.
    """
    normalized = [standardize_label(label) for label in ontology_labels]
    label_space: dict[str, dict[str, Any]] = {}

    if taxonomy is not None:
        group_rules = {k: v.children for k, v in taxonomy.entity_groups.items()}
        group_aliases = {k: v.aliases for k, v in taxonomy.entity_groups.items()}
    else:
        group_rules = DEFAULT_GROUP_RULES
        group_aliases = TOP_LEVEL_ALIASES

    for top_label, child_rules in group_rules.items():
        children = [
            label
            for label in normalized
            if label in child_rules or any(rule in label for rule in child_rules)
        ]
        if top_label in normalized and top_label not in children:
            children.insert(0, top_label)
        if children:
            label_space[top_label] = {
                "aliases": group_aliases.get(top_label, [top_label.lower()]),
                "children": sorted(set(children)),
                "child_aliases": {
                    child: _default_aliases_for_label(child) for child in sorted(set(children))
                },
            }

    return label_space


def build_gliner2_schema(
    candidate_labels: list[str],
    label_descriptions: dict[str, str] | None = None,
    taxonomy: "TaxonomyConfig | None" = None,
) -> dict[str, dict[str, str]]:
    """Build the entity-schema dict consumed by GLiNER2.

    When *taxonomy* is provided per-group descriptions are read from it;
    ``label_descriptions`` overrides take precedence over taxonomy descriptions,
    which take precedence over the module-level defaults.

    Args:
        candidate_labels: Labels to include (typically the top-level taxonomy keys).
        label_descriptions: Optional caller-supplied per-label description overrides.
        taxonomy: Optional domain taxonomy; provides per-group descriptions.

    Returns:
        ``{"entities": {label_lower: description, ...}}``
    """
    _defaults: dict[str, str] = {
        "PERSON": "A named person, political leader, office-holder, or individual speaker.",
        "ORGANIZATION": "An institution, parliament, commission, party, bank, or agency.",
        "LOCATION": "A named place, city, region, country, or geopolitical location.",
        "POLICY": "A named policy, directive, regulation, treaty, or governance framework.",
        "EVENT": "A summit, election, crisis, war, debate, or named political event.",
    }
    if taxonomy is not None:
        taxonomy_descs = {
            k: v.description.strip()
            for k, v in taxonomy.entity_groups.items()
            if v.description.strip()
        }
        descriptions = {**_defaults, **taxonomy_descs, **(label_descriptions or {})}
    else:
        descriptions = {**_defaults, **(label_descriptions or {})}
    return {
        "entities": {
            label.lower(): descriptions.get(label, label.lower()) for label in candidate_labels
        }
    }


def select_candidate_labels(
    text: str,
    label_space: dict[str, dict[str, Any]],
    top_k: int = 5,
    taxonomy: "TaxonomyConfig | None" = None,
) -> list[str]:
    """Rank top-level labels by how well they match *text* and return the best *top_k*.

    Uses alias overlap and domain-specific hint words for scoring.
    When *taxonomy* is provided, alias-based hints are derived from it;
    otherwise the module-level political-hint constants are used.
    Always returns at least *top_k* labels even when none score > 0.
    """
    lowered = text.lower()

    _hints: dict[str, list[str]] = (
        {k: v.aliases for k, v in taxonomy.entity_groups.items()}
        if taxonomy is not None
        else DEFAULT_CANDIDATE_HINTS
    )

    scores: list[tuple[str, int]] = []
    for top_label, spec in label_space.items():
        score = 0
        for alias in spec.get("aliases", []):
            if alias in lowered:
                score += max(3, len(alias.split()) * 2)
        for child in spec.get("children", []):
            for token in child.lower().replace("_", " ").split():
                if len(token) > 3 and token in lowered:
                    score += 1
        for hint in _hints.get(top_label, []):
            if hint in lowered:
                score += 4
        scores.append((top_label, score))

    ranked = sorted(scores, key=lambda item: (-item[1], item[0]))
    selected = [label for label, score in ranked if score > 0][:top_k]
    if not selected:
        selected = [label for label, _ in ranked[:top_k]]
    return selected


def refine_entities_with_ontology(
    entities: list[dict[str, Any]],
    label_space: dict[str, dict[str, Any]],
    taxonomy: "TaxonomyConfig | None" = None,
) -> list[dict[str, Any]]:
    """Map raw GLiNER2 entity labels to canonical ontology labels.

    Sets ``entity["ontology_label"]`` on every item in *entities* and returns
    the refined list.  When *taxonomy* is provided its ``overrides`` dict is
    used in place of the module-level ``POLITICAL_TYPE_OVERRIDES``.
    """
    active_overrides = taxonomy.overrides if taxonomy is not None else POLITICAL_TYPE_OVERRIDES

    refined: list[dict[str, Any]] = []
    for entity in entities:
        item = dict(entity)
        top_label = standardize_label(item.get("label", ""))
        text = item.get("text", "")
        text_norm = standardize_label(text)
        text_lower = text.lower()

        ontology_label = active_overrides.get(text_norm, top_label)

        spec = label_space.get(top_label)
        if spec and text_norm not in active_overrides:
            best_child = top_label
            best_score = -1
            for child in spec.get("children", []):
                aliases = spec.get("child_aliases", {}).get(
                    child, _default_aliases_for_label(child)
                )
                score = sum(1 for alias in aliases if alias in text_lower)
                if score > best_score:
                    best_child = child
                    best_score = score
            ontology_label = best_child

        item["ontology_label"] = ontology_label
        refined.append(item)
    return refined


def build_relations_from_entities(
    entities: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build pairwise co-occurrence relations from a list of refined entities."""
    relations: list[dict[str, str]] = []
    for idx, source in enumerate(entities):
        for target in entities[idx + 1 :]:
            relations.append(
                {
                    "source": source["text"],
                    "target": target["text"],
                    "relation": (
                        f"CO_OCCURS_{source['ontology_label']}_TO_{target['ontology_label']}"
                    ),
                }
            )
    return relations


def export_graphml_and_cypher_artifacts(
    entities: list[dict[str, Any]],
    relations: list[dict[str, str]],
    output_dir: Path,
) -> dict[str, str]:
    """Serialise entities and relations to GraphML, Cypher, and a JSON summary.

    Args:
        entities: Refined entity dicts (must have ``text``, ``label``, ``ontology_label``).
        relations: Relation dicts (must have ``source``, ``target``, ``relation``).
        output_dir: Directory where output files are written (created if absent).

    Returns:
        Mapping of ``{artifact_name: file_path}`` for all written files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    graph = nx.DiGraph()

    for entity in entities:
        node_id = standardize_label(entity["text"])
        graph.add_node(
            node_id,
            name=entity["text"],
            top_label=entity["label"],
            ontology_label=entity["ontology_label"],
            score=float(entity.get("score", 0.0)),
        )

    for relation in relations:
        source = standardize_label(relation["source"])
        target = standardize_label(relation["target"])
        graph.add_edge(source, target, relation=relation["relation"])

    graphml_path = output_dir / "ontology_graph.graphml"
    nx.write_graphml(graph, graphml_path)

    cypher_path = output_dir / "ontology_memgraph.cypher"
    with cypher_path.open("w", encoding="utf-8") as fh:
        fh.write("CREATE INDEX ON :ENTITY(name);\\n")
        for entity in entities:
            node_id = standardize_label(entity["text"])
            fh.write(
                "MERGE (n:ENTITY {{id: '{id}'}}) "
                "SET n.name={name}, n.top_label='{top}', "
                "n.ontology_label='{ont}', n.score={score};\\n".format(
                    id=node_id,
                    name=json.dumps(entity["text"]),
                    top=entity["label"],
                    ont=entity["ontology_label"],
                    score=float(entity.get("score", 0.0)),
                )
            )
        for relation in relations:
            source = standardize_label(relation["source"])
            target = standardize_label(relation["target"])
            fh.write(
                "MATCH (a:ENTITY {{id: '{src}'}}), (b:ENTITY {{id: '{tgt}'}}) "
                "MERGE (a)-[r:{rel_type}]->(b) "
                "SET r.weight = coalesce(r.weight, 0) + 1;\\n".format(
                    src=source, tgt=target, rel_type=standardize_label(relation["relation"])
                )
            )

    summary_path = output_dir / "ontology_summary.json"
    summary = {
        "entity_count": len(entities),
        "relation_count": len(relations),
        "top_labels": sorted({e["label"] for e in entities}),
        "ontology_labels": sorted({e["ontology_label"] for e in entities}),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "graphml": str(graphml_path),
        "memgraph_cypher": str(cypher_path),
        "summary_json": str(summary_path),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _default_aliases_for_label(label: str) -> list[str]:
    pretty = label.lower().replace("_", " ")
    aliases = {pretty}
    for token in pretty.split():
        if len(token) > 3:
            aliases.add(token)
    return sorted(aliases)
