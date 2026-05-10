from pathlib import Path
from typing import Dict, List, Any

import networkx as nx

from graphgen.utils.utils import standardize_label


DEFAULT_GROUP_RULES = {
    "PERSON": ["PERSON", "POLITICIAN", "COMMISSIONER", "PRESIDENT", "CHAIRPERSON"],
    "ORGANIZATION": ["ORGANIZATION", "POLITICAL_PARTY", "INSTITUTION", "AGENCY", "COMMISSION", "PARLIAMENT", "COUNCIL", "BANK"],
    "LOCATION": ["LOCATION", "COUNTRY", "CITY", "REGION", "PLACE", "CAPITAL"],
    "POLICY": ["POLICY", "REGULATION", "LAW", "DIRECTIVE", "TREATY", "FRAMEWORK"],
    "EVENT": ["EVENT", "SUMMIT", "MEETING", "ELECTION", "CRISIS", "WAR", "DEBATE"],
}

TOP_LEVEL_ALIASES = {
    "PERSON": ["person", "president", "commissioner", "leader", "prime minister", "speaker"],
    "ORGANIZATION": ["organization", "commission", "parliament", "council", "party", "agency", "bank", "kremlin", "army", "military", "union"],
    "LOCATION": ["location", "country", "city", "region", "brussels", "france", "germany", "ukraine", "mediterranean", "europe"],
    "POLICY": ["policy", "regulation", "law", "directive", "framework", "migration", "climate", "sanctions", "aid", "health", "security", "energy", "defence", "defense"],
    "EVENT": ["event", "summit", "meeting", "election", "crisis", "war", "debate", "conference"],
}

POLITICAL_TYPE_OVERRIDES = {
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


def build_top_level_label_space(ontology_labels: List[str]) -> Dict[str, Dict[str, Any]]:
    normalized = [standardize_label(label) for label in ontology_labels]
    label_space: Dict[str, Dict[str, Any]] = {}

    for top_label, child_rules in DEFAULT_GROUP_RULES.items():
        children = [label for label in normalized if label in child_rules or any(rule in label for rule in child_rules)]
        if top_label in normalized and top_label not in children:
            children.insert(0, top_label)
        if children:
            label_space[top_label] = {
                "aliases": TOP_LEVEL_ALIASES.get(top_label, [top_label.lower()]),
                "children": sorted(set(children)),
                "child_aliases": {
                    child: _default_aliases_for_label(child)
                    for child in sorted(set(children))
                },
            }

    return label_space


def _default_aliases_for_label(label: str) -> List[str]:
    pretty = label.lower().replace("_", " ")
    aliases = {pretty}
    for token in pretty.split():
        if len(token) > 3:
            aliases.add(token)
    return sorted(aliases)


def build_gliner2_schema(
    candidate_labels: List[str],
    label_descriptions: Dict[str, str] | None = None,
) -> Dict[str, Dict[str, str]]:
    default_descriptions = {
        "PERSON": "A named person, political leader, office-holder, or individual speaker.",
        "ORGANIZATION": "An institution, parliament, commission, party, bank, or agency.",
        "LOCATION": "A named place, city, region, country, or geopolitical location.",
        "POLICY": "A named policy, directive, regulation, treaty, or governance framework.",
        "EVENT": "A summit, election, crisis, war, debate, or named political event.",
    }
    descriptions = {**default_descriptions, **(label_descriptions or {})}
    return {"entities": {candidate.lower(): descriptions.get(candidate, candidate.lower()) for candidate in candidate_labels}}


def select_candidate_labels(text: str, label_space: Dict[str, Dict[str, Any]], top_k: int = 5) -> List[str]:
    lowered = text.lower()
    scores = []
    political_hints = {
        "PERSON": ["prime minister", "president", "speaker", "leader", "mario draghi", "roberta metsola"],
        "ORGANIZATION": ["kremlin", "army", "military", "parliament", "commission", "bank", "union"],
        "LOCATION": ["ukraine", "italy", "mediterranean", "europe", "eurozone"],
        "POLICY": ["sanctions", "aid", "migration", "climate", "health", "security", "energy", "policy"],
        "EVENT": ["invasion", "conference", "debate", "crisis", "war"],
    }
    for top_label, spec in label_space.items():
        score = 0
        for alias in spec.get("aliases", []):
            if alias in lowered:
                score += max(3, len(alias.split()) * 2)
        for child in spec.get("children", []):
            child_tokens = child.lower().replace("_", " ").split()
            if any(token in lowered for token in child_tokens if len(token) > 3):
                score += 1
        for hint in political_hints.get(top_label, []):
            if hint in lowered:
                score += 4
        scores.append((top_label, score))

    ranked = sorted(scores, key=lambda item: (-item[1], item[0]))
    selected = [label for label, score in ranked if score > 0][:top_k]
    if not selected:
        selected = [label for label, _ in ranked[:top_k]]
    return selected


def refine_entities_with_ontology(entities: List[Dict[str, Any]], label_space: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    refined = []
    for entity in entities:
        item = dict(entity)
        top_label = standardize_label(item.get("label", ""))
        text = item.get("text", "")
        text_norm = standardize_label(text)
        text_lower = text.lower()
        ontology_label = POLITICAL_TYPE_OVERRIDES.get(text_norm, top_label)

        spec = label_space.get(top_label)
        if spec and text_norm not in POLITICAL_TYPE_OVERRIDES:
            best_child = top_label
            best_score = -1
            for child in spec.get("children", []):
                aliases = spec.get("child_aliases", {}).get(child, _default_aliases_for_label(child))
                score = sum(1 for alias in aliases if alias in text_lower)
                if score > best_score:
                    best_child = child
                    best_score = score
            ontology_label = best_child

        item["ontology_label"] = ontology_label
        refined.append(item)
    return refined


def build_relations_from_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    relations = []
    for idx, source in enumerate(entities):
        for target in entities[idx + 1:]:
            relations.append(
                {
                    "source": source["text"],
                    "target": target["text"],
                    "relation": f"CO_OCCURS_{source['ontology_label']}_TO_{target['ontology_label']}",
                }
            )
    return relations


def export_graphml_and_memgraph_artifacts(
    entities: List[Dict[str, Any]],
    relations: List[Dict[str, str]],
    output_dir: Path,
) -> Dict[str, str]:
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

    graphml_path = output_dir / "prototype_ontology_graph.graphml"
    nx.write_graphml(graph, graphml_path)

    cypher_path = output_dir / "prototype_memgraph.cypher"
    with cypher_path.open("w", encoding="utf-8") as handle:
        handle.write("CREATE INDEX ON :Entity(name);\n")
        for entity in entities:
            node_id = standardize_label(entity["text"])
            handle.write(
                "MERGE (n:Entity {id: '%s'}) SET n.name=%s, n.top_label='%s', n.ontology_label='%s', n.score=%s;\n"
                % (
                    node_id,
                    __import__('json').dumps(entity["text"]),
                    entity["label"],
                    entity["ontology_label"],
                    float(entity.get("score", 0.0)),
                )
            )
        for relation in relations:
            source = standardize_label(relation["source"])
            target = standardize_label(relation["target"])
            handle.write(
                "MATCH (a:Entity {id: '%s'}), (b:Entity {id: '%s'}) MERGE (a)-[:%s]->(b);\n"
                % (source, target, standardize_label(relation["relation"]))
            )

    summary_path = output_dir / "prototype_summary.json"
    summary = {
        "entity_count": len(entities),
        "relation_count": len(relations),
        "top_labels": sorted({entity["label"] for entity in entities}),
        "ontology_labels": sorted({entity["ontology_label"] for entity in entities}),
    }
    summary_path.write_text(__import__('json').dumps(summary, indent=2), encoding="utf-8")

    return {
        "graphml": str(graphml_path),
        "memgraph_cypher": str(cypher_path),
        "summary_json": str(summary_path),
    }
