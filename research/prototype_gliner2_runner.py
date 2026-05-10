from pathlib import Path
from typing import Dict, List, Any

from gliner2 import GLiNER2

from graphgen.prototype_gliner2_ontology import (
    build_top_level_label_space,
    select_candidate_labels,
    refine_entities_with_ontology,
    build_relations_from_entities,
    export_graphml_and_memgraph_artifacts,
)
from graphgen.utils.labels import resolve_entity_labels


def load_default_ontology_labels() -> List[str]:
    extraction_config = {
        "entity_labels": ["Person", "Organization", "Location", "Policy", "Event"],
        "ontology": {
            "enabled": True,
            "ontology_dir": "input/ontology/cdm-4.13.2",
            "namespace_filter": None,
            "merge_with_manual": True,
            "top_level_only": True,
            "min_subclasses": 1,
            "include_local_names": True,
        },
    }
    return resolve_entity_labels(extraction_config)


def _schema_from_candidates(candidates: List[str]) -> Dict[str, Dict[str, str]]:
    descriptions = {
        "PERSON": "A named person, political leader, office-holder, or individual speaker.",
        "ORGANIZATION": "An institution, parliament, commission, party, bank, or agency.",
        "LOCATION": "A named place, city, region, country, or geopolitical location.",
        "POLICY": "A named policy, directive, regulation, treaty, or governance framework.",
        "EVENT": "A summit, election, crisis, war, debate, or named political event.",
    }
    return {"entities": {candidate.lower(): descriptions.get(candidate, candidate.lower()) for candidate in candidates}}


def _flatten_gliner2_result(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    entities = []
    for label, spans in (result.get("entities") or {}).items():
        for span in spans:
            entities.append(
                {
                    "text": span["text"],
                    "label": label.upper(),
                    "score": float(span.get("confidence", 0.0)),
                    "start": span.get("start"),
                    "end": span.get("end"),
                }
            )
    return entities


def run_gliner2_ontology_prototype(
    text: str,
    ontology_labels: List[str],
    output_dir: str,
    model_name: str = 'fastino/gliner2-base-v1',
    top_k: int = 4,
    threshold: float = 0.25,
) -> Dict[str, Any]:
    label_space = build_top_level_label_space(ontology_labels)
    candidates = select_candidate_labels(text, label_space, top_k=top_k)
    schema = _schema_from_candidates(candidates)

    model = GLiNER2.from_pretrained(model_name)
    raw = model.extract(text, schema, threshold=threshold, include_confidence=True, include_spans=True)
    entities = _flatten_gliner2_result(raw)
    refined = refine_entities_with_ontology(entities, label_space)
    relations = build_relations_from_entities(refined)
    artifacts = export_graphml_and_memgraph_artifacts(refined, relations, Path(output_dir))

    result = {
        "model_name": model_name,
        "candidate_top_labels": candidates,
        "schema": schema,
        "raw_result": raw,
        "entities": refined,
        "relations": relations,
        "artifacts": artifacts,
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir, "prototype_result.json").write_text(__import__('json').dumps(result, indent=2), encoding="utf-8")
    return result
