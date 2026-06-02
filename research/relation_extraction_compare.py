"""Compare GLiNER2-only and LLM-assisted relation extraction.

This is research code: it does not write production memory. The baseline path
uses GLiNER2 with entity and relation labels injected from the active taxonomy.
The optional LLM-assisted mode runs after that same GLiNER2 pass.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from graphknows.ingestion.domain.taxonomy import load_taxonomy_artifacts, resolve_taxonomy_path
from graphknows.memory.extractor import ExtractionResult, GLiNER2EntityExtractor


@dataclass(frozen=True)
class CaseScore:
    id: str
    mode: str
    elapsed_s: float
    entity_count: int
    relation_count: int
    expected_count: int
    found_count: int
    forbidden_count: int
    violation_count: int
    recall: float
    forbidden_precision: float
    missed: list[dict[str, str]]
    violations: list[dict[str, str]]
    relations: list[dict[str, Any]]


def _norm(value: object) -> str:
    return str(value or "").lower().replace("-", " ").replace("_", " ").strip()


def _hint_match(candidate: object, hint: object) -> bool:
    candidate_norm = _norm(candidate)
    hint_tokens = [token for token in _norm(hint).split() if token]
    return bool(hint_tokens) and all(token in candidate_norm for token in hint_tokens)


def _relation_parts(relation: dict[str, Any]) -> tuple[str, str, str]:
    head = relation.get("head", relation.get("subject", ""))
    predicate = relation.get("relation", relation.get("predicate", ""))
    tail = relation.get("tail", relation.get("object", ""))
    return str(head), str(predicate), str(tail)


def _matches(extracted: dict[str, Any], expected: dict[str, str]) -> bool:
    head, predicate, tail = _relation_parts(extracted)
    return (
        _norm(predicate) == _norm(expected.get("predicate"))
        and _hint_match(head, expected.get("subject"))
        and _hint_match(tail, expected.get("object"))
    )


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Load JSONL relation quality cases."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def extract_case(
    text: str,
    *,
    mode: str,
    taxonomy_file: str,
) -> tuple[ExtractionResult, list[dict[str, Any]]]:
    """Extract entities and relations for one case."""
    if mode == "off":
        return ExtractionResult(), []

    extractor = GLiNER2EntityExtractor(taxonomy_path=resolve_taxonomy_path(taxonomy_file))
    gliner_result = extractor.extract(text)
    relations = list(gliner_result.relations)

    if mode == "llm_assisted":
        from graphknows.ingestion.ontology.rel_extractor import OntologyRelExtractor

        taxonomy = load_taxonomy_artifacts(taxonomy_file)
        rel_extractor = OntologyRelExtractor(
            taxonomy_file=taxonomy.taxonomy_path,
            relation_schema=taxonomy.relation_schema,
        )
        llm_relations = rel_extractor.extract(
            text=text,
            entities=gliner_result.entities,
            ontology_schema=None,
            ontology_id="research-relation-compare",
        )
        relations.extend(llm_relations)

    return gliner_result, relations


def score_case(
    case: dict[str, Any],
    mode: str,
    relations: list[dict[str, Any]],
    elapsed_s: float,
    entity_count: int,
) -> CaseScore:
    """Score extracted relations against fuzzy labeled expectations."""
    expected = list(case.get("expected", []))
    forbidden = list(case.get("forbidden", []))
    found_indexes: set[int] = set()
    for index, expected_relation in enumerate(expected):
        if any(_matches(relation, expected_relation) for relation in relations):
            found_indexes.add(index)

    violations = [
        forbidden_relation
        for forbidden_relation in forbidden
        if any(_matches(relation, forbidden_relation) for relation in relations)
    ]
    missed = [expected[index] for index in range(len(expected)) if index not in found_indexes]
    recall = len(found_indexes) / len(expected) if expected else 1.0
    forbidden_precision = 1.0 - (len(violations) / len(forbidden)) if forbidden else 1.0
    return CaseScore(
        id=str(case.get("id", "case")),
        mode=mode,
        elapsed_s=round(elapsed_s, 4),
        entity_count=entity_count,
        relation_count=len(relations),
        expected_count=len(expected),
        found_count=len(found_indexes),
        forbidden_count=len(forbidden),
        violation_count=len(violations),
        recall=round(recall, 4),
        forbidden_precision=round(forbidden_precision, 4),
        missed=missed,
        violations=violations,
        relations=relations,
    )


def run_cases(cases: list[dict[str, Any]], *, mode: str, taxonomy_file: str) -> list[CaseScore]:
    """Run and score all cases."""
    scores: list[CaseScore] = []
    for case in cases:
        started = time.perf_counter()
        extraction, relations = extract_case(
            str(case.get("text", "")),
            mode=mode,
            taxonomy_file=taxonomy_file,
        )
        elapsed = time.perf_counter() - started
        scores.append(score_case(case, mode, relations, elapsed, len(extraction.entities)))
    return scores


def summarize(scores: list[CaseScore]) -> dict[str, Any]:
    """Return aggregate metrics for a scored run."""
    expected = sum(score.expected_count for score in scores)
    found = sum(score.found_count for score in scores)
    forbidden = sum(score.forbidden_count for score in scores)
    violations = sum(score.violation_count for score in scores)
    return {
        "cases": len(scores),
        "expected_relations": expected,
        "found_relations": found,
        "relation_recall": round(found / expected, 4) if expected else 1.0,
        "forbidden_relations": forbidden,
        "violations": violations,
        "forbidden_precision": round(1.0 - (violations / forbidden), 4) if forbidden else 1.0,
        "total_elapsed_s": round(sum(score.elapsed_s for score in scores), 4),
        "avg_elapsed_s": round(sum(score.elapsed_s for score in scores) / len(scores), 4)
        if scores
        else 0.0,
        "total_entities": sum(score.entity_count for score in scores),
        "total_extracted_relations": sum(score.relation_count for score in scores),
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["gliner2", "llm_assisted", "off"], default="gliner2")
    parser.add_argument("--cases", default="research/relation_quality_cases.jsonl")
    parser.add_argument("--taxonomy-file", default="taxonomy.generic.yaml")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    scores = run_cases(
        load_cases(Path(args.cases)),
        mode=args.mode,
        taxonomy_file=args.taxonomy_file,
    )
    payload = {
        "mode": args.mode,
        "taxonomy_file": args.taxonomy_file,
        "summary": summarize(scores),
        "cases": [asdict(score) for score in scores],
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
