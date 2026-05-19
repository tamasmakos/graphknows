import dspy
from typing import Any, Dict, List
from pydantic import BaseModel, Field


class Triplet(BaseModel):
    source: str = Field(description="The source entity")
    source_type: str = Field(description="The ontology class of the source entity")
    relation: str = Field(description="The relationship between source and target")
    target: str = Field(description="The target entity")
    target_type: str = Field(description="The ontology class of the target entity")
    confidence: float = Field(description="Confidence score (0.0-1.0) of this extraction")
    evidence: str = Field(description="Verbatim text snippet that supports this relation")


class EntityRelation(dspy.Signature):
    """Extract knowledge graph triplets (source, relation, target) from text.

    Rules:
    - Only extract relations EXPLICITLY stated in the text.
    - The *relation* field MUST be chosen from allowed_relation_types.
      Do NOT invent new relation labels.
    - Use entity_hints as the preferred surface forms for source/target.
    - Use candidate_relations as a starting point; verify them and add new ones.
    - Skip trivial or very uncertain relations.
    """

    text: str = dspy.InputField(desc="Input text to analyse.")
    ontology_classes: List[str] = dspy.InputField(
        desc="Allowed entity type labels for source/target classification."
    )
    entity_hints: List[str] = dspy.InputField(
        desc="Pre-identified entities (canonical surface forms) to guide extraction."
    )
    allowed_relation_types: List[str] = dspy.InputField(
        desc=(
            "Exhaustive list of permitted relation labels. "
            "The relation field in every output triplet MUST be one of these."
        )
    )
    candidate_relations: List[Dict[str, Any]] = dspy.InputField(
        desc=(
            "Candidate (source, relation, target) triplets produced by the fast "
            "GLiNER2 pass.  Verify these and include confirmed ones in the output."
        )
    )
    triplets: List[Triplet] = dspy.OutputField(
        desc="Verified and enriched list of triplets. relation must be from allowed_relation_types."
    )


class GraphExtractorModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(EntityRelation)

    def forward(
        self,
        text: str,
        ontology_classes: List[str],
        entity_hints: List[str],
        allowed_relation_types: List[str] | None = None,
        candidate_relations: List[Dict[str, Any]] | None = None,
    ):
        return self.extract(
            text=text,
            ontology_classes=ontology_classes,
            entity_hints=entity_hints,
            allowed_relation_types=allowed_relation_types or [],
            candidate_relations=candidate_relations or [],
        )
