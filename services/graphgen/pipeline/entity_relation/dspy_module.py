import dspy
from typing import List, Tuple
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
    """
    Extract knowledge graph triplets (source, relation, target) from text.
    Use the provided ontology classes to filter allowed entity types if applicable, but primarily focus on the text.
    Use the provided entity hints as a guide for what entities might be present.
    ONLY extract relations that are explicitly stated in the text and have high confidence.
    Avoid "fluff" or trivial relations. Focus on significant interactions, facts, and properties.
    """
    text: str = dspy.InputField(desc="The input text to extract relations from.")
    ontology_classes: List[str] = dspy.InputField(desc="List of allowed ontology classes/types for entities.")
    entity_hints: List[str] = dspy.InputField(desc="List of pre-identified entities to guide extraction.")
    triplets: List[Triplet] = dspy.OutputField(desc="List of extracted triplets.")

class GraphExtractorModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(EntityRelation)

    def forward(self, text: str, ontology_classes: List[str], entity_hints: List[str]):
        return self.extract(text=text, ontology_classes=ontology_classes, entity_hints=entity_hints)
