import dspy
from typing import List, Optional
from pydantic import BaseModel, Field

class Finding(BaseModel):
    summary: str = Field(description="Short title of a key insight/finding")
    explanation: str = Field(description="Detailed explanation of the finding, citing specific entities or patterns observed.")

class CommunityReport(BaseModel):
    title: str = Field(description="A short, descriptive title for the community (3-10 words)")
    summary: str = Field(description="A comprehensive executive summary (3-5 sentences) covering the main topics, entities, and dynamics.")
    findings: List[Finding] = Field(description="List of key insights and findings about the community.")

class CommunitySummarizerSignature(dspy.Signature):
    """
    Synthesize a comprehensive report on a specific community of entities within a larger network.
    Analyze the provided community structure (entities and relationships) and text chunks.
    Identify the core theme, key patterns, and significant insights.
    """
    
    community_context = dspy.InputField(desc="XML-formatted string containing entities, relationships, and text chunks.")
    report: CommunityReport = dspy.OutputField(desc="Structured report containing title, summary, and findings.")

class CommunitySummarizerModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_report = dspy.ChainOfThought(CommunitySummarizerSignature)

    def forward(self, community_context: str):
        return self.generate_report(community_context=community_context)
