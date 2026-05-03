import dspy
import logging
import os
from typing import Dict, Any, List, Optional
from graphgen.config.llm import _extract_secret
from graphgen.config.llm import normalize_groq_model
from graphgen.pipeline.summarization.dspy_module import CommunitySummarizerModule
from graphgen.pipeline.summarization.models import SummarizationTask

logger = logging.getLogger(__name__)

class DSPySummarizer:
    """DSPy-based community summarizer."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize with config."""
        self.config = config
        self._configure_dspy()
        self.module = CommunitySummarizerModule()

    def _configure_dspy(self):
        """Configure DSPy LM (shared logic with entity extractor)."""
        llm_config = self.config.get('llm', {})
        if hasattr(llm_config, 'model_dump'):
            llm_config = llm_config.model_dump()

        # Try to find the best model name from config
        model = llm_config.get('summarization_model') or llm_config.get('base_model') or llm_config.get('model') or 'gpt-4o'

        # Check for Groq API Key
        infra_config = self.config.get('infra', {})
        groq_api_key = _extract_secret(infra_config, 'groq_api_key') or _extract_secret(llm_config, 'groq_api_key')
        if not groq_api_key:
            groq_api_key = os.environ.get('GROQ_API_KEY')

        # Determine provider and configure
        try:
             if groq_api_key:
                 logger.info(f"Configuring DSPy for Groq with model {model}")
                 groq_model = normalize_groq_model(model)
                 lm = dspy.LM(
                     model=groq_model,
                     api_key=groq_api_key,
                     api_base="https://api.groq.com/openai/v1",
                     temperature=0.0,
                     max_tokens=4096 # Higher limit for summarization
                 )
                 dspy.configure(lm=lm)
             else:
                 api_key = _extract_secret(llm_config, 'api_key') or os.environ.get('OPENAI_API_KEY')
                 base_url = llm_config.get('base_url')
                 lm = dspy.LM(
                     model=model,
                     api_key=api_key,
                     api_base=base_url,
                     temperature=0.0,
                     max_tokens=4096
                 )
                 dspy.configure(lm=lm)

        except Exception as e:
             logger.warning(f"Failed to configure DSPy LM: {e}")
             pass

    def _format_context_xml(self, task: SummarizationTask) -> str:
        """
        Format the task data into XML sections for the prompt.
        Adapted from core.py but specifically for DSPy context.
        """
        xml_parts = []
        
        # 1. Community Structure (Entities & Relations)
        structure_xml = ["<community_structure>"]
        
        if task.entities:
            structure_xml.append("  <entities>")
            # Sort by degree if available, else random
            sorted_ents = sorted(task.entities, key=lambda x: x.get('degree', 0), reverse=True)[:50]
            for ent in sorted_ents:
                structure_xml.append(f"    <entity name=\"{ent['name']}\" type=\"{ent.get('type', 'Unknown')}\" />")
            structure_xml.append("  </entities>")
            
        if task.relationships:
            structure_xml.append("  <relationships>")
            # Limit relationships - Task.relationships is now expected to be (src, rel, tgt, props) based on new robust design
            # But legacy code might still pass (src, rel, tgt). Handle both.
            
            rels_to_process = task.relationships[:100] # Slightly higher limit for DSPy
            
            for rel in rels_to_process:
                # Unpack safely
                if len(rel) == 4:
                    src, mode, tgt, props = rel
                elif len(rel) == 3:
                     src, mode, tgt = rel
                     props = {}
                else:
                    continue

                # Format attributes
                attr_str = ""
                if props:
                    # Prioritize evidence and confidence
                    evidence = props.get('evidence', '')
                    confidence = props.get('confidence')
                    
                    if evidence:
                        # Truncate evidence if too long
                        evidence = evidence[:100] + "..." if len(evidence) > 100 else evidence
                        attr_str += f" evidence=\"{evidence}\""
                    
                    if confidence is not None:
                        attr_str += f" confidence=\"{confidence}\""
                        
                structure_xml.append(f"    <rel source=\"{src}\" type=\"{mode}\" target=\"{tgt}\"{attr_str} />")
            structure_xml.append("  </relationships>")
        
        structure_xml.append("</community_structure>")
        xml_parts.append("\n".join(structure_xml))
        
        # 2. Sub-community Summaries
        if task.sub_summaries:
            sub_xml = ["<sub_communities>"]
            for sub in task.sub_summaries:
                sub_xml.append(f"  <sub_community id=\"{sub.get('id')}\">\n    {sub.get('summary')}\n  </sub_community>")
            sub_xml.append("</sub_communities>")
            xml_parts.append("\n".join(sub_xml))

        # 3. Text Chunks
        if task.chunk_texts:
            xml_parts.append("<text_chunks>")
            for i, text in enumerate(task.chunk_texts):
                # Simple truncation
                clean_text = text[:2000].replace("<", "&lt;").replace(">", "&gt;") 
                xml_parts.append(f"  <chunk id=\"{i}\">\n{clean_text}\n  </chunk>")
            xml_parts.append("</text_chunks>")
        
        return "\n".join(xml_parts)

    def summarize(self, task: SummarizationTask) -> Optional[Dict[str, Any]]:
        """Run the DSPy module to generate a report."""
        try:
            context_xml = self._format_context_xml(task)
            
            # DSPy prediction
            prediction = self.module(community_context=context_xml)
            
            if prediction.report:
                report = prediction.report
                # Convert to dict
                return report.model_dump()
            
            return None
            
        except Exception as e:
            logger.error(f"DSPy Summarization failed for task {task.task_id}: {e}", exc_info=True)
            return None
