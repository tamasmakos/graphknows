import re
import logging
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

def split_sentences(text: str) -> List[str]:
    """Split text into sentences using regex."""
    # Simple regex for sentence splitting
    sentence_endings = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
    sentences = re.split(sentence_endings, text)
    return [s.strip() for s in sentences if s.strip()]

async def process_document_splitting(
    content: str, 
    config: Dict[str, Any]
) -> List[str]: # Returns list of chunk strings
    """Process text using LangChain splitter."""
    
    try:
        # Configure splitter
        extraction_config = config.get('extraction', {})
        if hasattr(extraction_config, 'model_dump'):
            extraction_config = extraction_config.model_dump()
            
        chunk_size = extraction_config.get('chunk_size', 512)
        chunk_overlap = extraction_config.get('chunk_overlap', 50)
        
        # Ensure overlap is not larger than chunk_size
        if chunk_overlap >= chunk_size:
            chunk_overlap = max(20, chunk_size // 10)
            logger.warning(f"chunk_overlap too large, adjusted to {chunk_overlap}")
        
        logger.debug(f"RecursiveCharacterTextSplitter configured: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        chunks = splitter.split_text(content)
        return chunks
        
    except Exception as e:
        logger.error(f"Error in splitting: {e}")
        raise
