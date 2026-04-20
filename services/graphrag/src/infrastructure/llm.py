import os
import logging
from dotenv import load_dotenv

# Native LlamaIndex imports
try:
    from llama_index.llms.groq import Groq
except ImportError:
    Groq = None

try:
    from llama_index.llms.openai import OpenAI
except ImportError:
    OpenAI = None

from src.llama.embeddings import embed_query

logger = logging.getLogger(__name__)
load_dotenv()


def get_llm(purpose: str = None):
    """
    Get a LlamaIndex-native LLM.
    Prioritizes Groq, then OpenAI.
    """
    from src.infrastructure.config import get_app_config
    
    settings = get_app_config()
    
    groq_api_key = os.environ.get("GROQ_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    if Groq is None:
        logger.warning("llama_index.llms.groq.Groq import failed. Is llama-index-llms-groq installed?")
    if OpenAI is None:
        logger.warning("llama_index.llms.openai.OpenAI import failed. Is llama-index-llms-openai installed?")

    if Groq and groq_api_key:
        model = settings.groq_model
        
        # Override based on purpose
        if purpose == "keywords":
             model = settings.keywords_model
        elif purpose == "chat":
             model = settings.chat_model

        return Groq(
            model=model,
            api_key=groq_api_key,
            temperature=0.0
        )
    
    if OpenAI and openai_api_key:
        return OpenAI(
            model="gpt-3.5-turbo",
            api_key=openai_api_key,
            temperature=0.0
        )
    
    if not groq_api_key:
        logger.warning("GROQ_API_KEY not found in environment.")
    if not openai_api_key:
        logger.warning("OPENAI_API_KEY not found in environment.")
        
    logger.warning("No suitable LLM provider found (checked GROQ_API_KEY, OPENAI_API_KEY).")
    return None


class LlamaIndexEmbeddingsAdapter:
    """Adapts src.llama.embeddings to expected interface."""
    def embed_query(self, text: str):
        return embed_query(text)
    
    def embed_documents(self, texts: list[str]):
        """Embed a list of texts."""
        instance = get_embedding_model() # Recursive usage if not careful, but here we likely mean the base function
        # Actually this adapter seems to wrap the sync function embed_query.
        # Let's keep it simple as it was before, just fixing the duplicates if any.
        return [embed_query(t) for t in texts]

    async def aembed_query(self, text: str):
        return embed_query(text)

    async def aembed_documents(self, texts: list[str]):
        return [embed_query(t) for t in texts]


def get_embedding_model():
    """Get embedding model instance."""
    return LlamaIndexEmbeddingsAdapter()
