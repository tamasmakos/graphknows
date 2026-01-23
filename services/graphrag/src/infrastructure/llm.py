import os
import logging
from dotenv import load_dotenv

# Try importing langchain integrations
try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

from src.llama.embeddings import embed_query

logger = logging.getLogger(__name__)
load_dotenv()


def get_llm():
    """
    Get a LangChain-compatible LLM.
    Prioritizes Groq, then OpenAI.
    """
    groq_api_key = os.environ.get("GROQ_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    if ChatGroq and groq_api_key:
        model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        return ChatGroq(
            temperature=0,
            model_name=model,
            api_key=groq_api_key
        )
    
    if ChatOpenAI and openai_api_key:
        return ChatOpenAI(
            temperature=0,
            model="gpt-3.5-turbo",
            api_key=openai_api_key
        )
        
    logger.warning("No suitable LLM provider found (checked GROQ_API_KEY, OPENAI_API_KEY).")
    return None


class LlamaIndexEmbeddingsAdapter:
    """Adapts src.llama.embeddings to expected interface."""
    def embed_query(self, text: str):
        return embed_query(text)


def get_embedding_model():
    """Get embedding model instance."""
    return LlamaIndexEmbeddingsAdapter()