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


def get_llm(purpose: str = None):
    """
    Get a LangChain-compatible LLM.
    Prioritizes Groq, then OpenAI.
    """
    from src.infrastructure.config import get_app_config
    
    settings = get_app_config()
    
    groq_api_key = os.environ.get("GROQ_API_KEY") # Keep getting API key from env/os if not in settings? 
    # Wait, AppSettings has openai_api_key but not groq_api_key in its definition file? 
    # Ah, I didn't add groq_api_key to common/config/settings.py in previous step.
    # The original llm.py used os.environ for keys. 
    # Since only secrets go in .env, getting keys from os.environ is correct/standard.
    # BUT I should check if I should add keys to settings.py too for consistency?
    # No, AppSettings usually loads from .env.
    # Let's stick to using settings for MODELS. Keys can stay env.
    
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    if ChatGroq and groq_api_key:
        model = settings.groq_model
        
        # Override based on purpose
        if purpose == "keywords":
             model = settings.keywords_model
        elif purpose == "chat":
             model = settings.chat_model

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