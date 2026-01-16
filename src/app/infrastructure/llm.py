import os

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from src.kg.llm import get_langchain_llm

# Load environment variables (including GROQ_API_KEY / GROQ_MODEL) from the
# same .env file used by the KG pipeline so LLM config is shared.
load_dotenv()


def get_llm():
    """
    Get a LangChain-compatible Groq LLM using the shared KG configuration.

    Delegates to `src.kg.llm.get_langchain_llm`, which reads:
    - GROQ_API_KEY / GROQ_MODEL from the environment, or
    - any overrides defined in the shared `config.yaml`.
    """
    return get_langchain_llm()


class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            self.model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            self.model = SentenceTransformer(model_name)

    def embed_query(self, text: str):
        return self.model.encode(text).tolist()


def get_embedding_model():
    """Get embedding model instance."""
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")



