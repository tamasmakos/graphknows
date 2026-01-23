from pydantic_settings import BaseSettings
from typing import Optional

class PipelineSettings(BaseSettings):
    # Define all env vars here with types
    falkordb_host: str = "falkordb"
    falkordb_port: int = 6379
    openai_api_key: Optional[str] = None
    input_dir: str = "/app/input"
    output_dir: str = "/app/output"
    
    # Add other necessary settings from the old config structure as needed
    # For now, we stick to the manifesto's example and essential paths

    class Config:
        env_file = ".env"
        extra = "ignore"
