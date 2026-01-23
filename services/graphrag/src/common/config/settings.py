from pydantic_settings import BaseSettings
from typing import Optional

class AppSettings(BaseSettings):
    # Define all env vars here with types
    falkordb_host: str = "falkordb"
    falkordb_port: int = 6379
    openai_api_key: Optional[str] = None
    
    class Config:
        env_file = ".env"
        extra = "ignore"
