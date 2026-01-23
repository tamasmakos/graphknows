from src.common.config.settings import AppSettings

def get_app_config() -> AppSettings:
    """
    Load the application configuration using Pydantic Settings.
    """
    return AppSettings()