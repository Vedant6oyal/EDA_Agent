import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for the InsightBot application."""
    
    # Application settings
    APP_NAME = "InsightBot"
    VERSION = "0.1.0"
    DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
    
    # LLM Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))
    
    # Data settings
    DATA_DIR = os.getenv("DATA_DIR", "data")
    DEFAULT_DATASET = os.getenv("DEFAULT_DATASET", "snapshots_2000.csv")
    
    # Memory settings
    MEMORY_WINDOW_SIZE = int(os.getenv("MEMORY_WINDOW_SIZE", "5"))
    
    # Logging configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        return True
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Return configuration as a dictionary, excluding sensitive information."""
        return {
            "app_name": cls.APP_NAME,
            "version": cls.VERSION,
            "debug": cls.DEBUG,
            "default_model": cls.DEFAULT_MODEL,
            "temperature": cls.TEMPERATURE,
            "data_dir": cls.DATA_DIR,
            "default_dataset": cls.DEFAULT_DATASET,
            "memory_window_size": cls.MEMORY_WINDOW_SIZE,
            "log_level": cls.LOG_LEVEL
        }

# Validate configuration on import
Config.validate()
