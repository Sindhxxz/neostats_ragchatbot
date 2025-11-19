"""
Configuration file for API keys and settings
"""
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Attempt to load environment variables from a .env file if present.
# We look for a .env alongside the project root (two directories above this file)
# and fall back to python-dotenv's default discovery if not found.
_default_env_path = Path(__file__).resolve().parents[1] / ".env"
if not load_dotenv(dotenv_path=_default_env_path, override=False):
    # Fall back to the parent of the project root (workspace root).
    _workspace_env_path = Path(__file__).resolve().parents[2] / ".env"
    if not load_dotenv(dotenv_path=_workspace_env_path, override=False):
        load_dotenv(override=False)


class Config:
    """Configuration class for managing API keys and settings"""
    
    # Web Search API Keys (Free Services Only)
    TAVILY_API_KEY: Optional[str] = os.getenv("TAVILY_API_KEY", "")
    
    # LLM API Keys (Free Services Only)
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY", "")
    
    # LLM Model Settings
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    
    # Embedding API Keys (Free Services Only)
    JINA_API_KEY: Optional[str] = os.getenv("JINA_API_KEY", "")
    
    # Web Search Settings
    DEFAULT_SEARCH_PROVIDER: str = os.getenv("DEFAULT_SEARCH_PROVIDER", "tavily")  # Only Tavily (free)
    MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
    
    @classmethod
    def get_groq_api_key(cls) -> Optional[str]:
        """Get Groq API key"""
        return cls.GROQ_API_KEY
    
    @classmethod
    def get_groq_model(cls) -> str:
        """Get Groq model name"""
        return cls.GROQ_MODEL
    
    @classmethod
    def set_groq_api_key(cls, api_key: str):
        """Set Groq API key"""
        cls.GROQ_API_KEY = api_key
        os.environ["GROQ_API_KEY"] = api_key
    
    @classmethod
    def set_groq_model(cls, model: str):
        """Set Groq model"""
        cls.GROQ_MODEL = model
        os.environ["GROQ_MODEL"] = model
    
    @classmethod
    def get_tavily_api_key(cls) -> Optional[str]:
        """Get Tavily API key"""
        return cls.TAVILY_API_KEY
    
    @classmethod
    def get_jina_api_key(cls) -> Optional[str]:
        """Get Jina API key"""
        return cls.JINA_API_KEY
    
    @classmethod
    def set_tavily_api_key(cls, api_key: str):
        """Set Tavily API key"""
        cls.TAVILY_API_KEY = api_key
        os.environ["TAVILY_API_KEY"] = api_key
    
    @classmethod
    def set_jina_api_key(cls, api_key: str):
        """Set Jina API key"""
        cls.JINA_API_KEY = api_key
        os.environ["JINA_API_KEY"] = api_key
    
    @classmethod
    def get_default_search_provider(cls) -> str:
        """Get default search provider"""
        return cls.DEFAULT_SEARCH_PROVIDER
    
    @classmethod
    def set_default_search_provider(cls, provider: str):
        """Set default search provider"""
        cls.DEFAULT_SEARCH_PROVIDER = provider
        os.environ["DEFAULT_SEARCH_PROVIDER"] = provider

