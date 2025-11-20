import os
from pathlib import Path
from typing import Optional
import streamlit as st

class Config:
    
    # Web Search API Keys
    TAVILY_API_KEY: Optional[str] = st.secrets.get("TAVILY_API_KEY", "")
    
    # LLM API Keys
    GROQ_API_KEY: Optional[str] = st.secrets.get("GROQ_API_KEY", "")
    
    # LLM Model Settings
    GROQ_MODEL: str = st.secrets.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    
    # Embedding API Keys
    JINA_API_KEY: Optional[str] = st.secrets.get("JINA_API_KEY", "")
    
    # Web Search Settings
    DEFAULT_SEARCH_PROVIDER: str = st.secrets.get("DEFAULT_SEARCH_PROVIDER", "tavily")
    MAX_SEARCH_RESULTS: int = int(st.secrets.get("MAX_SEARCH_RESULTS", "5"))

    @classmethod
    def get_groq_api_key(cls) -> Optional[str]:
        return cls.GROQ_API_KEY

    @classmethod
    def get_groq_model(cls) -> str:
        return cls.GROQ_MODEL

    @classmethod
    def get_tavily_api_key(cls) -> Optional[str]:
        return cls.TAVILY_API_KEY

    @classmethod
    def get_jina_api_key(cls) -> Optional[str]:
        return cls.JINA_API_KEY

    @classmethod
    def get_default_search_provider(cls) -> str:
        return cls.DEFAULT_SEARCH_PROVIDER


