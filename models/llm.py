import os
import sys
from typing import Optional
from langchain_groq import ChatGroq

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.config import Config
from utils.logger import default_logger as logger


def get_chatgroq_model(api_key: Optional[str] = None, model: Optional[str] = None):
    """
    Initialize and return the Groq chat model
    
    Args:
        api_key: Groq API key (optional, will use config/env if not provided)
        model: Groq model name (optional, will use config/env if not provided)
    
    Returns:
        ChatGroq model instance
    
    Raises:
        ValueError: If API key or model is not provided and not in config
        RuntimeError: If model initialization fails
    """
    try:
        # Get API key from parameter, config, or environment
        if not api_key:
            api_key = Config.get_groq_api_key()
        
        if not api_key:
            error_msg = (
                "Groq API key is required. "
                "Set GROQ_API_KEY environment variable, "
                "configure in config, or pass api_key parameter."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Get model from parameter, config, or environment
        if not model:
            model = Config.get_groq_model()
        
        if not model:
            error_msg = (
                "Groq model name is required. "
                "Set GROQ_MODEL environment variable, "
                "configure in config, or pass model parameter."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Initializing Groq model: {model}")
        
        # Initialize the Groq chat model
        groq_model = ChatGroq(
            api_key=api_key,
            model=model,
            temperature=0.7  # Default temperature for balanced responses
        )
        
        logger.info("Groq model initialized successfully")
        return groq_model
    
    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
        raise
    except Exception as e:
        error_msg = f"Failed to initialize Groq model: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e