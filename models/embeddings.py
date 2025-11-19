import os
import sys
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.logger import default_logger as logger
from config.config import Config


def get_embedding_model(provider="jina", api_key=None):
    """
    Initialize and return an embedding model (Jina - Free)
    
    Args:
        provider: "jina" (default and only option - free service)
        api_key: Jina API key (optional, will use config/env if not provided)
    
    Returns:
        Embedding model instance
    """
    try:
        provider_lower = provider.lower()
        logger.info(f"Initializing embedding model: {provider_lower}")
        
        if provider_lower == "jina":
            # Get API key from parameter, config, or environment
            if not api_key:
                api_key = Config.get_jina_api_key()
            
            if not api_key:
                error_msg = (
                    "Jina API key is required (free tier available). "
                    "Get your API key from https://jina.ai/embeddings/ "
                    "Set JINA_API_KEY environment variable or pass api_key parameter."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            try:
                # Jina embeddings using langchain-community
                from langchain_community.embeddings import JinaEmbeddings
                
                embedding_model = JinaEmbeddings(
                    jina_api_key=api_key,
                    model_name="jina-embeddings-v2-base-en"  # Free model
                )
                logger.info("Jina embedding model initialized successfully")
                return embedding_model
            except ImportError:
                error_msg = (
                    "Jina embeddings package not found. "
                    "Install with: pip install langchain-community"
                )
                logger.error(error_msg)
                raise ImportError(error_msg)
            except Exception as e:
                logger.error(f"Failed to initialize Jina model: {str(e)}", exc_info=True)
                raise RuntimeError(f"Failed to initialize Jina embedding model: {str(e)}") from e
        else:
            error_msg = f"Unsupported provider: {provider}. Only 'jina' is supported (free service)."
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
        raise
    except ImportError as e:
        logger.error(f"Import error: {str(e)}")
        raise
    except Exception as e:
        error_msg = f"Failed to initialize embedding model: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e

