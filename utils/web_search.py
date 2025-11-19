import os
import sys
from typing import List, Dict, Optional

# Add parent directory to path for config import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.config import Config
from utils.logger import default_logger as logger


def search_duckduckgo(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search using DuckDuckGo (free, no API key required)
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
    
    Returns:
        List of dictionaries with 'title', 'snippet', and 'link' keys
    """
    try:
        from duckduckgo_search import DDGS
        
        results = []
        with DDGS() as ddgs:
            search_results = ddgs.text(query, max_results=max_results)
            for result in search_results:
                results.append({
                    'title': result.get('title', ''),
                    'snippet': result.get('body', ''),
                    'link': result.get('href', '')
                })
        return results
    except ImportError:
        raise ImportError("duckduckgo-search package is required. Install with: pip install duckduckgo-search")
    except Exception as e:
        raise RuntimeError(f"DuckDuckGo search error: {str(e)}")


def search_tavily(query: str, max_results: int = 5, api_key: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Search using Tavily API (optimized for RAG applications)
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        api_key: Tavily API key (optional, will use config if not provided)
    
    Returns:
        List of dictionaries with 'title', 'snippet', and 'link' keys
    """
    try:
        from tavily import TavilyClient
        
        if not api_key:
            api_key = Config.get_tavily_api_key()
        
        if not api_key:
            raise ValueError("Tavily API key is required. Set TAVILY_API_KEY environment variable or pass api_key parameter.")
        
        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, max_results=max_results, search_depth="basic")
        
        results = []
        for result in response.get('results', []):
            results.append({
                'title': result.get('title', ''),
                'snippet': result.get('content', ''),
                'link': result.get('url', '')
            })
        return results
    except ImportError:
        raise ImportError("tavily-python package is required. Install with: pip install tavily-python")
    except Exception as e:
        raise RuntimeError(f"Tavily search error: {str(e)}")


def search_serpapi(query: str, max_results: int = 5, api_key: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Search using SerpAPI (Google search results)
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        api_key: SerpAPI key (optional, will use config if not provided)
    
    Returns:
        List of dictionaries with 'title', 'snippet', and 'link' keys
    """
    try:
        from serpapi import GoogleSearch
        
        if not api_key:
            api_key = Config.get_serpapi_api_key()
        
        if not api_key:
            raise ValueError("SerpAPI key is required. Set SERPAPI_API_KEY environment variable or pass api_key parameter.")
        
        params = {
            "q": query,
            "api_key": api_key,
            "num": max_results
        }
        
        search = GoogleSearch(params)
        results_data = search.get_dict()
        
        results = []
        organic_results = results_data.get('organic_results', [])
        for result in organic_results[:max_results]:
            results.append({
                'title': result.get('title', ''),
                'snippet': result.get('snippet', ''),
                'link': result.get('link', '')
            })
        return results
    except ImportError:
        raise ImportError("google-search-results package is required. Install with: pip install google-search-results")
    except Exception as e:
        raise RuntimeError(f"SerpAPI search error: {str(e)}")


def perform_web_search(
    query: str,
    provider: Optional[str] = None,
    max_results: int = 5,
    api_keys: Optional[Dict[str, str]] = None
) -> List[Dict[str, str]]:
    """
    Perform web search using the specified provider
    
    Args:
        query: Search query
        provider: Search provider ("duckduckgo", "tavily", "serpapi"). 
                 If None, uses default from config
        max_results: Maximum number of results to return
        api_keys: Dictionary with API keys (e.g., {"tavily": "key", "serpapi": "key"})
    
    Returns:
        List of dictionaries with 'title', 'snippet', and 'link' keys
    """
    if not provider:
        provider = Config.get_default_search_provider()
    
    provider = provider.lower()
    
    # Override API keys if provided
    if api_keys:
        if provider == "tavily" and "tavily" in api_keys:
            Config.set_tavily_api_key(api_keys["tavily"])
    
    try:
        if provider == "tavily":
            tavily_key = api_keys.get("tavily") if api_keys else None
            return search_tavily(query, max_results, tavily_key)
        else:
            raise ValueError(f"Unsupported search provider: {provider}. Only 'tavily' is supported (free service).")
    except Exception as e:
        error_msg = f"Tavily search failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e


def format_search_results(results: List[Dict[str, str]]) -> str:
    """
    Format search results as a string for LLM context
    
    Args:
        results: List of search result dictionaries
    
    Returns:
        Formatted string with search results
    """
    if not results:
        return "No search results found."
    
    formatted = "Web Search Results:\n\n"
    for i, result in enumerate(results, 1):
        formatted += f"[{i}] {result.get('title', 'No title')}\n"
        formatted += f"    {result.get('snippet', 'No snippet')}\n"
        formatted += f"    Source: {result.get('link', 'No link')}\n\n"
    
    return formatted


def should_use_web_search(
    query: str,
    vector_store=None,
    rag_enabled: bool = True
) -> bool:
    """
    Determine if web search should be used based on query and available knowledge
    
    Args:
        query: User query
        vector_store: Vector store instance (optional)
        rag_enabled: Whether RAG is enabled
    
    Returns:
        True if web search should be used, False otherwise
    """
    # Keywords that suggest need for real-time information
    real_time_keywords = [
        "current", "latest", "recent", "today", "now", "2024", "2025",
        "news", "update", "happening", "trending", "stock", "price",
        "weather", "forecast", "live", "real-time"
    ]
    
    query_lower = query.lower()
    
    # Check if query contains real-time keywords
    has_real_time_keyword = any(keyword in query_lower for keyword in real_time_keywords)
    
    # If RAG is enabled and vector store exists, check if relevant chunks exist
    if rag_enabled and vector_store is not None:
        try:
            # Import here to avoid circular dependencies
            import sys
            import os
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
            from utils.vector_store import retrieve_relevant_chunks
            relevant_chunks = retrieve_relevant_chunks(vector_store, query, k=1)
            has_relevant_docs = len(relevant_chunks) > 0 and len(relevant_chunks[0].page_content) > 50
        except Exception:
            has_relevant_docs = False
    else:
        has_relevant_docs = False
    
    # Use web search if:
    # 1. Query has real-time keywords, OR
    # 2. No relevant documents found in knowledge base
    return has_real_time_keyword or not has_relevant_docs

