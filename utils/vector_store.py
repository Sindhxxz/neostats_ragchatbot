import os
import sys
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
import chromadb

# Add parent directory to path for logger import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.logger import default_logger as logger


def create_vector_store(
    documents: List[Document],
    embedding_model: Embeddings,
    persist_directory: str = "./chroma_db",
    collection_name: str = "documents"
) -> Chroma:
    """
    Create a vector store from documents using ChromaDB
    
    Args:
        documents: List of Document objects to embed and store
        embedding_model: Embedding model instance
        persist_directory: Directory to persist the vector store
        collection_name: Name of the collection in ChromaDB
    
    Returns:
        Chroma vector store instance
    """
    try:
        if not documents:
            raise ValueError("Cannot create vector store: documents list is empty")
        
        logger.info(f"Creating vector store with {len(documents)} documents in {persist_directory}")
        
        # Create vector store with persistence
        # Note: Chroma 0.4.x+ automatically persists, so persist() is not needed
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        
        logger.info(f"Vector store created successfully with {len(documents)} documents")
        return vector_store
    
    except ValueError as e:
        logger.error(f"Validation error creating vector store: {str(e)}")
        raise
    except Exception as e:
        error_msg = f"Error creating vector store: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e


def get_embedding_dimension(embedding_model: Embeddings) -> int:
    """
    Get the dimension of embeddings produced by the model
    
    Args:
        embedding_model: Embedding model instance
    
    Returns:
        Embedding dimension (number of dimensions)
    """
    try:
        # Test with a small text to get embedding dimension
        test_embedding = embedding_model.embed_query("test")
        return len(test_embedding)
    except Exception as e:
        logger.error(f"Error getting embedding dimension: {str(e)}", exc_info=True)
        raise RuntimeError(f"Failed to get embedding dimension: {str(e)}") from e


def get_collection_embedding_dimension(
    persist_directory: str = "./chroma_db",
    collection_name: str = "documents"
) -> Optional[int]:
    """
    Get the embedding dimension of an existing ChromaDB collection
    
    Args:
        persist_directory: Directory where the vector store is persisted
        collection_name: Name of the collection in ChromaDB
    
    Returns:
        Embedding dimension or None if collection doesn't exist
    """
    try:
        if not os.path.exists(persist_directory):
            return None
        
        # Connect to ChromaDB to check collection
        client = chromadb.PersistentClient(path=persist_directory)
        
        # Check if collection exists by trying to get it
        try:
            collection = client.get_collection(name=collection_name)
        except Exception:
            # Collection doesn't exist
            return None
        
        # Try to get dimension from a sample embedding
        try:
            count = collection.count()
            if count > 0:
                # Peek at the first item to get embedding dimension
                results = collection.peek(limit=1)
                if results and 'embeddings' in results and len(results['embeddings']) > 0:
                    embedding = results['embeddings'][0]
                    if isinstance(embedding, list) and len(embedding) > 0:
                        return len(embedding)
        except Exception as e:
            logger.debug(f"Could not get dimension from collection peek: {str(e)}")
        
        return None
    except Exception as e:
        logger.debug(f"Could not get collection dimension: {str(e)}")
        return None


def migrate_collection_embeddings(
    embedding_model: Embeddings,
    persist_directory: str = "./chroma_db",
    collection_name: str = "documents"
) -> Chroma:
    """
    Migrate an existing collection to use a new embedding model by re-embedding all documents.
    This function reads all documents from the existing collection, re-embeds them with the new model,
    and replaces the collection.
    
    Args:
        embedding_model: New embedding model instance to use
        persist_directory: Directory where the vector store is persisted
        collection_name: Name of the collection in ChromaDB
    
    Returns:
        New Chroma vector store instance with migrated embeddings
    
    Raises:
        RuntimeError: If migration fails
        ValueError: If collection doesn't exist or is empty
    """
    try:
        if not os.path.exists(persist_directory):
            raise ValueError(f"Vector store directory does not exist: {persist_directory}")
        
        logger.info(f"Starting migration of collection '{collection_name}' to new embedding dimensions")
        
        # Connect to ChromaDB
        client = chromadb.PersistentClient(path=persist_directory)
        
        # Get the old collection
        try:
            old_collection = client.get_collection(name=collection_name)
        except Exception as e:
            raise ValueError(f"Collection '{collection_name}' does not exist: {str(e)}")
        
        # Get all documents from the old collection
        count = old_collection.count()
        if count == 0:
            raise ValueError(f"Collection '{collection_name}' is empty, nothing to migrate")
        
        logger.info(f"Found {count} documents to migrate")
        
        # Get all data from the collection
        results = old_collection.get()
        
        if not results or 'ids' not in results or len(results['ids']) == 0:
            raise ValueError("No documents found in collection")
        
        # Extract text content and metadata
        texts = results.get('documents', [])
        metadatas = results.get('metadatas', [])
        ids = results.get('ids', [])
        
        if not texts:
            raise ValueError("No text content found in collection documents")
        
        logger.info(f"Extracted {len(texts)} documents from old collection")
        
        # Convert to LangChain Document format
        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            doc = Document(
                page_content=text,
                metadata=metadata
            )
            documents.append(doc)
        
        # Delete the old collection
        logger.info("Deleting old collection...")
        try:
            client.delete_collection(name=collection_name)
            logger.info("Old collection deleted successfully")
        except Exception as e:
            logger.warning(f"Could not delete old collection (may not exist): {str(e)}")
        
        # Create new collection with new embeddings
        logger.info(f"Creating new collection with {len(documents)} documents using new embedding model...")
        new_vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        
        logger.info(f"✅ Successfully migrated {len(documents)} documents to new embedding dimensions")
        return new_vector_store
    
    except ValueError as e:
        logger.error(f"Validation error during migration: {str(e)}")
        raise
    except Exception as e:
        error_msg = f"Error migrating collection embeddings: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e


def load_vector_store(
    embedding_model: Embeddings,
    persist_directory: str = "./chroma_db",
    collection_name: str = "documents"
) -> Optional[Chroma]:
    """
    Load an existing vector store from disk with dimension validation
    
    Args:
        embedding_model: Embedding model instance
        persist_directory: Directory where the vector store is persisted
        collection_name: Name of the collection in ChromaDB
    
    Returns:
        Chroma vector store instance or None if not found or dimension mismatch
    """
    try:
        if not os.path.exists(persist_directory):
            logger.debug(f"Vector store directory does not exist: {persist_directory}")
            return None
        
        # Check if collection exists and get its dimension
        collection_dimension = get_collection_embedding_dimension(
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        
        if collection_dimension is not None:
            # Get current embedding model dimension
            model_dimension = get_embedding_dimension(embedding_model)
            
            if collection_dimension != model_dimension:
                error_msg = (
                    f"Embedding dimension mismatch detected! "
                    f"Existing collection uses {collection_dimension}-dimensional embeddings, "
                    f"but current model produces {model_dimension}-dimensional embeddings. "
                    f"Use migrate_collection_embeddings() to automatically convert the collection."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        logger.info(f"Loading vector store from {persist_directory}")
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model,
            collection_name=collection_name
        )
        
        logger.info("Vector store loaded successfully")
        return vector_store
    
    except ValueError as e:
        # Re-raise dimension mismatch errors
        raise
    except Exception as e:
        logger.warning(f"Could not load vector store: {str(e)}", exc_info=True)
        return None


def add_documents_to_vector_store(
    vector_store: Chroma,
    documents: List[Document]
) -> None:
    """
    Add new documents to an existing vector store
    
    Args:
        vector_store: Existing Chroma vector store
        documents: List of Document objects to add
    
    Raises:
        RuntimeError: If adding documents fails, including dimension mismatch errors
    """
    try:
        if not documents:
            raise ValueError("Cannot add documents: documents list is empty")
        
        logger.info(f"Adding {len(documents)} documents to vector store")
        vector_store.add_documents(documents)
        # Note: Chroma 0.4.x+ automatically persists, so persist() is not needed
        logger.info("Documents added to vector store successfully")
    
    except ValueError as e:
        logger.error(f"Validation error adding documents: {str(e)}")
        raise
    except Exception as e:
        error_msg = str(e)
        # Check if it's a dimension mismatch error
        if "dimension" in error_msg.lower() or "embedding" in error_msg.lower():
            error_msg = (
                f"Embedding dimension mismatch detected. "
                f"The vector store was created with a different embedding model than the one currently being used. "
                f"Please clear the knowledge base and recreate it with the current embedding model. "
                f"Original error: {error_msg}"
            )
        else:
            error_msg = f"Error adding documents to vector store: {error_msg}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e


def retrieve_relevant_chunks(
    vector_store: Chroma,
    query: str,
    k: int = 4
) -> List[Document]:
    """
    Retrieve relevant document chunks for a query
    
    Args:
        vector_store: Chroma vector store instance
        query: Query string
        k: Number of chunks to retrieve (default: 4)
    
    Returns:
        List of relevant Document objects
    
    Raises:
        RuntimeError: If retrieval fails, including dimension mismatch errors
    """
    try:
        if not query or not query.strip():
            logger.warning("Empty query provided for retrieval")
            return []
        
        logger.debug(f"Retrieving {k} relevant chunks for query: {query[:50]}...")
        # Use similarity search to find relevant chunks
        relevant_docs = vector_store.similarity_search(query, k=k)
        logger.debug(f"Retrieved {len(relevant_docs)} relevant chunks")
        return relevant_docs
    
    except Exception as e:
        error_msg = str(e)
        # Check if it's a dimension mismatch error
        if "dimension" in error_msg.lower() or "embedding" in error_msg.lower():
            error_msg = (
                f"Embedding dimension mismatch detected during retrieval. "
                f"This usually means the vector store was created with a different embedding model. "
                f"Please clear the knowledge base and recreate it. Original error: {error_msg}"
            )
        else:
            error_msg = f"Error retrieving relevant chunks: {error_msg}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e


def retrieve_relevant_chunks_with_scores(
    vector_store: Chroma,
    query: str,
    k: int = 4
) -> List[tuple]:
    """
    Retrieve relevant document chunks with similarity scores
    
    Args:
        vector_store: Chroma vector store instance
        query: Query string
        k: Number of chunks to retrieve (default: 4)
    
    Returns:
        List of tuples (Document, score)
    """
    try:
        # Use similarity search with scores
        relevant_docs = vector_store.similarity_search_with_score(query, k=k)
        return relevant_docs
    
    except Exception as e:
        raise RuntimeError(f"Error retrieving relevant chunks with scores: {str(e)}")

