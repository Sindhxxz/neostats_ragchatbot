import os
import sys
from typing import List, Optional, Tuple
from langchain_core.documents import Document

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.document_loader import load_document, load_documents_from_directory
from utils.text_splitter import split_documents
from utils.logger import default_logger as logger


def process_documents_for_vector_store(
    file_path: Optional[str] = None,
    directory_path: Optional[str] = None,
    chunk_size: int = 200,
    chunk_overlap: int = 40,
    chunk_by_words: bool = True
) -> Tuple[List[Document], List[Document]]:
    """
    Process documents from file or directory and prepare them for vector store
    
    Args:
        file_path: Path to a single document file
        directory_path: Path to directory containing documents
        chunk_size: Size of chunks (words if chunk_by_words=True, characters otherwise)
        chunk_overlap: Overlap between chunks
        chunk_by_words: Whether to chunk by words (True) or characters (False)
    
    Returns:
        Tuple of (original_documents, split_documents)
    
    Raises:
        ValueError: If neither file_path nor directory_path is provided
        FileNotFoundError: If file or directory doesn't exist
        RuntimeError: If document processing fails
    """
    try:
        if not file_path and not directory_path:
            raise ValueError("Either file_path or directory_path must be provided")
        
        # Load documents
        if file_path:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            logger.info(f"Loading document from file: {file_path}")
            documents = load_document(file_path)
            logger.info(f"Loaded {len(documents)} pages from {file_path}")
        else:
            if not os.path.exists(directory_path):
                raise FileNotFoundError(f"Directory not found: {directory_path}")
            logger.info(f"Loading documents from directory: {directory_path}")
            documents = load_documents_from_directory(directory_path)
            logger.info(f"Loaded {len(documents)} documents from directory")
        
        if not documents:
            logger.warning("No documents were loaded")
            return [], []
        
        # Split documents into chunks
        logger.info(f"Splitting documents into chunks (size={chunk_size}, overlap={chunk_overlap}, by_words={chunk_by_words})")
        split_docs = split_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_by_words=chunk_by_words
        )
        logger.info(f"Created {len(split_docs)} chunks from {len(documents)} documents")
        
        return documents, split_docs
    
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Document processing error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing documents: {str(e)}", exc_info=True)
        raise RuntimeError(f"Error processing documents: {str(e)}") from e


def process_uploaded_file(
    uploaded_file,
    temp_dir: str = "./temp_docs",
    chunk_size: int = 200,
    chunk_overlap: int = 40,
    chunk_by_words: bool = True
) -> Tuple[List[Document], str]:
    """
    Process an uploaded file and return split documents
    
    Args:
        uploaded_file: Streamlit uploaded file object
        temp_dir: Temporary directory to save uploaded file
        chunk_size: Size of chunks
        chunk_overlap: Overlap between chunks
        chunk_by_words: Whether to chunk by words
    
    Returns:
        Tuple of (split_documents, temp_file_path)
    
    Raises:
        RuntimeError: If file processing fails
    """
    try:
        # Create temp directory if it doesn't exist
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save uploaded file temporarily
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        logger.info(f"Saving uploaded file to: {temp_path}")
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process the document
        _, split_docs = process_documents_for_vector_store(
            file_path=temp_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_by_words=chunk_by_words
        )
        
        return split_docs, temp_path
    
    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}", exc_info=True)
        raise RuntimeError(f"Error processing uploaded file: {str(e)}") from e

