from typing import List
from langchain_community.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def count_words(text: str) -> int:
    """Count the number of words in a text string"""
    return len(text.split())


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    chunk_by_words: bool = False
) -> List[Document]:
    """
    Split documents into smaller chunks for embedding
    
    Args:
        documents: List of Document objects to split
        chunk_size: Maximum size of each chunk (default: 1000)
                   If chunk_by_words=True, this is the number of words
                   If chunk_by_words=False, this is the number of characters
        chunk_overlap: Overlap between chunks (default: 200)
                      If chunk_by_words=True, this is the number of words
                      If chunk_by_words=False, this is the number of characters
        chunk_by_words: If True, chunk by word count; if False, chunk by character count (default: False)
    
    Returns:
        List of split Document objects
    """
    if chunk_by_words:
        # Use word count for chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=count_words,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    else:
        # Use character count for chunking (default)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    split_docs = text_splitter.split_documents(documents)
    return split_docs

