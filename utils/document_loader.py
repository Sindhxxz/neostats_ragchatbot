import os
from typing import List
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader
)
from langchain_core.documents import Document


def load_document(file_path: str) -> List[Document]:
    """
    Load a document from file path
    
    Args:
        file_path: Path to the document file
    
    Returns:
        List of Document objects
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension == '.txt':
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_extension == '.csv':
            loader = CSVLoader(file_path)
        elif file_extension in ['.doc', '.docx']:
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            # Try text loader as fallback
            loader = TextLoader(file_path, encoding='utf-8')
        
        documents = loader.load()
        return documents
    
    except Exception as e:
        raise RuntimeError(f"Error loading document {file_path}: {str(e)}")


def load_documents_from_directory(directory_path: str) -> List[Document]:
    """
    Load all supported documents from a directory
    
    Args:
        directory_path: Path to the directory containing documents
    
    Returns:
        List of Document objects from all files
    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    supported_extensions = ['.pdf', '.txt', '.csv', '.doc', '.docx']
    all_documents = []
    
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_extension = os.path.splitext(file)[1].lower()
            
            if file_extension in supported_extensions:
                try:
                    documents = load_document(file_path)
                    all_documents.extend(documents)
                except Exception as e:
                    print(f"Warning: Failed to load {file_path}: {str(e)}")
                    continue
    
    return all_documents

