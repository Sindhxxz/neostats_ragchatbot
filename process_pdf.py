"""
Script to process a PDF file and add it to the knowledge base with word-based chunking
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models.embeddings import get_embedding_model
from utils.document_loader import load_document
from utils.text_splitter import split_documents
from utils.vector_store import create_vector_store, load_vector_store, add_documents_to_vector_store


def process_pdf_to_knowledge_base(
    pdf_path: str,
    chunk_size_words: int = 200,
    chunk_overlap_words: int = 40,
    embedding_provider: str = "jina",
    embedding_api_key: str = None
):
    """
    Process a PDF file and add it to the knowledge base
    
    Args:
        pdf_path: Path to the PDF file
        chunk_size_words: Number of words per chunk (default: 200)
        chunk_overlap_words: Number of words overlap between chunks (default: 40)
        embedding_provider: "jina" (default - free service)
        embedding_api_key: Jina API key (required - free tier available)
    """
    print(f"Processing PDF: {pdf_path}")
    
    # Check if file exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Load the PDF document
    print("Loading PDF document...")
    documents = load_document(pdf_path)
    print(f"Loaded {len(documents)} pages from PDF")
    
    # Split documents into chunks (by words)
    print(f"Splitting documents into chunks of {chunk_size_words} words...")
    split_docs = split_documents(
        documents,
        chunk_size=chunk_size_words,
        chunk_overlap=chunk_overlap_words,
        chunk_by_words=True
    )
    print(f"Created {len(split_docs)} chunks")
    
    # Initialize embedding model (Jina - Free)
    print(f"Initializing embedding model ({embedding_provider})...")
    if not embedding_api_key:
        embedding_api_key = os.getenv("JINA_API_KEY", "")
    if not embedding_api_key:
        raise ValueError("Jina API key is required. Set JINA_API_KEY environment variable or pass embedding_api_key parameter.")
    embedding_model = get_embedding_model(embedding_provider, embedding_api_key)
    print("Embedding model initialized")
    
    # Create or load vector store
    persist_directory = "./chroma_db"
    collection_name = "documents"
    
    print("Checking for existing vector store...")
    vector_store = load_vector_store(
        embedding_model,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    
    if vector_store is None:
        print("Creating new vector store...")
        vector_store = create_vector_store(
            split_docs,
            embedding_model,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        print(f"✅ Created new knowledge base with {len(split_docs)} chunks")
    else:
        print("Adding documents to existing vector store...")
        add_documents_to_vector_store(vector_store, split_docs)
        print(f"✅ Added {len(split_docs)} chunks to existing knowledge base")
    
    print("\n" + "="*50)
    print("PDF processing completed successfully!")
    print(f"Total chunks in knowledge base: {len(split_docs)}")
    print(f"Vector store location: {persist_directory}")
    print("="*50)


if __name__ == "__main__":
    # PDF file path - relative to workspace root
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    pdf_path = os.path.join(workspace_root, "HR-POLICIES-1-1-1.pdf")
    
    # Configuration
    chunk_size_words = 200
    chunk_overlap_words = 40  # 20% overlap (40 words out of 200)
    embedding_provider = "jina"  # Jina (free service)
    embedding_api_key = os.getenv("JINA_API_KEY", None)  # Set JINA_API_KEY environment variable or pass here
    
    try:
        process_pdf_to_knowledge_base(
            pdf_path=pdf_path,
            chunk_size_words=chunk_size_words,
            chunk_overlap_words=chunk_overlap_words,
            embedding_provider=embedding_provider,
            embedding_api_key=embedding_api_key
        )
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        import traceback
        traceback.print_exc()

