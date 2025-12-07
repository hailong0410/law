# # apps/backend/app/services/vector_store.py
# from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
# from langchain_core.documents import Document
# from typing import List

# # In a production app, use a persistent vector store like ChromaDB or a managed service.
# # For this example, we use an in-memory FAISS store. It will reset on server restart.
# embeddings = OpenAIEmbeddings()

# # A simple in-memory store. You'll need a more robust, persistent solution for production.
# # We initialize it with a dummy document to create the index.
# _vector_store = FAISS.from_texts(
#     ["This is the initial content."], 
#     embedding=embeddings
# )

# def get_retriever(search_kwargs={"k": 3}):
#     """Returns a retriever for the global vector store."""
#     return _vector_store.as_retriever(search_kwargs=search_kwargs)

# def add_documents_to_store(documents: List[Document]):
#     """Adds a list of LangChain documents to the vector store."""
#     global _vector_store
#     if documents:
#         _vector_store.add_documents(documents)



# apps/backend/app/services/vector_store.py
import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List
from app.core.config import settings
import os 
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
# Choose embedding model: "local" (sentence-transformers) or "google" (Google API)
# Local model doesn't require API key and has no quota limits
# Note: Changing this will create a new collection. Old data won't be accessible.
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").lower()

# Define the path for the persistent ChromaDB database and the collection name
CHROMA_PERSIST_DIRECTORY = "chroma_db"
# Use different collection names for different embedding providers to avoid conflicts
CHROMA_COLLECTION_NAME = f"multimodal_rag_collection_{EMBEDDING_PROVIDER}"

# --- Initialization ---
# Initialize the embedding model based on provider choice
if EMBEDDING_PROVIDER == "google":
    # Use Google Generative AI embeddings (requires API key, has quota limits)
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from google import genai
    
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=GOOGLE_API_KEY)
    
    embedding_function = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        task_type="retrieval_document"
    )
    print("Using Google Generative AI embeddings (embedding-001)")
else:
    # Use local sentence-transformers model (no API key needed, no quota limits)
    from langchain_huggingface import HuggingFaceEmbeddings
    
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("Using local HuggingFace embeddings (all-MiniLM-L6-v2) - No API key required")

# Initialize a persistent ChromaDB client
persistent_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)

# Initialize the Chroma vector store with the client, collection name, and embedding function
vector_store = Chroma(
    client=persistent_client,
    collection_name=CHROMA_COLLECTION_NAME,
    embedding_function=embedding_function,
)

# --- Service Functions ---
def add_documents_to_store(documents: List[Document],session_id: str):
    """
    Adds a list of LangChain documents to the persistent Chroma vector store.
    The documents will be embedded using the configured embedding function.
    """
    if not documents:
        return
    
    # --- CORE CHANGE: Add metadata to each document chunk ---
    for doc in documents:
        doc.metadata = {"session_id": session_id, **doc.metadata}
    # --------------------------------------------------------

    print(f"Adding {len(documents)} document chunks with session_id '{session_id}' to ChromaDB.")
    vector_store.add_documents(documents)
    print("Successfully added documents to the store.")


def get_retriever(session_id: str, search_kwargs={"k": 3}):
    """
    Returns a retriever for the Chroma vector store that is filtered
    to only search documents matching the given session_id.
    """
    print(f"Creating retriever for session_id: {session_id}")
    # --- CORE CHANGE: Use the 'filter' argument in as_retriever ---
    return vector_store.as_retriever(
        search_kwargs={
            "k": search_kwargs.get("k", 3),
            "filter": {"session_id": session_id}
        }
    )


def get_all_documents_for_session(session_id: str, max_docs: int = 20) -> List[Document]:
    """
    Retrieves all documents for a given session by querying ChromaDB directly.
    This is useful when you need all documents rather than just the top-k similar ones.
    
    Args:
        session_id: The session ID to filter documents by
        max_docs: Maximum number of documents to retrieve
        
    Returns:
        List of Document objects matching the session_id
    """
    try:
        # Query ChromaDB collection directly with filter
        collection = persistent_client.get_collection(name=CHROMA_COLLECTION_NAME)
        
        # Get all results with session_id filter
        results = collection.get(
            where={"session_id": session_id},
            limit=max_docs
        )
        
        # Convert ChromaDB results to LangChain Documents
        documents = []
        if results['ids']:
            for i, doc_id in enumerate(results['ids']):
                content = results['documents'][i] if results['documents'] else ""
                metadata = results['metadatas'][i] if results['metadatas'] else {}
                
                documents.append(Document(
                    page_content=content,
                    metadata=metadata
                ))
        
        return documents
    except Exception as e:
        print(f"Error retrieving all documents for session {session_id}: {e}")
        return []