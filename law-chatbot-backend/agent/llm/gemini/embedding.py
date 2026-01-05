"""Google Embedding API for Chroma."""
from typing import List, Optional
import os
from agent.logging import logger

# Try to import google-generativeai
try:
    import google.generativeai as genai
    GOOGLE_GENERATIVEAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENERATIVEAI_AVAILABLE = False

# Try to import Chroma EmbeddingFunction
try:
    from chromadb import EmbeddingFunction
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    # Fallback for older versions
    EmbeddingFunction = object


class GoogleEmbeddingFunction(EmbeddingFunction):
    """
    Google Embedding Function for Chroma (compatible with Chroma 0.4.16+).
    
    This class implements the EmbeddingFunction interface required by Chroma 0.4.16+,
    which expects __call__ method with signature (self, input) instead of (self, *args, **kwargs).
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "models/embedding-001", task_type: str = "retrieval_document"):
        """
        Initialize Google Embedding Function.
        
        Args:
            api_key: Google API key. If None, will look for GEMINI_API_KEY env var
            model: Embedding model name (default: "models/embedding-001")
            task_type: Task type for embedding ("retrieval_document" or "retrieval_query")
        """
        if not GOOGLE_GENERATIVEAI_AVAILABLE:
            raise ImportError(
                "google-generativeai is not installed. "
                "Install it with: pip install google-generativeai"
            )
        
        # Get API key from parameter or environment variable
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "No API key provided. Either:\n"
                "1. Set GEMINI_API_KEY environment variable\n"
                "2. Create .env file with GEMINI_API_KEY=your-key\n"
                "3. Pass api_key parameter"
            )
        
        self.model = model
        self.task_type = task_type
        
        # Configure the API
        genai.configure(api_key=self.api_key)
        
        logger.info(f"Initialized Google Embedding function with model: {model}, task_type: {task_type}")
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Embed a list of texts using Google Embedding API.
        
        This method signature (self, input) is required by Chroma 0.4.16+.
        
        Args:
            input: List of text strings to embed
        
        Returns:
            List of embedding vectors (each is a list of floats)
        """
        try:
            # Process each text individually or in batch
            all_embeddings = []
            
            for text in input:
                # Call Google Embedding API for each text
                result = genai.embed_content(
                    model=self.model,
                    content=text,
                    task_type=self.task_type
                )
                
                # Extract embedding - result is a dict with 'embedding' key
                embedding = result['embedding']
                
                # Ensure it's a list of floats
                if isinstance(embedding, list):
                    all_embeddings.append(embedding)
                else:
                    # Convert to list if needed
                    all_embeddings.append(list(embedding))
            
            return all_embeddings
        
        except Exception as e:
            logger.error(f"Error calling Google Embedding API: {str(e)}")
            raise RuntimeError(f"Google Embedding API error: {str(e)}")


def get_google_embedding_function(api_key: Optional[str] = None, model: str = "models/embedding-001"):
    """
    Create a Google Embedding function for Chroma.
    
    Args:
        api_key: Google API key. If None, will look for GEMINI_API_KEY env var
        model: Embedding model name (default: "models/embedding-001")
    
    Returns:
        Embedding function compatible with Chroma 0.4.16+
    
    Raises:
        ImportError: If google-generativeai is not installed
        ValueError: If no API key provided
    """
    return GoogleEmbeddingFunction(api_key=api_key, model=model, task_type="retrieval_document")


def get_google_query_embedding_function(api_key: Optional[str] = None, model: str = "models/embedding-001"):
    """
    Create a Google Embedding function for queries (optimized for search queries).
    
    Args:
        api_key: Google API key. If None, will look for GEMINI_API_KEY env var
        model: Embedding model name (default: "models/embedding-001")
    
    Returns:
        Embedding function compatible with Chroma 0.4.16+ for queries
    """
    return GoogleEmbeddingFunction(api_key=api_key, model=model, task_type="retrieval_query")

