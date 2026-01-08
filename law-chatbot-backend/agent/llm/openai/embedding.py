"""OpenAI Embedding API for Chroma."""
from typing import List, Optional
import os
from agent.logging import logger

# Try to import openai
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Try to import Chroma EmbeddingFunction
try:
    from chromadb import EmbeddingFunction
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    # Fallback for older versions
    EmbeddingFunction = object


class OpenAIEmbeddingFunction(EmbeddingFunction):
    """
    OpenAI Embedding Function for Chroma (compatible with Chroma 0.4.16+).
    
    This class implements the EmbeddingFunction interface required by Chroma 0.4.16+,
    which expects __call__ method with signature (self, input) instead of (self, *args, **kwargs).
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small", base_url: Optional[str] = None):
        """
        Initialize OpenAI Embedding Function.
        
        Args:
            api_key: OpenAI API key. If None, will look for OPENAI_API_KEY env var
            model: Embedding model name (default: "text-embedding-3-small")
            base_url: Optional base URL for API (useful for OpenAI-compatible APIs)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai is not installed. "
                "Install it with: pip install openai"
            )
        
        # Get API key from parameter or environment variable
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "No API key provided. Either:\n"
                "1. Set OPENAI_API_KEY environment variable\n"
                "2. Create .env file with OPENAI_API_KEY=your-key\n"
                "3. Pass api_key parameter"
            )
        
        self.model = model
        
        # Initialize OpenAI client
        client_kwargs = {"api_key": self.api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        else:
            # Check for base_url in environment
            env_base_url = os.getenv("OPENAI_BASE_URL")
            if env_base_url:
                client_kwargs["base_url"] = env_base_url
        
        self.client = OpenAI(**client_kwargs)
        
        logger.info(f"Initialized OpenAI Embedding function with model: {model}")
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Embed a list of texts using OpenAI Embedding API.
        
        This method signature (self, input) is required by Chroma 0.4.16+.
        
        Args:
            input: List of text strings to embed
        
        Returns:
            List of embedding vectors (each is a list of floats)
        """
        try:
            # OpenAI supports batch embedding, so we can send all texts at once
            # Maximum batch size is typically 2048, but we'll be conservative
            batch_size = 100
            all_embeddings = []
            
            # Process in batches
            for i in range(0, len(input), batch_size):
                batch = input[i:i + batch_size]
                
                # Call OpenAI Embedding API for the batch
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                
                # Extract embeddings from response
                # response.data is a list of Embedding objects with 'embedding' attribute
                for embedding_obj in response.data:
                    embedding = embedding_obj.embedding
                    # Ensure it's a list of floats
                    if isinstance(embedding, list):
                        all_embeddings.append(embedding)
                    else:
                        # Convert to list if needed
                        all_embeddings.append(list(embedding))
            
            return all_embeddings
        
        except Exception as e:
            logger.error(f"Error calling OpenAI Embedding API: {str(e)}")
            raise RuntimeError(f"OpenAI Embedding API error: {str(e)}")


def get_openai_embedding_function(api_key: Optional[str] = None, model: str = "text-embedding-3-small", base_url: Optional[str] = None):
    """
    Create an OpenAI Embedding function for Chroma.
    
    Args:
        api_key: OpenAI API key. If None, will look for OPENAI_API_KEY env var
        model: Embedding model name (default: "text-embedding-3-small")
        base_url: Optional base URL for API (useful for OpenAI-compatible APIs)
    
    Returns:
        Embedding function compatible with Chroma 0.4.16+
    
    Raises:
        ImportError: If openai is not installed
        ValueError: If no API key provided
    """
    return OpenAIEmbeddingFunction(api_key=api_key, model=model, base_url=base_url)


def get_openai_query_embedding_function(api_key: Optional[str] = None, model: str = "text-embedding-3-small", base_url: Optional[str] = None):
    """
    Create an OpenAI Embedding function for queries (optimized for search queries).
    
    Note: OpenAI embedding models are typically the same for documents and queries,
    but you can use different models if needed.
    
    Args:
        api_key: OpenAI API key. If None, will look for OPENAI_API_KEY env var
        model: Embedding model name (default: "text-embedding-3-small")
        base_url: Optional base URL for API (useful for OpenAI-compatible APIs)
    
    Returns:
        Embedding function compatible with Chroma 0.4.16+ for queries
    """
    return OpenAIEmbeddingFunction(api_key=api_key, model=model, base_url=base_url)

