"""VNPT Embedding API for Chroma."""
from typing import List, Optional
import os
from agent.logging import logger

# Auto-load .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use system environment variables

# Try to import requests for VNPT API
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Try to import Chroma EmbeddingFunction
try:
    from chromadb import EmbeddingFunction
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    # Fallback for older versions
    EmbeddingFunction = object


class VNPTEmbeddingFunction(EmbeddingFunction):
    """
    VNPT Embedding Function for Chroma (compatible with Chroma 0.4.16+).
    
    This class implements the EmbeddingFunction interface required by Chroma 0.4.16+,
    which expects __call__ method with signature (self, input) instead of (self, *args, **kwargs).
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        token_id: Optional[str] = None,
        token_key: Optional[str] = None,
        model: str = "vnptai_hackathon_embedding",
        base_url: str = "https://api.idg.vnpt.vn/data-service"
    ):
        """
        Initialize VNPT Embedding Function.
        
        Args:
            api_key: Authorization Bearer token (or use VNPT_AUTHORIZATION env var)
            token_id: Token ID (or use VNPT_TOKEN_ID env var)
            token_key: Token Key (or use VNPT_TOKEN_KEY env var)
            model: Embedding model name (default: "vnptai_hackathon_embedding")
            base_url: Base URL for VNPT API
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError(
                "requests is not installed. "
                "Install it with: pip install requests"
            )
        
        # Get credentials from parameters or environment variables
        # Priority: explicit parameters > embedding-specific env vars > general env vars
        self.api_key = api_key or os.getenv("VNPT_EMBEDDING_AUTHORIZATION")
        self.token_id = token_id or os.getenv("VNPT_EMBEDDING_TOKEN_ID")
        self.token_key = token_key or os.getenv("VNPT_EMBEDDING_TOKEN_KEY")
        
        # Log credential status (masked for security) - use INFO level to ensure visibility
        if self.api_key:
            masked_key = self.api_key[:8] + "*" * (len(self.api_key) - 12) + self.api_key[-4:] if len(self.api_key) > 12 else "***"
            logger.info(f"VNPT Embedding Authorization loaded: {masked_key} (length: {len(self.api_key)})")
        else:
            logger.warning("VNPT Embedding Authorization NOT loaded - checking env vars...")
            logger.warning(f"  VNPT_EMBEDDING_AUTHORIZATION: {'SET' if os.getenv('VNPT_EMBEDDING_AUTHORIZATION') else 'NOT SET'}")
            logger.warning(f"  VNPT_AUTHORIZATION: {'SET' if os.getenv('VNPT_AUTHORIZATION') else 'NOT SET'}")
        
        if self.token_id:
            masked_id = self.token_id[:4] + "*" * (len(self.token_id) - 8) + self.token_id[-4:] if len(self.token_id) > 8 else "***"
            logger.info(f"VNPT Embedding Token ID loaded: {masked_id} (length: {len(self.token_id)})")
        else:
            logger.warning("VNPT Embedding Token ID NOT loaded")
            logger.warning(f"  VNPT_EMBEDDING_TOKEN_ID: {'SET' if os.getenv('VNPT_EMBEDDING_TOKEN_ID') else 'NOT SET'}")
            logger.warning(f"  VNPT_TOKEN_ID: {'SET' if os.getenv('VNPT_TOKEN_ID') else 'NOT SET'}")
        
        if self.token_key:
            masked_key_val = self.token_key[:4] + "*" * (len(self.token_key) - 8) + self.token_key[-4:] if len(self.token_key) > 8 else "***"
            logger.info(f"VNPT Embedding Token Key loaded: {masked_key_val} (length: {len(self.token_key)})")
        else:
            logger.warning("VNPT Embedding Token Key NOT loaded")
            logger.warning(f"  VNPT_EMBEDDING_TOKEN_KEY: {'SET' if os.getenv('VNPT_EMBEDDING_TOKEN_KEY') else 'NOT SET'}")
            logger.warning(f"  VNPT_TOKEN_KEY: {'SET' if os.getenv('VNPT_TOKEN_KEY') else 'NOT SET'}")
        
        if not self.api_key:
            raise ValueError(
                "No Authorization token provided for VNPT Embedding. Either:\n"
                "1. Set VNPT_EMBEDDING_AUTHORIZATION or VNPT_AUTHORIZATION environment variable\n"
                "2. Create .env file with VNPT_EMBEDDING_AUTHORIZATION=your-token (or VNPT_AUTHORIZATION)\n"
                "3. Pass api_key parameter"
            )
        
        if not self.token_id or not self.token_key:
            raise ValueError(
                "No Token ID/Key provided for VNPT Embedding. Either:\n"
                "1. Set VNPT_EMBEDDING_TOKEN_ID/TOKEN_KEY or VNPT_TOKEN_ID/TOKEN_KEY environment variables\n"
                "2. Create .env file with VNPT_EMBEDDING_TOKEN_ID=... and VNPT_EMBEDDING_TOKEN_KEY=...\n"
                "   (or use general VNPT_TOKEN_ID and VNPT_TOKEN_KEY)\n"
                "3. Pass token_id and token_key parameters"
            )
        
        self.model = model
        self.endpoint = f"{base_url}/vnptai-hackathon-embedding"
        
        logger.info(f"Initialized VNPT Embedding function with model: {model}")
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Embed a list of texts using VNPT Embedding API.
        
        This method signature (self, input) is required by Chroma 0.4.16+.
        
        Args:
            input: List of text strings to embed
        
        Returns:
            List of embedding vectors (each is a list of floats)
        """
        try:
            # Strip whitespace from credentials (common issue with .env files)
            api_key_clean = self.api_key.strip() if self.api_key else None
            token_id_clean = self.token_id.strip() if self.token_id else None
            token_key_clean = self.token_key.strip() if self.token_key else None
            
            # Prepare headers
            headers = {
                'Authorization': f'Bearer {api_key_clean}',
                'Token-id': token_id_clean,
                'Token-key': token_key_clean,
                'Content-Type': 'application/json',
            }
            
            # Log request details (first call only to avoid spam)
            if not hasattr(self, '_logged_request'):
                logger.info(f"VNPT Embedding API endpoint: {self.endpoint}")
                logger.info(f"VNPT Embedding headers keys: {list(headers.keys())}")
                # Log if credentials are empty (masked)
                if api_key_clean:
                    logger.info(f"Authorization header present (length: {len(api_key_clean)})")
                else:
                    logger.error("Authorization header is EMPTY!")
                if token_id_clean:
                    logger.info(f"Token-id header present (length: {len(token_id_clean)})")
                else:
                    logger.error("Token-id header is EMPTY!")
                if token_key_clean:
                    logger.info(f"Token-key header present (length: {len(token_key_clean)})")
                else:
                    logger.error("Token-key header is EMPTY!")
                self._logged_request = True
            
            # Process each text individually
            all_embeddings = []
            
            for text in input:
                # Prepare request body (exact format from API documentation)
                # Use 'float' format to get list of floats directly (easier to process)
                json_data = {
                    'model': self.model,
                    'input': text,
                    'encoding_format': 'float'  # Changed from 'base64' to 'float' for easier processing
                }
                
                # Call VNPT Embedding API (format matches documentation example)
                # Endpoint: https://api.idg.vnpt.vn/data-service/vnptai-hackathon-embedding
                response = requests.post(
                    self.endpoint, 
                    headers=headers, 
                    json=json_data
                )
                
                # Log error details if request fails
                if response.status_code != 200:
                    logger.error(f"VNPT Embedding API returned status {response.status_code}")
                    try:
                        error_detail = response.json()
                        logger.error(f"API error response: {error_detail}")
                    except:
                        logger.error(f"API error response (text): {response.text[:500]}")
                
                response.raise_for_status()
                
                result = response.json()
                
                # Extract embedding from VNPT API format
                if result.get('data') and len(result['data']) > 0:
                    embedding = result['data'][0].get('embedding', [])
                else:
                    raise ValueError(f"No embedding found in VNPT API response: {result}")
                
                # Ensure it's a list of floats
                if isinstance(embedding, list):
                    all_embeddings.append(embedding)
                else:
                    # Convert to list if needed
                    all_embeddings.append(list(embedding))
            
            return all_embeddings
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling VNPT Embedding API: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    logger.error(f"API error details: {error_detail}")
                except:
                    logger.error(f"API error response: {e.response.text}")
            raise RuntimeError(f"VNPT Embedding API error: {str(e)}")
        except Exception as e:
            logger.error(f"Error calling VNPT Embedding API: {str(e)}")
            raise RuntimeError(f"VNPT Embedding API error: {str(e)}")


def get_vnpt_embedding_function(
    api_key: Optional[str] = None,
    token_id: Optional[str] = None,
    token_key: Optional[str] = None,
    model: str = "vnptai_hackathon_embedding"
):
    """
    Create a VNPT Embedding function for Chroma.
    
    Args:
        api_key: Authorization Bearer token (or use VNPT_AUTHORIZATION env var)
        token_id: Token ID (or use VNPT_TOKEN_ID env var)
        token_key: Token Key (or use VNPT_TOKEN_KEY env var)
        model: Embedding model name (default: "vnptai_hackathon_embedding")
    
    Returns:
        Embedding function compatible with Chroma 0.4.16+
    
    Raises:
        ImportError: If requests is not installed
        ValueError: If credentials not provided
    """
    return VNPTEmbeddingFunction(
        api_key=api_key,
        token_id=token_id,
        token_key=token_key,
        model=model
    )

