"""LLM manager / factory for selecting available LLM adapters.

This module exposes `get_llm(llm_type, config)` which returns
a GeminiLLM or VNPTLLM instance.
"""
from typing import Any, Dict, Optional
import os

from .gemini import GeminiLLM
from .vnpt import VNPTLLM
from agent.logging import logger

# Auto-load .env file at module import time
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.debug("Loaded environment variables from .env file")
except ImportError:
    # dotenv not installed, just use environment variables
    pass


def get_llm(llm_type: str = "gemini", config: Optional[Dict[str, Any]] = None):
    """Return an instantiated LLM instance (GeminiLLM or VNPTLLM).

    Args:
        llm_type: string identifier ('gemini' or 'vnpt')
        config: configuration dict with keys:
            For Gemini:
            - 'api_key': API key for Gemini (or use GEMINI_API_KEY env var)
            - 'model': model name (or use GEMINI_MODEL env var, default: 'gemini-pro')
            For VNPT:
            - 'api_key': Authorization Bearer token (or use VNPT_AUTHORIZATION env var)
            - 'token_id': Token ID (or use VNPT_TOKEN_ID env var)
            - 'token_key': Token Key (or use VNPT_TOKEN_KEY env var)
            - 'model': model name - 'vnptai_hackathon_small' or 'vnptai_hackathon_large'
                       (or use VNPT_MODEL env var, default: 'vnptai_hackathon_large')
    
    Returns:
        GeminiLLM or VNPTLLM instance
        
    Raises:
        ValueError: If llm_type is not 'gemini' or 'vnpt', or if credentials not provided
        ImportError: If required packages not installed
    
    Examples:
        # Using Gemini with environment variable for API key
        llm = get_llm("gemini")
        
        # Using VNPT with environment variables
        llm = get_llm("vnpt")
        
        # Using VNPT with explicit credentials and model
        llm = get_llm("vnpt", config={
            "api_key": "your-auth-token",
            "token_id": "your-token-id",
            "token_key": "your-token-key",
            "model": "vnptai_hackathon_small"
        })
    """
    config = config or {}
    llm_type = (llm_type or "").lower()
    
    logger.info(f"Creating LLM of type: {llm_type}")

    if llm_type == "gemini":
        api_key = config.get("api_key")
        # Priority: config > GEMINI_MODEL env var > default
        model = config.get("model") or os.getenv("GEMINI_MODEL", "gemini-pro")
        
        logger.info(f"Creating GeminiLLM (model: {model})")
        llm = GeminiLLM(api_key=api_key, model=model)
        logger.info("GeminiLLM created successfully")
        
        return llm
    
    elif llm_type == "vnpt":
        api_key = config.get("api_key")
        token_id = config.get("token_id")
        token_key = config.get("token_key")
        # Use fixed model (default: large)
        model = config.get("model") or os.getenv("VNPT_MODEL", "vnptai_hackathon_large")
        logger.info(f"Creating VNPTLLM (model: {model})")
        llm = VNPTLLM(
            api_key=api_key,
            token_id=token_id,
            token_key=token_key,
            model=model
        )
        logger.info("VNPTLLM created successfully")
        return llm

    raise ValueError(f"Unknown llm_type: {llm_type}. Supported types: 'gemini', 'vnpt'.")
