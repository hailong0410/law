"""
AgentChain - Retrieval-Augmented Generation Agent Framework

This package provides a complete RAG agent implementation with:
- Vector database for document storage and retrieval
- Function calling coordinator for tool management
- LLM adapter with Gemini API support
- Planning executor for complex task decomposition
- Comprehensive logging system
"""

# Auto-load environment variables from .env file on package import
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, will use system environment variables
    pass

# Export main classes
from .agent import RAGAgent
from .logging import logger, setup_logger

__version__ = "0.1.0"
__all__ = ["RAGAgent", "logger", "setup_logger"]
