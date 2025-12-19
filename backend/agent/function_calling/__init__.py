"""
Initialization file for function_calling module.
"""

from .tools import Tool, ToolRegistry, ToolParameter, ToolType
from .coordinator import FunctionCallingCoordinator, FunctionCallResult
# rag_tools defines VectorDatabaseRetriever and DocumentProcessor. Provide
# compatibility aliases so older imports (DocumentRetriever, TextProcessor)
# continue to work.
from .rag_tools import (
    VectorDatabaseRetriever as DocumentRetriever,
    DocumentProcessor as TextProcessor,
    DatabaseQuerier,
)

__all__ = [
    "Tool",
    "ToolRegistry",
    "ToolParameter",
    "ToolType",
    "FunctionCallingCoordinator",
    "FunctionCallResult",
    "DocumentRetriever",
    "TextProcessor",
    "DatabaseQuerier",
]
