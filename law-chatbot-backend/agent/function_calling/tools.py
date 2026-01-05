"""
Tool definitions for RAG Agent function calling.
"""

from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class ToolType(Enum):
    """Types of tools available for the agent."""
    RETRIEVAL = "retrieval"
    DATABASE = "database"
    PROCESSING = "processing"
    EXTERNAL = "external"


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: str
    description: str
    required: bool = True
    enum_values: Optional[List[str]] = None


@dataclass
class Tool:
    """Definition of a callable tool."""
    name: str
    description: str
    parameters: List[ToolParameter]
    tool_type: ToolType
    callable: Callable
    
    def to_schema(self) -> Dict[str, Any]:
        """Convert tool to JSON schema format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            param_schema = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum_values:
                param_schema["enum"] = param.enum_values
            properties[param.name] = param_schema
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            }
        }


class ToolRegistry:
    """Registry for managing available tools."""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """Register a new tool."""
        self.tools[tool.name] = tool
    
    def unregister(self, tool_name: str) -> None:
        """Unregister a tool."""
        if tool_name in self.tools:
            del self.tools[tool_name]
    
    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(tool_name)
    
    def list_tools(self) -> List[Tool]:
        """List all registered tools."""
        return list(self.tools.values())
    
    def get_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all registered tools."""
        return [tool.to_schema() for tool in self.tools.values()]
    
    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool with given parameters."""
        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found in registry")
        
        try:
            result = tool.callable(**kwargs)
            return {"status": "success", "result": result}
        except Exception as e:
            return {"status": "error", "error": str(e)}
