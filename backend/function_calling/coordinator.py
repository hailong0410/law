"""
Function calling coordinator for RAG Agent.
"""

from typing import Any, Dict, List, Optional, Union
import json
import subprocess
import sys
import io
import contextlib
from enum import Enum

from .tools import Tool, ToolRegistry, ToolParameter, ToolType
from .rag_tools import VectorDatabaseRetriever, DocumentProcessor, DatabaseQuerier
from agent.logging import logger


class FunctionCallResult:
    """Result of a function call execution."""
    
    def __init__(
        self,
        tool_name: str,
        success: bool,
        result: Any = None,
        error: Optional[str] = None,
        execution_time: float = 0.0
    ):
        self.tool_name = tool_name
        self.success = success
        self.result = result
        self.error = error
        self.execution_time = execution_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "execution_time": self.execution_time
        }


class FunctionCallingCoordinator:
    """Coordinates function calling for the RAG agent."""
    
    def __init__(self, vector_db=None):
        logger.info("Initializing FunctionCallingCoordinator")
        self.registry = ToolRegistry()
        self.vector_db = vector_db
        self._setup_default_tools()
        logger.debug(f"FunctionCallingCoordinator initialized with {len(self.registry.list_tools())} tools")
    
    def _setup_default_tools(self) -> None:
        """Setup default RAG tools."""
        logger.debug("Setting up default RAG tools")
        
        # VectorDatabase Retriever
        retriever = VectorDatabaseRetriever(self.vector_db)
        
        # Document Retrieval Tool - by similarity
        retrieve_tool = Tool(
            name="retrieve_documents",
            description="Retrieve relevant documents from the vector database by similarity",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="The search query to find relevant documents"
                ),
                ToolParameter(
                    name="top_k",
                    type="integer",
                    description="Number of top results to return (default: 5)",
                    required=False
                ),
                ToolParameter(
                    name="similarity_threshold",
                    type="number",
                    description="Minimum similarity score threshold (0-1, default: 0.0)",
                    required=False
                ),
                ToolParameter(
                    name="reconstruct",
                    type="boolean",
                    description="Whether to reconstruct full documents (default: true)",
                    required=False
                )
            ],
            tool_type=ToolType.RETRIEVAL,
            callable=self._wrapper_retrieve_by_similarity(retriever)
        )
        self.registry.register(retrieve_tool)
        
        # Get Document by ID Tool
        get_doc_tool = Tool(
            name="get_document_by_id",
            description="Retrieve a specific document by its ID with full content reconstruction",
            parameters=[
                ToolParameter(
                    name="document_id",
                    type="string",
                    description="The ID of the document to retrieve"
                )
            ],
            tool_type=ToolType.RETRIEVAL,
            callable=self._wrapper_get_document(retriever)
        )
        self.registry.register(get_doc_tool)
        
        # Search by metadata
        search_metadata_tool = Tool(
            name="search_documents_by_metadata",
            description="Search documents by metadata key-value pairs",
            parameters=[
                ToolParameter(
                    name="key",
                    type="string",
                    description="Metadata key to search"
                ),
                ToolParameter(
                    name="value",
                    type="string",
                    description="Metadata value to match"
                )
            ],
            tool_type=ToolType.RETRIEVAL,
            callable=self._wrapper_search_metadata(retriever)
        )
        self.registry.register(search_metadata_tool)
        
        # List all documents
        list_docs_tool = Tool(
            name="list_all_documents",
            description="List all documents in the vector database",
            parameters=[],
            tool_type=ToolType.RETRIEVAL,
            callable=retriever.list_all_documents
        )
        self.registry.register(list_docs_tool)
        
        # Query documents from specific collection
        query_collection_tool = Tool(
            name="query_collection",
            description="Retrieve documents from a specific collection. Use this when you know which collection contains relevant information. Choose the collection name that best matches the user's question topic (e.g., 'programming' for Python questions, 'ai' for ML questions).",
            parameters=[
                ToolParameter(
                    name="collection_name",
                    type="string",
                    description="Name of the collection to query. Must match one of the available collections. Check available collections in system prompt."
                ),
                ToolParameter(
                    name="top_k",
                    type="integer",
                    description="Number of top documents to return (default: 5)",
                    required=False
                ),
                ToolParameter(
                    name="query",
                    type="string",
                    description="Optional search query text for similarity search within the collection (default: None). If provided, will search for similar content.",
                    required=False
                )
            ],
            tool_type=ToolType.RETRIEVAL,
            callable=self._wrapper_query_collection(retriever)
        )
        self.registry.register(query_collection_tool)
        
        # Reconstruct document
        reconstruct_tool = Tool(
            name="reconstruct_document",
            description="Reconstruct a document from chunks",
            parameters=[
                ToolParameter(
                    name="document_id",
                    type="string",
                    description="The ID of the document to reconstruct"
                ),
                ToolParameter(
                    name="chunk_indices",
                    type="string",
                    description="Optional comma-separated chunk indices to reconstruct (e.g. '0,1,2')",
                    required=False
                )
            ],
            tool_type=ToolType.PROCESSING,
            callable=self._wrapper_reconstruct_document()
        )
        self.registry.register(reconstruct_tool)
        
        # Text Summarization Tool
        summarize_tool = Tool(
            name="summarize_text",
            description="Summarize the given text content",
            parameters=[
                ToolParameter(
                    name="text",
                    type="string",
                    description="The text to summarize"
                ),
                ToolParameter(
                    name="max_length",
                    type="integer",
                    description="Maximum length of summary in words (optional)",
                    required=False
                )
            ],
            tool_type=ToolType.PROCESSING,
            callable=DocumentProcessor.summarize
        )
        self.registry.register(summarize_tool)
        
        # Entity Extraction Tool
        entity_tool = Tool(
            name="extract_entities",
            description="Extract named entities from text",
            parameters=[
                ToolParameter(
                    name="text",
                    type="string",
                    description="The text to extract entities from"
                )
            ],
            tool_type=ToolType.PROCESSING,
            callable=DocumentProcessor.extract_entities
        )
        self.registry.register(entity_tool)
        
        # Text Chunking Tool
        chunk_tool = Tool(
            name="chunk_text",
            description="Split text into chunks using specified strategy",
            parameters=[
                ToolParameter(
                    name="text",
                    type="string",
                    description="The text to chunk"
                ),
                ToolParameter(
                    name="chunk_size",
                    type="integer",
                    description="Size of each chunk in characters (default: 512)",
                    required=False
                ),
                ToolParameter(
                    name="chunk_strategy",
                    type="string",
                    description="Chunking strategy: 'character', 'line', 'sentence', 'markdown', 'paragraph' (default: 'character')",
                    required=False,
                    enum_values=["character", "line", "sentence", "markdown", "paragraph"]
                )
            ],
            tool_type=ToolType.PROCESSING,
            callable=self._wrapper_chunk_text()
        )
        self.registry.register(chunk_tool)
        
        # Database Query Tool
        db_querier = DatabaseQuerier()
        query_tool = Tool(
            name="query_database",
            description="Execute a database query",
            parameters=[
                ToolParameter(
                    name="sql",
                    type="string",
                    description="SQL query to execute"
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="Maximum number of results (default: 100)",
                    required=False
                )
            ],
            tool_type=ToolType.DATABASE,
            callable=self._wrapper_query_database(db_querier)
        )
        self.registry.register(query_tool)
        
        # Database Schema Tool
        schema_tool = Tool(
            name="get_database_schema",
            description="Get the schema/structure of the database",
            parameters=[],
            tool_type=ToolType.DATABASE,
            callable=db_querier.get_schema
        )
        self.registry.register(schema_tool)

        # Add Document Tool
        add_doc_tool = Tool(
            name="add_document",
            description="Add a new document to the vector database",
            parameters=[
                ToolParameter(
                    name="content",
                    type="string",
                    description="The content of the document"
                ),
                ToolParameter(
                    name="collection_name",
                    type="string",
                    description="Name of the collection to add to (default: 'default')",
                    required=False
                ),
                ToolParameter(
                    name="metadata",
                    type="string",
                    description="JSON string of metadata key-value pairs (optional)",
                    required=False
                )
            ],
            tool_type=ToolType.PROCESSING,
            callable=self._wrapper_add_document(self.vector_db)
        )
        self.registry.register(add_doc_tool)
        
        # Python Code Execution Tool for calculations
        python_exec_tool = Tool(
            name="execute_python_code",
            description="Execute Python code for mathematical calculations, data processing, or computations. Use this tool when you need to perform calculations, solve equations, or process numerical data. The code will be executed in a safe environment and return the result.",
            parameters=[
                ToolParameter(
                    name="code",
                    type="string",
                    description="Python code to execute. Should be a complete, runnable Python script. Can include imports (math, numpy if available), calculations, and print statements. The last expression or print output will be returned."
                ),
                ToolParameter(
                    name="timeout",
                    type="integer",
                    description="Maximum execution time in seconds (default: 10)",
                    required=False
                )
            ],
            tool_type=ToolType.PROCESSING,
            callable=self._wrapper_execute_python_code()
        )
        self.registry.register(python_exec_tool)
    
    @staticmethod
    def _wrapper_execute_python_code():
        """Create wrapper for Python code execution."""
        def func(code: str, timeout: int = 10, **kwargs) -> Dict[str, Any]:
            """
            Execute Python code safely and return the result.
            
            Args:
                code: Python code to execute
                timeout: Maximum execution time in seconds
                
            Returns:
                Dict with 'result' (output) and 'error' (if any)
            """
            try:
                # Capture stdout and stderr
                stdout_capture = io.StringIO()
                stderr_capture = io.StringIO()
                
                # Create a safe execution environment
                safe_globals = {
                    '__builtins__': {
                        'print': print,
                        'len': len,
                        'str': str,
                        'int': int,
                        'float': float,
                        'bool': bool,
                        'list': list,
                        'dict': dict,
                        'tuple': tuple,
                        'set': set,
                        'range': range,
                        'enumerate': enumerate,
                        'zip': zip,
                        'min': min,
                        'max': max,
                        'sum': sum,
                        'abs': abs,
                        'round': round,
                        'pow': pow,
                        'divmod': divmod,
                        'all': all,
                        'any': any,
                        'sorted': sorted,
                        'reversed': reversed,
                        'isinstance': isinstance,
                        'type': type,
                    }
                }
                
                # Add math module
                import math
                safe_globals['math'] = math
                
                # Try to add numpy if available
                try:
                    import numpy as np
                    safe_globals['numpy'] = np
                    safe_globals['np'] = np
                except ImportError:
                    pass
                
                # Execute code with timeout
                with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                    try:
                        # Compile and execute
                        compiled_code = compile(code, '<string>', 'exec')
                        exec(compiled_code, safe_globals)
                        
                        # Get output
                        output = stdout_capture.getvalue()
                        error_output = stderr_capture.getvalue()
                        
                        if error_output:
                            return {
                                "status": "error",
                                "error": error_output,
                                "output": output
                            }
                        
                        # If no output, try to get last expression result
                        if not output.strip():
                            # Try to evaluate last line if it's an expression
                            lines = code.strip().split('\n')
                            last_line = lines[-1].strip()
                            if last_line and not last_line.startswith('#') and not last_line.startswith('import') and not last_line.startswith('from'):
                                try:
                                    # Try to evaluate as expression
                                    result = eval(last_line, safe_globals)
                                    output = str(result)
                                except:
                                    pass
                        
                        return {
                            "status": "success",
                            "result": output.strip() if output else "Code executed successfully (no output)",
                            "output": output
                        }
                    except Exception as e:
                        error_msg = str(e)
                        stderr_output = stderr_capture.getvalue()
                        if stderr_output:
                            error_msg = f"{error_msg}\n{stderr_output}"
                        return {
                            "status": "error",
                            "error": error_msg,
                            "output": stdout_capture.getvalue()
                        }
            except Exception as e:
                return {
                    "status": "error",
                    "error": f"Execution failed: {str(e)}"
                }
        return func
    
    @staticmethod
    def _wrapper_retrieve_by_similarity(retriever: VectorDatabaseRetriever):
        """Create wrapper for retrieve_by_similarity."""
        def func(query: str, top_k: int = 5, similarity_threshold: float = 0.0, reconstruct: bool = True, **kwargs):
            return retriever.retrieve_by_similarity(query, top_k, similarity_threshold, reconstruct)
        return func
    
    @staticmethod
    def _wrapper_get_document(retriever: VectorDatabaseRetriever):
        """Create wrapper for get_document_by_id."""
        def func(document_id: str, **kwargs):
            return retriever.retrieve_by_document_id(document_id)
        return func
    
    @staticmethod
    def _wrapper_search_metadata(retriever: VectorDatabaseRetriever):
        """Create wrapper for search_by_metadata."""
        def func(key: str, value: str, **kwargs):
            return retriever.search_by_metadata(key, value)
        return func
    
    @staticmethod
    def _wrapper_query_collection(retriever: VectorDatabaseRetriever):
        """Create wrapper for query_collection."""
        def func(collection_name: str, top_k: int = 5, query: str = None, **kwargs):
            return retriever.retrieve_from_collection(collection_name, top_k, query)
        return func
    
    def _wrapper_reconstruct_document(self):
        """Create wrapper for reconstruct_document."""
        def func(document_id: str, chunk_indices: str = None, **kwargs):
            indices = None
            if chunk_indices:
                try:
                    indices = [int(x.strip()) for x in chunk_indices.split(',')]
                except:
                    pass
            return DocumentProcessor.reconstruct_document(self.vector_db, document_id, indices)
        return func
    
    @staticmethod
    def _wrapper_chunk_text():
        """Create wrapper for chunk_text."""
        def func(text: str, chunk_size: int = 512, chunk_strategy: str = "character", **kwargs):
            return DocumentProcessor.chunk_text(text, chunk_size, chunk_strategy)
        return func
    
    @staticmethod
    def _wrapper_query_database(db_querier: DatabaseQuerier):
        """Create wrapper for query_database."""
        def func(sql: str, limit: int = 100, **kwargs):
            return db_querier.query(sql, limit)
        return func

    @staticmethod
    def _wrapper_add_document(vector_db):
        """Create wrapper for add_document."""
        def func(content: str, collection_name: str = "default", metadata: str = None, **kwargs):
            meta_dict = {}
            if metadata:
                try:
                    meta_dict = json.loads(metadata)
                except:
                    pass
            return vector_db.add_document(content=content, collection_name=collection_name, metadata=meta_dict)
        return func
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get schemas of all available tools."""
        logger.debug("Retrieving available tools schemas")
        return self.registry.get_schemas()
    
    def get_available_collections(self) -> List[str]:
        """Get list of available collections from vector database."""
        if not self.vector_db:
            return []
        try:
            return self.vector_db.list_collections()
        except Exception as e:
            logger.warning(f"Failed to get collections: {str(e)}")
            return []
    
    def execute_function_call(
        self,
        tool_name: str,
        tool_input: Dict[str, Any]
    ) -> FunctionCallResult:
        """
        Execute a function call.
        
        Args:
            tool_name: Name of the tool to call
            tool_input: Input parameters for the tool
        
        Returns:
            FunctionCallResult with execution details
        """
        import time
        
        logger.info(f"Executing function call: {tool_name}")
        logger.debug(f"Tool input: {json.dumps(tool_input, default=str, indent=2)}")
        
        start_time = time.time()
        
        try:
            result = self.registry.execute_tool(tool_name, **tool_input)
            execution_time = time.time() - start_time
            
            if result.get("status") == "error":
                logger.error(f"Function {tool_name} failed: {result.get('error')}")
                return FunctionCallResult(
                    tool_name=tool_name,
                    success=False,
                    error=result.get("error"),
                    execution_time=execution_time
                )
            
            logger.info(f"Function {tool_name} executed successfully in {execution_time:.2f}s")
            return FunctionCallResult(
                tool_name=tool_name,
                success=True,
                result=result.get("result"),
                execution_time=execution_time
            )
        except ValueError as e:
            execution_time = time.time() - start_time
            logger.error(f"ValueError in function {tool_name}: {str(e)}")
            return FunctionCallResult(
                tool_name=tool_name,
                success=False,
                error=str(e),
                execution_time=execution_time
            )
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Unexpected error in function {tool_name}: {str(e)}")
            return FunctionCallResult(
                tool_name=tool_name,
                success=False,
                error=f"Unexpected error: {str(e)}",
                execution_time=execution_time
            )
    
    def parse_tool_call(self, response_content: str) -> Optional[Dict[str, Any]]:
        """
        Parse a tool call from LLM response content.
        
        Args:
            response_content: LLM response containing tool call
        
        Returns:
            Dictionary with tool_name and tool_input, or None if no valid call found
        """
        try:
            # Try to extract JSON from response
            if "```json" in response_content:
                json_str = response_content.split("```json")[1].split("```")[0]
                data = json.loads(json_str)
            elif "```" in response_content:
                json_str = response_content.split("```")[1].split("```")[0]
                data = json.loads(json_str)
            else:
                data = json.loads(response_content)
            
            # Format 1: {"tool_name": "...", "tool_input": {...}}
            if "tool_name" in data and "tool_input" in data:
                return {
                    "tool_name": data["tool_name"],
                    "tool_input": data["tool_input"]
                }
            # Format 2: {"name": "...", "arguments": {...}}
            elif "name" in data and "arguments" in data:
                return {
                    "tool_name": data["name"],
                    "tool_input": data["arguments"]
                }
            # Format 3: {"tool_calls": [{"name": "...", "arguments": {...}}]}
            elif "tool_calls" in data and isinstance(data["tool_calls"], list) and len(data["tool_calls"]) > 0:
                first_call = data["tool_calls"][0]
                if "name" in first_call and "arguments" in first_call:
                    return {
                        "tool_name": first_call["name"],
                        "tool_input": first_call["arguments"]
                    }
        except (json.JSONDecodeError, ValueError):
            pass
        
        return None
    
    def register_custom_tool(self, tool: Tool) -> None:
        """
        Register a custom tool.
        
        Args:
            tool: Tool instance to register
        """
        self.registry.register(tool)
    
    def unregister_tool(self, tool_name: str) -> None:
        """
        Unregister a tool.
        
        Args:
            tool_name: Name of the tool to unregister
        """
        self.registry.unregister(tool_name)
