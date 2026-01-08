"""OpenAI LLM - Direct OpenAI client."""
from typing import Any, Dict, List, Optional
import json
import os
from agent.logging import logger

# Try to import openai
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class OpenAILLM:
    """Direct OpenAI LLM API client."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize OpenAI backend with API key.
        
        Args:
            api_key: OpenAI API key. If None, will look for OPENAI_API_KEY env var
            model: Model name to use. If None, will look for OPENAI_MODEL env var, then default to "gpt-3.5-turbo"
            base_url: Optional base URL for API (useful for OpenAI-compatible APIs)
        
        Raises:
            ValueError: If no API key provided and OPENAI_API_KEY env var not set
            ImportError: If openai is not installed
        """
        if not OPENAI_AVAILABLE:
            logger.error("openai is not installed")
            raise ImportError(
                "openai is not installed. "
                "Install it with: pip install openai"
            )
        
        # Get API key from parameter or environment variable
        # Priority: explicit parameter > OPENAI_API_KEY env var
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            logger.error("No API key found in OPENAI_API_KEY environment variable or parameter")
            raise ValueError(
                "No API key provided. Either:\n"
                "1. Set OPENAI_API_KEY environment variable\n"
                "2. Create .env file with OPENAI_API_KEY=your-key\n"
                "3. Pass api_key parameter: OpenAILLM(api_key='your-key')"
            )
        
        # Get model from parameter or environment variable
        # Priority: explicit parameter > OPENAI_MODEL env var > default
        self.model_name = model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        
        # Get base_url from parameter or environment variable
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        
        logger.info(f"Configuring OpenAI API with model: {self.model_name}")
        
        # Initialize OpenAI client
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        
        self.client = OpenAI(**client_kwargs)
        
        logger.info(f"OpenAILLM initialized successfully with model: {self.model_name}")
    
    def create_chat_completion(
        self, 
        messages: List[Dict[str, Any]], 
        coordinator: Optional[Any] = None, 
        max_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Call the OpenAI API with the given messages.
        
        Args:
            messages: List of message dicts in format [{"role": "user", "content": "..."}, ...]
            coordinator: Optional function calling coordinator for tool execution
            max_iterations: Maximum number of function calling iterations
        
        Returns:
            Dict with 'content' key containing the response text
        """
        try:
            # If coordinator is provided, implement function calling loop
            if coordinator:
                return self._execute_with_function_calling(messages, coordinator, max_iterations)
            
            # Simple call without function calling
            logger.debug(f"Calling OpenAI API with {len(messages)} messages")
            
            # Call the API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            
            # Extract the response text
            if response and response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                logger.debug(f"Received response from OpenAI API (length: {len(content) if content else 0})")
                return {
                    "content": content or "",
                    "model": self.model_name,
                    "finish_reason": response.choices[0].finish_reason
                }
            else:
                logger.warning("Empty response from OpenAI API")
                return {"content": "No response from API"}
        
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            raise RuntimeError(f"OpenAI API error: {str(e)}")
    
    def _execute_with_function_calling(
        self, 
        messages: List[Dict[str, Any]], 
        coordinator: Any, 
        max_iterations: int
    ) -> Dict[str, Any]:
        """Execute chat completion with function calling support."""
        iteration = 0
        current_messages = messages.copy()
        
        # Get tools from coordinator
        tools = coordinator.get_available_tools() if coordinator else None
        
        while iteration < max_iterations:
            iteration += 1
            logger.debug(f"Function calling iteration {iteration}/{max_iterations}")
            
            # Prepare API call parameters
            api_params = {
                "model": self.model_name,
                "messages": current_messages
            }
            
            # Add tools if available (always include tools to allow function calling)
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = "auto"
            
            # Call OpenAI API
            response = self.client.chat.completions.create(**api_params)
            
            # Get response message
            response_message = response.choices[0].message
            response_text = response_message.content or ""
            tool_calls = response_message.tool_calls or []
            
            # Add assistant message to conversation
            assistant_msg = {
                "role": "assistant",
                "content": response_text
            }
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in tool_calls
                ]
            current_messages.append(assistant_msg)
            
            if tool_calls and len(tool_calls) > 0:
                # Execute all tool calls in sequence
                tool_results = []
                tool_names = []
                
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        tool_input = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        tool_input = {}
                    
                    tool_names.append(tool_name)
                    
                    logger.info(f"Executing tool: {tool_name}")
                    result = coordinator.execute_function_call(tool_name, tool_input)
                    
                    if result.success:
                        tool_result = json.dumps(result.result, default=str, indent=2)
                        tool_results.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_name,
                            "content": f"Công cụ '{tool_name}' đã được thực thi thành công. Kết quả: {tool_result}"
                        })
                    else:
                        tool_results.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_name,
                            "content": f"Công cụ '{tool_name}' thực thi thất bại. Lỗi: {result.error}"
                        })
                
                # Add all tool results to conversation
                current_messages.extend(tool_results)
                
                logger.debug(f"All {len(tool_calls)} tool executions completed, continuing conversation")
                continue
            
            # No tool calls found, return final response
            if not response_text:
                logger.warning("Empty response from OpenAI API")
                return {"content": "No response from API"}
            
            logger.info(f"Final response received after {iteration} iterations")
            return {
                "content": response_text,
                "model": self.model_name,
                "finish_reason": response.choices[0].finish_reason,
                "__debug_loop": {"iterations": iteration}
            }
        
        # Max iterations reached
        logger.warning(f"Max iterations ({max_iterations}) reached")
        return {
            "content": response_text if 'response_text' in locals() else "Max iterations reached",
            "model": self.model_name
        }

