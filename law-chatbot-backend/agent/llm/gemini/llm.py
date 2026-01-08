"""Gemini LLM - Direct Google Generative AI client."""
from typing import Any, Dict, List, Optional
import json
import os
from agent.logging import logger

# Try to import google-generativeai
try:
    import google.generativeai as genai
    GOOGLE_GENERATIVEAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENERATIVEAI_AVAILABLE = False


class GeminiLLM:
    """Direct Google Gemini LLM API client."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize Gemini backend with API key.
        
        Args:
            api_key: Google Gemini API key. If None, will look for GEMINI_API_KEY env var
            model: Model name to use. If None, will look for GEMINI_MODEL env var, then default to "gemini-pro"
        
        Raises:
            ValueError: If no API key provided and GEMINI_API_KEY env var not set
            ImportError: If google-generativeai is not installed
        """
        if not GOOGLE_GENERATIVEAI_AVAILABLE:
            logger.error("google-generativeai is not installed")
            raise ImportError(
                "google-generativeai is not installed. "
                "Install it with: pip install google-generativeai"
            )
        
        # Get API key from parameter or environment variable
        # Priority: explicit parameter > GEMINI_API_KEY env var
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            logger.error("No API key found in GEMINI_API_KEY environment variable or parameter")
            raise ValueError(
                "No API key provided. Either:\n"
                "1. Set GEMINI_API_KEY environment variable\n"
                "2. Create .env file with GEMINI_API_KEY=your-key\n"
                "3. Pass api_key parameter: GeminiBackend(api_key='your-key')"
            )
        
        # Get model from parameter or environment variable
        # Priority: explicit parameter > GEMINI_MODEL env var > default
        self.model_name = model or os.getenv("GEMINI_MODEL", "gemini-pro")
        
        logger.info(f"Configuring Gemini API with model: {model}")
        
        # Configure the API
        genai.configure(api_key=self.api_key)
        
        # Get the model
        self.model = genai.GenerativeModel(model)
        
        logger.info(f"GeminiLLM initialized successfully with model: {model}")
    
    def create_chat_completion(
        self, 
        messages: List[Dict[str, Any]], 
        coordinator: Optional[Any] = None, 
        max_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Call the Gemini API with the given messages.
        
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
            gemini_messages = self._convert_messages(messages)
            
            logger.debug(f"Calling Gemini API with {len(gemini_messages)} messages")
            
            # Call the API
            response = self.model.generate_content(gemini_messages)
            
            # Extract the response text
            if response and response.text:
                logger.debug(f"Received response from Gemini API (length: {len(response.text)})")
                return {
                    "content": response.text,
                    "model": self.model_name,
                    "finish_reason": response.candidates[0].finish_reason if response.candidates else None
                }
            else:
                logger.warning("Empty response from Gemini API")
                return {"content": "No response from API"}
        
        except Exception as e:
            logger.error(f"Error calling Gemini API: {str(e)}")
            raise RuntimeError(f"Gemini API error: {str(e)}")
    
    def _execute_with_function_calling(
        self, 
        messages: List[Dict[str, Any]], 
        coordinator: Any, 
        max_iterations: int
    ) -> Dict[str, Any]:
        """Execute chat completion with function calling support."""
        iteration = 0
        current_messages = messages.copy()
        
        while iteration < max_iterations:
            iteration += 1
            logger.debug(f"Function calling iteration {iteration}/{max_iterations}")
            
            # Convert and call Gemini API
            gemini_messages = self._convert_messages(current_messages)
            response = self.model.generate_content(gemini_messages)
            
            # Check for native function calls only (Gemini API native format)
            tool_calls = self._extract_native_function_calls(response)
            
            # Get response text if available
            response_text = response.text if response and response.text else ""
            
            if tool_calls and len(tool_calls) > 0:
                # Execute all tool calls in sequence
                tool_results = []
                tool_names = []
                
                for tool_call in tool_calls:
                    tool_name = tool_call.get("tool_name")
                    tool_input = tool_call.get("tool_input", {})
                    tool_names.append(tool_name)
                    
                    logger.info(f"Executing tool: {tool_name}")
                    result = coordinator.execute_function_call(tool_name, tool_input)
                    
                    if result.success:
                        tool_result = json.dumps(result.result, default=str, indent=2)
                        tool_results.append(f"Công cụ '{tool_name}' đã được thực thi thành công. Kết quả: {tool_result}")
                    else:
                        tool_results.append(f"Công cụ '{tool_name}' thực thi thất bại. Lỗi: {result.error}")
                
                # Add assistant message with all tool calls
                tool_calls_summary = ", ".join(tool_names)
                current_messages.append({
                    "role": "assistant", 
                    "content": response_text if response_text else f"Tool calls: {tool_calls_summary}"
                })
                
                # Add all tool results as user message
                all_results = "\n\n".join(tool_results)
                current_messages.append({
                    "role": "user",
                    "content": f"Các công cụ đã được thực thi:\n\n{all_results}\n\nTiếp tục nhiệm vụ sử dụng thông tin này."
                })
                
                logger.debug(f"All {len(tool_calls)} tool executions completed, continuing conversation")
                continue
            
            # No tool calls found, return final response
            if not response or not response.text:
                logger.warning("Empty response from Gemini API")
                return {"content": "No response from API"}
            
            response_text = response.text
            logger.info(f"Final response received after {iteration} iterations")
            return {
                "content": response_text,
                "model": self.model_name,
                "finish_reason": response.candidates[0].finish_reason if response.candidates else None,
                "__debug_loop": {"iterations": iteration}
            }
        
        # Max iterations reached
        logger.warning(f"Max iterations ({max_iterations}) reached")
        return {
            "content": response_text if 'response_text' in locals() else "Max iterations reached",
            "model": self.model_name
        }
    
    def _extract_native_function_calls(self, response: Any) -> List[Dict[str, Any]]:
        """
        Extract all native function calls from Gemini API response.
        
        Gemini API can return multiple function calls in response.candidates[0].content.parts
        
        Args:
            response: Gemini API response object
            
        Returns:
            List of dictionaries with tool_name and tool_input, empty list if no function calls found
        """
        tool_calls = []
        try:
            if not response or not response.candidates:
                return tool_calls
            
            candidate = response.candidates[0]
            if not hasattr(candidate, 'content') or not candidate.content:
                return tool_calls
            
            if not hasattr(candidate.content, 'parts') or not candidate.content.parts:
                return tool_calls
            
            # Check each part for function_call
            for part in candidate.content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    func_call = part.function_call
                    func_name = func_call.name if hasattr(func_call, 'name') else None
                    func_args = func_call.args if hasattr(func_call, 'args') else {}
                    
                    if func_name:
                        logger.info(f"Found native function call: {func_name}")
                        tool_calls.append({
                            "tool_name": func_name,
                            "tool_input": func_args if isinstance(func_args, dict) else {}
                        })
            
            if tool_calls:
                logger.info(f"Found {len(tool_calls)} native function call(s)")
            
            return tool_calls
        except Exception as e:
            logger.debug(f"Error extracting native function calls: {e}")
            return tool_calls
    
    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert messages from standard format to Gemini format.
        
        Standard format: [{"role": "user", "content": "..."}, ...]
        Gemini format: [{"role": "user", "parts": ["..."]}, ...]
        
        Note: Gemini API doesn't support "system" role, so system messages
        are merged into the first user message.
        """
        gemini_messages = []
        system_content = []
        
        # Collect all system messages
        for msg in messages:
            role = msg.get("role", "user")
            if role == "system":
                system_content.append(msg.get("content", ""))
        
        # Process non-system messages
        first_user = True
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Skip system messages (already collected)
            if role == "system":
                continue
            
            # Gemini API roles: "user" or "model" (not "assistant" or "system")
            gemini_role = "model" if role == "assistant" else "user"
            
            # Merge system content into first user message
            if first_user and gemini_role == "user" and system_content:
                full_content = "\n\n".join(system_content) + "\n\n" + content
                first_user = False
            else:
                full_content = content
                if gemini_role == "user":
                    first_user = False
            
            gemini_msg = {
                "role": gemini_role,
                "parts": [full_content] if isinstance(full_content, str) else full_content
            }
            
            gemini_messages.append(gemini_msg)
        
        return gemini_messages

