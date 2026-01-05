"""VNPT LLM API client - Compatible with OpenAI-style API."""
from typing import Any, Dict, List, Optional
import json
import os
import time
from agent.logging import logger

# Try to import requests
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class VNPTLLM:
    """VNPT LLM API client compatible with OpenAI-style API."""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        token_id: Optional[str] = None,
        token_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: str = "https://api.idg.vnpt.vn/data-service"
    ):
        """
        Initialize VNPT LLM backend with API credentials.
        
        Args:
            api_key: Authorization Bearer token (or use VNPT_AUTHORIZATION env var)
            token_id: Token ID (or use VNPT_TOKEN_ID env var)
            token_key: Token Key (or use VNPT_TOKEN_KEY env var)
            model: Model name - "vnptai_hackathon_small" or "vnptai_hackathon_large"
                   (or use VNPT_MODEL env var, default: "vnptai_hackathon_large")
            base_url: Base URL for VNPT API (default: https://api.idg.vnpt.vn/data-service)
        
        Raises:
            ValueError: If required credentials not provided
            ImportError: If requests is not installed
        """
        if not REQUESTS_AVAILABLE:
            logger.error("requests is not installed")
            raise ImportError(
                "requests is not installed. "
                "Install it with: pip install requests"
            )
        
        # Get model from parameter or environment variable first (to determine which credentials to use)
        model_name = model or os.getenv("VNPT_MODEL", "vnptai_hackathon_large")
        
        # Get credentials from parameters or environment variables
        # Priority: explicit parameters > model-specific env vars > general env vars
        if not api_key:
            if "small" in model_name:
                api_key = os.getenv("VNPT_LLM_SMALL_AUTHORIZATION") or os.getenv("VNPT_AUTHORIZATION")
            elif "large" in model_name:
                api_key = os.getenv("VNPT_LLM_LARGE_AUTHORIZATION") or os.getenv("VNPT_AUTHORIZATION")
            else:
                api_key = os.getenv("VNPT_AUTHORIZATION")
        
        if not token_id:
            if "small" in model_name:
                token_id = os.getenv("VNPT_LLM_SMALL_TOKEN_ID") or os.getenv("VNPT_TOKEN_ID")
            elif "large" in model_name:
                token_id = os.getenv("VNPT_LLM_LARGE_TOKEN_ID") or os.getenv("VNPT_TOKEN_ID")
            else:
                token_id = os.getenv("VNPT_TOKEN_ID")
        
        if not token_key:
            if "small" in model_name:
                token_key = os.getenv("VNPT_LLM_SMALL_TOKEN_KEY") or os.getenv("VNPT_TOKEN_KEY")
            elif "large" in model_name:
                token_key = os.getenv("VNPT_LLM_LARGE_TOKEN_KEY") or os.getenv("VNPT_TOKEN_KEY")
            else:
                token_key = os.getenv("VNPT_TOKEN_KEY")
        
        self.api_key = api_key
        self.token_id = token_id
        self.token_key = token_key
        
        if not self.api_key:
            logger.error(f"No VNPT Authorization token found for model {model_name}")
            model_suffix = "_SMALL" if "small" in model_name else "_LARGE" if "large" in model_name else ""
            raise ValueError(
                f"No Authorization token provided for {model_name}. Either:\n"
                f"1. Set VNPT_LLM{model_suffix}_AUTHORIZATION or VNPT_AUTHORIZATION environment variable\n"
                f"2. Create .env file with VNPT_LLM{model_suffix}_AUTHORIZATION=your-token (or VNPT_AUTHORIZATION)\n"
                f"3. Pass api_key parameter"
            )
        
        if not self.token_id or not self.token_key:
            logger.error(f"No VNPT Token ID/Key found for model {model_name}")
            model_suffix = "_SMALL" if "small" in model_name else "_LARGE" if "large" in model_name else ""
            raise ValueError(
                f"No Token ID/Key provided for {model_name}. Either:\n"
                f"1. Set VNPT_LLM{model_suffix}_TOKEN_ID/TOKEN_KEY or VNPT_TOKEN_ID/TOKEN_KEY environment variables\n"
                f"2. Create .env file with VNPT_LLM{model_suffix}_TOKEN_ID=... and VNPT_LLM{model_suffix}_TOKEN_KEY=...\n"
                f"   (or use general VNPT_TOKEN_ID and VNPT_TOKEN_KEY)\n"
                f"3. Pass token_id and token_key parameters"
            )
        
        # Set model name
        self.model_name = model_name
        
        # Validate model name
        if self.model_name not in ["vnptai_hackathon_small", "vnptai_hackathon_large"]:
            logger.warning(f"Unknown model name: {self.model_name}, using vnptai_hackathon_large")
            self.model_name = "vnptai_hackathon_large"
        
        # Base URL should be without trailing slashes
        base_url = base_url.rstrip('/')
        self.base_url = base_url
        
        # Determine endpoint based on model
        # Endpoint format according to API documentation:
        # https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-small
        # https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-large
        if "small" in self.model_name:
            self.endpoint = f"{base_url}/v1/chat/completions/vnptai-hackathon-small"
        else:
            self.endpoint = f"{base_url}/v1/chat/completions/vnptai-hackathon-large"
        
        logger.info(f"VNPTLLM initialized with model: {self.model_name}")
        logger.info(f"Using endpoint: {self.endpoint}")
    
    def get_request_count(self) -> int:
        """Get the number of API requests made by this instance."""
        return self.request_count
    
    def reset_request_count(self) -> None:
        """Reset the request counter."""
        self.request_count = 0
    
    def create_chat_completion(
        self, 
        messages: List[Dict[str, Any]], 
        coordinator: Optional[Any] = None, 
        max_iterations: int = 10,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 20,
        max_completion_tokens: int = 512,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call the VNPT LLM API with the given messages.
        
        Args:
            messages: List of message dicts in format [{"role": "user", "content": "..."}, ...]
            coordinator: Optional function calling coordinator for tool execution
            max_iterations: Maximum number of function calling iterations
            temperature: Sampling temperature (default: 1.0)
            top_p: Top-p sampling parameter (default: 1.0)
            top_k: Top-k sampling parameter (default: 20)
            max_completion_tokens: Maximum tokens in completion (default: 512)
            **kwargs: Additional parameters for API call
        
        Returns:
            Dict with 'content' key containing the response text
        """
        try:
            # If coordinator is provided, implement function calling loop
            if coordinator:
                return self._execute_with_function_calling(
                    messages, coordinator, max_iterations,
                    temperature, top_p, top_k, max_completion_tokens, **kwargs
                )
            
            # Simple call without function calling
            logger.debug(f"Calling VNPT API with {len(messages)} messages")
            
            # Prepare headers
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Token-id': self.token_id,
                'Token-key': self.token_key,
                'Content-Type': 'application/json',
            }
            
            # Prepare request body
            json_data = {
                'model': self.model_name,
                'messages': messages,
                'temperature': temperature,
                'top_p': top_p,
                'top_k': top_k,
                'n': 1,
                'max_completion_tokens': max_completion_tokens,
                **kwargs
            }
            
            # Retry logic with 30s backoff for 500 errors
            max_retries = 3
            retry_delay = 10  # seconds
            
            for attempt in range(max_retries):
                try:
                    # Increment request counter
                    self.request_count += 1
                    logger.debug(f"VNPT API request #{self.request_count} (model: {self.model_name})")
                    
                    # Call the API
                    response = requests.post(self.endpoint, headers=headers, json=json_data, timeout=60)
                    
                    # Check if it's a server error (5xx) that we should retry
                    if response.status_code >= 500 and attempt < max_retries - 1:
                        logger.warning(
                            f"VNPT API returned {response.status_code} (attempt {attempt + 1}/{max_retries}). "
                            f"Retrying after {retry_delay} seconds..."
                        )
                        time.sleep(retry_delay)
                        continue
                    
                    # Raise for other errors or if this is the last attempt
                    response.raise_for_status()
                    
                    result = response.json()
                    
                    # Extract the response text from VNPT API format
                    if result.get('choices') and len(result['choices']) > 0:
                        content = result['choices'][0].get('message', {}).get('content', '')
                        logger.debug(f"Received response from VNPT API (length: {len(content)})")
                        return {
                            "content": content,
                            "model": self.model_name,
                            "finish_reason": result['choices'][0].get('finish_reason'),
                            "raw_response": result
                        }
                    else:
                        logger.warning("Empty response from VNPT API")
                        return {"content": "No response from API", "model": self.model_name}
                
                except requests.exceptions.HTTPError as http_err:
                    # If it's a 5xx error and we have retries left, continue the loop
                    if hasattr(http_err, 'response') and http_err.response is not None:
                        if http_err.response.status_code >= 500 and attempt < max_retries - 1:
                            logger.warning(
                                f"VNPT API HTTP error {http_err.response.status_code} (attempt {attempt + 1}/{max_retries}). "
                                f"Retrying after {retry_delay} seconds..."
                            )
                            time.sleep(retry_delay)
                            continue
                    # Re-raise if not retryable or last attempt
                    raise
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling VNPT API: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    logger.error(f"API error details: {error_detail}")
                except:
                    logger.error(f"API error response: {e.response.text}")
            raise RuntimeError(f"VNPT API error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error calling VNPT API: {str(e)}")
            raise RuntimeError(f"VNPT API error: {str(e)}")
    
    def _execute_with_function_calling(
        self, 
        messages: List[Dict[str, Any]], 
        coordinator: Any, 
        max_iterations: int,
        temperature: float,
        top_p: float,
        top_k: int,
        max_completion_tokens: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute chat completion with function calling support."""
        iteration = 0
        current_messages = messages.copy()
        
        while iteration < max_iterations:
            iteration += 1
            logger.debug(f"Function calling iteration {iteration}/{max_iterations}")
            
            # Call VNPT API
            response = self.create_chat_completion(
                current_messages,
                coordinator=None,  # Prevent recursion
                max_iterations=1,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_completion_tokens=max_completion_tokens,
                **kwargs
            )
            
            if not response or not response.get("content"):
                logger.warning("Empty response from VNPT API")
                return {"content": "No response from API", "model": self.model_name}
            
            response_text = response["content"]
            
            # Try to parse tool calls from response
            tool_call_parsed = coordinator.parse_tool_call(response_text)
            
            if tool_call_parsed:
                # Execute the tool call
                tool_name = tool_call_parsed.get("tool_name")
                tool_input = tool_call_parsed.get("tool_input", {})
                
                logger.info(f"Executing tool: {tool_name}")
                result = coordinator.execute_function_call(tool_name, tool_input)
                
                # Add assistant message with tool call
                current_messages.append({
                    "role": "assistant", 
                    "content": response_text
                })
                
                # Add tool result as user message
                if result.success:
                    tool_result = json.dumps(result.result, default=str, indent=2)
                else:
                    tool_result = f"Error: {result.error}"
                
                current_messages.append({
                    "role": "user",
                    "content": f"Công cụ '{tool_name}' đã được thực thi. Kết quả: {tool_result}\n\nTiếp tục nhiệm vụ sử dụng thông tin này."
                })
                
                logger.debug(f"Tool execution completed, continuing conversation")
                continue
            
            # No tool calls found, return final response
            logger.info(f"Final response received after {iteration} iterations")
            return {
                "content": response_text,
                "model": self.model_name,
                "finish_reason": response.get("finish_reason"),
                "__debug_loop": {"iterations": iteration}
            }
        
        # Max iterations reached
        logger.warning(f"Max iterations ({max_iterations}) reached")
        return {
            "content": response_text if 'response_text' in locals() else "Max iterations reached",
            "model": self.model_name
        }

