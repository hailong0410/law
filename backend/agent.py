"""Main RAG Agent implementation with optional planner integration.

This file provides a single, clean `RAGAgent` class that integrates with the
planner (`AutoPlanExecutor`) when an LLM adapter is available. The previous
version contained nested/duplicated class definitions; this patch replaces it
with a straightforward, maintainable implementation.
"""

from typing import Optional, Dict, Any, List
import json

from agent.function_calling.coordinator import FunctionCallingCoordinator
from agent.memory import VectorDatabase
from agent.logging import logger


class RAGAgent:
    """RAG (Retrieval-Augmented Generation) Agent with function calling and optional planning.

    The agent will prefer an explicitly provided `llm_client`. If not provided,
    it will attempt to use `get_llm` from `agent.llm.manager` when `llm_type` or
    `llm_config` are provided.
    """

    def __init__(
        self,
        vector_db: Optional[Any] = None,
        coordinator: Optional[Any] = None,
        llm_client: Optional[Any] = None,
        llm_type: Optional[str] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        enable_planner: bool = True,
        use_conversation_history: bool = True,
    ) -> None:
        logger.info("Initializing RAGAgent...")
        # Initialize vector DB and coordinator
        if vector_db is None:
            vector_db = VectorDatabase()
            logger.debug("Created new VectorDatabase instance")
        self.vector_db = vector_db

        self.coordinator = coordinator or FunctionCallingCoordinator(self.vector_db)
        logger.debug("FunctionCallingCoordinator initialized")

        # Resolve LLM (prefer explicit llm_client)
        self.llm = llm_client
        if self.llm is None:
            try:
                from agent.llm.manager import get_llm

                # Only attempt to create an LLM if some selector/config is provided
                if llm_type or llm_config:
                    logger.info(f"Creating LLM with type: {llm_type or 'gemini'}")
                    self.llm = get_llm(llm_type or "gemini", llm_config or {})
                    logger.info("LLM created successfully")
            except Exception as e:
                # Keep self.llm as None if we cannot import or construct an LLM
                logger.warning(f"Failed to create LLM: {str(e)}")
                self.llm = None

        # Optionally wire in the planner (if LLM is present and planner module exists)
        self.enable_planner = enable_planner if self.llm is not None else False
        self.planner = None
        if self.enable_planner and self.llm is not None:
            try:
                from agent.planner.plan_executor import AutoPlanExecutor
                from agent.llm.manager import get_llm

                logger.info("Initializing AutoPlanExecutor for planner")
                # Pass available tools to the planner so it can include tool instructions in prompts
                available_tools = self.coordinator.get_available_tools()
                
                # Get question_classify LLM from config if available
                question_classify = None
                if llm_config and "question_classify" in llm_config:
                    question_classify_config = llm_config.get("question_classify", {})
                    question_classify_type = question_classify_config.get("type", llm_type or "vnpt")
                    logger.info(f"Creating question_classify LLM with type: {question_classify_type}")
                    question_classify = get_llm(question_classify_type, question_classify_config)
                
                self.planner = AutoPlanExecutor(
                    self.llm, 
                    available_tools=available_tools,
                    question_classify=question_classify
                )
                logger.info("Planner initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize planner: {str(e)}")
                self.planner = None

        self.conversation_history: List[Dict[str, Any]] = []
        self.use_conversation_history = use_conversation_history
        self.max_iterations = 10
        logger.info(f"RAGAgent initialization completed (use_conversation_history={use_conversation_history})")

    def chat(self, user_message: str, system_prompt: Optional[str] = None, use_planner: Optional[bool] = None) -> str:
        """Process a user message and generate a response.

        If a planner is enabled (either at construction or via the per-call
        `use_planner=True`), the agent will dispatch the message to the planner
        which will classify and run either a plan flow or a regular non-goal
        execution. When the planner is not used or not available, the message
        is passed to the LLM adapter with the function-calling coordinator.

        Returns a string (final assistant content or aggregated plan result).
        """
        logger.info(f"User message received: {user_message[:100]}..." if len(user_message) > 100 else f"User message received: {user_message}")
        self.conversation_history.append({"role": "user", "content": user_message})

        # Decide whether to use planner for this call
        call_planner = self.enable_planner if use_planner is None else bool(use_planner)
        logger.info(f"[RAG AGENT] Planner check: enable_planner={self.enable_planner}, use_planner={use_planner}, call_planner={call_planner}, planner is None={self.planner is None}")

        if call_planner and self.planner is not None:
            logger.info("[RAG AGENT] Using planner for message processing")
            # Planner returns a dict; normalize to string output
            result = self.planner.run(user_message, coordinator=self.coordinator, max_iterations=self.max_iterations)
            # The planner may return different keys depending on the path
            if isinstance(result, dict):
                content = result.get("content") or result.get("aggregated") or json.dumps(result)
            else:
                content = str(result)

            self.conversation_history.append({"role": "assistant", "content": content})
            logger.info("Response generated via planner")
            return content

        # If no LLM configured, return helper listing tools
        if not self.llm:
            logger.warning("No LLM configured, returning available tools")
            tools = self.coordinator.get_available_tools()
            return (
                f"No LLM configured. Available tools:\n{json.dumps(tools, indent=2)}\n"
                "Configure an LLM via llm_type/llm_config or provide a custom llm_client."
            )

        # Prepare messages and delegate to the LLM adapter which will handle function calls
        logger.debug("Processing message with LLM and function calling coordinator")
        messages = self._prepare_messages(system_prompt)
        response = self.llm.create_chat_completion(messages, coordinator=self.coordinator, max_iterations=self.max_iterations)

        final_response = response.get("content", "") if isinstance(response, dict) else str(response)
        self.conversation_history.append({"role": "assistant", "content": final_response})
        logger.info("Response generated via LLM")
        return final_response

    def _prepare_messages(self, system_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt + self._get_tool_instructions()})
        else:
            messages.append({"role": "system", "content": self._get_default_system_prompt()})
        
        # Only include conversation history if enabled
        if self.use_conversation_history:
            messages.extend(self.conversation_history)
        else:
            # If disabled, only include the last user message (current question)
            # This ensures each question is independent
            if self.conversation_history:
                last_user_msg = None
                for msg in reversed(self.conversation_history):
                    if msg.get("role") == "user":
                        last_user_msg = msg
                        break
                if last_user_msg:
                    messages.append(last_user_msg)
        
        return messages

    def _get_default_system_prompt(self) -> str:
        tools_info = "\n".join([f"- {tool['function']['name']}: {tool['function']['description']}" for tool in self.coordinator.get_available_tools()])
        
        # Get available collections from vector database
        collections_info = ""
        try:
            collections = self.vector_db.list_collections()
            if collections:
                collections_info = f"\n\nCác Collection có sẵn trong Database:\n"
                for col in collections:
                    # Get document count for each collection
                    try:
                        docs = self.vector_db.retrieve_by_collection(col)
                        doc_count = len(docs) if docs else 0
                        collections_info += f"- '{col}': {doc_count} tài liệu\n"
                    except:
                        collections_info += f"- '{col}'\n"
                collections_info += "\nKhi truy vấn tài liệu, hãy chọn collection phù hợp nhất dựa trên câu hỏi của người dùng.\n"
                collections_info += "Nếu không chắc chắn, bạn có thể truy vấn nhiều collections hoặc sử dụng 'retrieve_documents' để tìm kiếm trên tất cả collections.\n"
        except Exception as e:
            logger.debug(f"Could not retrieve collections info: {str(e)}")
        
        return (
            f"Bạn là một trợ lý RAG (Retrieval-Augmented Generation) hữu ích. \n"
            f"Bạn có quyền truy cập các công cụ sau:\n\n{tools_info}\n"
            f"{collections_info}\n"
            "Khi bạn cần lấy thông tin hoặc thực hiện hành động, hãy gọi các công cụ phù hợp.\n"
            "QUAN TRỌNG: Khi truy vấn một collection cụ thể, sử dụng 'query_collection' với tên collection phù hợp nhất với chủ đề câu hỏi của người dùng."
        )

    def _get_tool_instructions(self) -> str:
        return """\n\nAvailable tools and their schemas:\n""" + json.dumps(self.coordinator.get_available_tools(), indent=2)

    def reset_conversation(self) -> None:
        logger.info("Resetting conversation history")
        self.conversation_history = []

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        logger.debug("Retrieving conversation history")
        return self.conversation_history.copy()


if __name__ == "__main__":
    # Small demo when run as a script
    agent = RAGAgent()
    print("Available Tools:")
    print("=" * 50)
    for tool in agent.coordinator.get_available_tools():
        func = tool["function"]
        print(f"\n{func['name']}")
        print(f"Description: {func['description']}")
        print(f"Parameters: {func['parameters']}")

    # If user has configured an LLM via agent.llm.manager, this shows planner usage
    try:
        from agent.llm.manager import get_llm

        llm = get_llm("gemini", {})
        agent_with_llm = RAGAgent(llm_client=llm)
        print("\nCreated agent with LLM. Chat example (planner enabled):")
        print(agent_with_llm.chat("Create a plan to learn Python in 3 months"))
    except Exception:
        print("No LLM available for the demo. Install/configure an LLM to try planner flows.")
