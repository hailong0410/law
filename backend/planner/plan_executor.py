"""Plan executor utilities.

Contains two orchestrators:
- NonGoalExecutor: runs an open-ended prompt through the LLM adapter and lets
  the adapter + coordinator handle any function-calling loop until a final
  response is returned.

- PlanExecutor: asks the LLM to produce a todo list for a goal, then executes
  each todo item sequentially by calling the LLM for each task, passing the
  original goal and the accumulated results of previous todo executions.

Both orchestrators accept an LLM adapter that implements
`create_chat_completion(messages, coordinator=None, max_iterations=...)`.

The LLM adapter expects tool calls in this format when available:
- If the model needs to call tools, it should return a response with:
  "tool_calls": [{"name": "tool_name", "arguments": {"param": "value"}}]
- When done, return a response without "tool_calls" (or empty list), just "content"
"""
from typing import Any, Dict, List, Optional
import json
import re

from .plan_classifier import classify_prompt
from agent.logging import logger


def _build_tool_instructions(available_tools: List[Dict[str, Any]], collections: Optional[List[str]] = None) -> str:
    """Build instructions for tool calling based on available tools."""
    tools_text = "Các công cụ có sẵn:\n"
    for tool in available_tools:
        func = tool.get("function", {})
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        tools_text += f"\n- {name}: {desc}\n"
        if params.get("properties"):
            tools_text += f"  Tham số: {json.dumps(params.get('properties'), indent=2)}\n"
    
    # Add collections information if available
    if collections:
        tools_text += f"\n\nCác Collection có sẵn:\n"
        for col in collections:
            tools_text += f"- '{col}'\n"
        tools_text += "\nKhi sử dụng 'query_collection', hãy chọn collection phù hợp nhất với chủ đề câu hỏi của người dùng.\n"
    
    tools_text += """

QUAN TRỌNG: Khi bạn cần gọi một công cụ:
1. Trả lời bằng cấu trúc JSON: {"tool_calls": [{"name": "tool_name", "arguments": {"param": "value"}}]}
2. Các 'arguments' phải khớp với tham số của công cụ.
3. Đối với 'query_collection', bạn PHẢI chỉ định collection_name tồn tại trong các collection có sẵn.
4. Bạn có thể gọi nhiều công cụ trong một phản hồi (liệt kê chúng trong mảng tool_calls).
5. Sau khi kết quả công cụ được cung cấp, bạn có thể tiếp tục giải quyết nhiệm vụ.
6. Khi bạn có câu trả lời cuối cùng và không cần thêm công cụ, trả lời bình thường chỉ với "content" (không có tool_calls).
"""
    return tools_text


NON_PLAN_SYSTEM_PROMPT = (
    """
Bạn là một trợ lý AI thông minh. Trả lời câu hỏi của người dùng một cách chính xác và chi tiết.

QUAN TRỌNG - ĐỐI VỚI CÂU HỎI TÍNH TOÁN (toán học, số học, tài chính, công thức, phương trình, phép tính):
- KHÔNG được tìm kiếm hay xây dựng câu trả lời dựa trên kiến thức.
- BẮT BUỘC phải sử dụng tool 'execute_python_code' để thực thi Python code và tính toán.
- Viết Python code rõ ràng, đầy đủ để thực hiện tính toán:
  * Sử dụng thư viện math cho các phép toán (sin, cos, sqrt, log, exp, pow, etc.)
  * Sử dụng numpy nếu có sẵn cho tính toán phức tạp
  * In kết quả bằng print() để trả về
- KHÔNG được chỉ mô tả cách tính, PHẢI viết code Python thực tế và gọi execute_python_code.
- Ví dụ: Câu hỏi "Tính 400000 / 1000000" → BẮT BUỘC gọi execute_python_code với code: "result = 400000 / 1000000\nprint(result)"
- Ví dụ phức tạp: "Tính căn bậc hai của (400000 / 1000000)" → BẮT BUỘC gọi execute_python_code với code: "import math\nvalue = 400000 / 1000000\nresult = math.sqrt(value)\nprint(result)"
- Sau khi có kết quả từ Python code, sử dụng kết quả đó để trả lời câu hỏi.

Đối với câu hỏi không phải tính toán:
- Câu hỏi của người dùng không cần lập kế hoạch từng bước. Hãy trả lời câu hỏi hoặc thực hiện nhiệm vụ trực tiếp.
- Nếu bạn cần gọi một hàm/công cụ, sử dụng giao diện function-calling khi cần.
- Khi bạn cần thực hiện một hành động, yêu cầu công cụ phù hợp (retrieve_documents, etc.).
- Cung cấp các tham số chính xác cho mỗi lần gọi công cụ.
- Sử dụng kết quả công cụ để tạo câu trả lời cuối cùng.
- Ngừng gọi công cụ khi bạn có đủ thông tin để trả lời câu hỏi của người dùng.
- Khi hoàn thành, trả về câu trả lời cuối cùng.
"""
)

PLAN_SYSTEM_PROMPT = (
    """
Bạn là một người lập kế hoạch và thực thi chuyên nghiệp. Đầu tiên, chia mục tiêu của người dùng thành danh sách công việc (dưới dạng mảng JSON các đối tượng có 'id' và 'task'). Sau đó, đối với mỗi công việc, thực hiện nhiệm vụ, sử dụng function calls nếu cần, và trả về kết quả. Mỗi công việc nên nhận kết quả của các công việc trước đó làm ngữ cảnh.

Giai đoạn lập kế hoạch:
- Trả lời bằng mảng JSON: [{"id": "1", "task": "mô tả"}, {"id": "2", "task": "mô tả"}, ...]
- Mỗi nhiệm vụ phải là nguyên tử và có thể thực thi được.
- Chỉ trả về mảng JSON, không có văn bản khác.

Giai đoạn thực thi của mỗi công việc:
- Thực hiện mô tả nhiệm vụ từng bước.
- Gọi công cụ khi cần để hoàn thành nhiệm vụ.
- Khi công cụ trả về kết quả, phân tích chúng và tiến tới hoàn thành nhiệm vụ.
- Khi nhiệm vụ hoàn thành (hoặc bạn đã làm hết sức), trả về tóm tắt ngắn gọn (không có tool_calls).

ĐỐI VỚI CÂU HỎI TÍNH TOÁN (toán học, số học, tài chính, công thức, phương trình, phép tính):
- QUAN TRỌNG: KHÔNG được tìm kiếm hay xây dựng câu trả lời. BẮT BUỘC phải sử dụng tool 'execute_python_code' để thực thi Python code.
- Mỗi bước tính toán phải là một công việc riêng biệt trong kế hoạch.
- Mỗi công việc tính toán PHẢI gọi tool 'execute_python_code' với mã Python để tính toán.
- Viết Python code rõ ràng, đầy đủ cho từng bước:
  * Sử dụng thư viện math cho các phép toán (sin, cos, sqrt, log, exp, pow, etc.)
  * Sử dụng numpy nếu có sẵn cho tính toán phức tạp
  * In kết quả bằng print() để trả về
  * Có thể lưu kết quả vào biến để dùng cho bước tiếp theo
- Cố gắng viết code Python tính toán nhiều nhất có thể cho một lần gọi tool.
- KHÔNG được chỉ mô tả cách tính, PHẢI viết code Python thực tế.
- Ví dụ công việc: "Tính 400000 / 1000000" → BẮT BUỘC gọi execute_python_code với code: "result = 400000 / 1000000\nprint(result)"
- Ví dụ công việc phức tạp: "Tính căn bậc hai của (400000 / 1000000)" → BẮT BUỘC gọi execute_python_code với code: "import math\nvalue = 400000 / 1000000\nresult = math.sqrt(value)\nprint(result)"
- Kết quả từ mỗi bước sẽ được truyền cho bước tiếp theo trong kế hoạch.
- Cuối cùng, sử dụng tất cả kết quả tính toán để chọn đáp án đúng.

VÍ DỤ PLAN CHO CÂU HỎI TÍNH TOÁN:
Câu hỏi: "Tính lãi suất kép: vốn ban đầu 1000000, lãi suất 5% mỗi năm, gửi trong 3 năm. Sau đó tính căn bậc hai của số tiền cuối cùng"

Plan JSON phải trả về:
[
  {"id": "1", "task": "Gọi execute_python_code để tính số tiền sau năm 1: 1000000 * (1 + 0.05)"},
  {"id": "2", "task": "Gọi execute_python_code để tính số tiền sau năm 2: kết quả bước 1 * (1 + 0.05)"},
  {"id": "3", "task": "Gọi execute_python_code để tính số tiền sau năm 3: kết quả bước 2 * (1 + 0.05)"},
  {"id": "4", "task": "Gọi execute_python_code để tính căn bậc hai của số tiền cuối cùng: math.sqrt(kết quả bước 3)"}
]

QUAN TRỌNG - Khi thực thi mỗi task:
- VÍ DỤ PLAN CHO CÂU HỎI TÍNH TOÁN:
Câu hỏi: "Tính lãi suất kép: vốn ban đầu 1000000, lãi suất 5% mỗi năm, gửi trong 3 năm. Sau đó tính căn bậc hai của số tiền cuối cùng"

Plan JSON phải trả về:
[
  {"id": "1", "task": "Gọi execute_python_code để tính số tiền sau năm 1: 1000000 * (1 + 0.05) bằng Python code"},
  {"id": "2", "task": "Gọi execute_python_code để tính số tiền sau năm 2: kết quả bước 1 * (1 + 0.05) bằng Python code"},
  {"id": "3", "task": "Gọi execute_python_code để tính số tiền sau năm 3: kết quả bước 2 * (1 + 0.05) bằng Python code"},
  {"id": "4", "task": "Gọi execute_python_code để tính căn bậc hai của số tiền cuối cùng: math.sqrt(kết quả bước 3) bằng Python code"}
]

"""
)
class AutoPlanExecutor:
    """Unified entrypoint: classify prompt and dispatch to plan or non-plan executor."""
    def __init__(
        self, 
        llm_adapter: Any, 
        available_tools: Optional[List[Dict[str, Any]]] = None,
        question_classify: Optional[Any] = None
    ):
        """
        Initialize AutoPlanExecutor.
        
        Args:
            llm_adapter: LLM adapter for answer generation (plan/non-plan execution)
            available_tools: List of available tools
            question_classify: Optional LLM adapter for question classification. 
                             If None, uses llm_adapter for classification.
        """
        logger.info("Initializing AutoPlanExecutor")
        self.llm = llm_adapter
        # Use question_classify if provided, otherwise use llm_adapter
        self.question_classify = question_classify or llm_adapter
        self.plan_executor = PlanExecutor(llm_adapter)
        self.nonge_executor = NonGoalExecutor(llm_adapter)
        self.available_tools = available_tools or []
        logger.debug(f"AutoPlanExecutor initialized with {len(self.available_tools)} available tools")
        if question_classify:
            logger.debug(f"Using separate LLM for classification: {getattr(question_classify, 'model_name', 'unknown')}")

    def run(
        self, 
        user_prompt: str, 
        coordinator: Optional[Any] = None, 
        max_iterations: int = 10,
        available_tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Classify and dispatch to plan or non-plan executor."""
        tools_to_use = available_tools or self.available_tools
        logger.info("[AUTO PLAN EXECUTOR] Classifying prompt to determine execution mode (plan or non-plan)")
        mode = classify_prompt(self.question_classify, user_prompt)
        logger.info(f"[AUTO PLAN EXECUTOR] Prompt classified as: '{mode}' (type: {type(mode)})")
        logger.info(f"[AUTO PLAN EXECUTOR] Checking if mode == 'plan': {mode == 'plan'}")
        
        if mode == "plan":
            logger.info("[AUTO PLAN EXECUTOR] Dispatching to PlanExecutor")
            return self.plan_executor.execute_plan(
                user_prompt,
                coordinator=coordinator,
                system_prompt=PLAN_SYSTEM_PROMPT,
                max_iterations=max_iterations,
                available_tools=tools_to_use,
            )
        else:
            logger.info(f"[AUTO PLAN EXECUTOR] Dispatching to NonGoalExecutor (mode was: '{mode}')")
            return self.nonge_executor.run(
                user_prompt,
                coordinator=coordinator,
                system_prompt=NON_PLAN_SYSTEM_PROMPT,
                max_iterations=max_iterations,
                available_tools=tools_to_use,
            )


class NonGoalExecutor:
    """Execute an open-ended prompt via the LLM adapter and (optionally)
    a coordinator for tool/function calls.

    The LLM adapter is expected to perform function-calling loops (e.g., the
    GeminiLLM adapter) when `coordinator` is provided. The executor simply
    prepares messages and returns the final model output.
    """

    def __init__(self, llm_adapter: Any):
        self.llm = llm_adapter

    def run(
        self, 
        prompt: str, 
        coordinator: Optional[Any] = None, 
        system_prompt: Optional[str] = None, 
        max_iterations: int = 10,
        available_tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Run the prompt and return the LLM's final response dict.

        Returns a dict containing at least:
        - content: str (final assistant content)
        - __debug_loop: optional debug history provided by some adapters
        """
        logger.info("NonGoalExecutor starting execution")
        messages = []
        if system_prompt:
            # Append tool instructions if tools available
            full_system = system_prompt
            if available_tools:
                # Get collections from coordinator if available
                collections = None
                if coordinator and hasattr(coordinator, 'get_available_collections'):
                    collections = coordinator.get_available_collections()
                full_system += "\n\n" + _build_tool_instructions(available_tools, collections=collections)
            messages.append({"role": "system", "content": full_system})
        messages.append({"role": "user", "content": prompt})

        logger.debug(f"Calling LLM with {len(messages)} messages")
        response = self.llm.create_chat_completion(messages, coordinator=coordinator, max_iterations=max_iterations)
        
        # Log raw response from LLM
        if isinstance(response, dict):
            content = response.get("content", "")
            logger.info(f"Raw LLM response (NonGoalExecutor):\n{content}\n{'='*80}")
        else:
            logger.info(f"Raw LLM response (NonGoalExecutor):\n{str(response)}\n{'='*80}")
        
        logger.info("NonGoalExecutor execution completed")
        
        if isinstance(response, dict):
            return response
        return {"content": str(response)}


class PlanExecutor:
    """Create and execute a todo-style plan for a goal using the LLM.

    Workflow:
    1. Ask the LLM to return a todo list for the provided goal (as JSON array of strings or objects).
    2. Parse the todo list.
    3. For each todo in order, invoke the LLM with the original goal, the todo
       instructions, and the accumulated results of previous todos. Collect
       each todo result and pass them forward.

    This executor uses the LLM adapter directly (not the coordinator) to run
    each todo. If your environment requires tool/function calls during todo
    execution, pass a coordinator to the adapter so it can perform them.
    """

    def __init__(self, llm_adapter: Any):
        self.llm = llm_adapter

    def _ask_for_todos(
        self, 
        goal: str, 
        system_prompt: Optional[str] = None, 
        max_iterations: int = 6,
        available_tools: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        logger.info("Asking LLM to create todo list for goal")
        # Instruct the model to return a JSON array with clear format requirements
        json_format_example = """[
  {"id": "1", "task": "Task description 1"},
  {"id": "2", "task": "Task description 2"},
  {"id": "3", "task": "Task description 3"}
]"""
        sp = (system_prompt or "") + f"\n\nQUAN TRỌNG: Bạn PHẢI trả lời CHỈ bằng một mảng JSON hợp lệ, không có văn bản nào khác.\n\nFormat bắt buộc:\n{json_format_example}\n\nYêu cầu:\n- Mỗi phần tử là một object có 'id' (string hoặc number) và 'task' (string)\n- Không có văn bản giải thích, không có markdown code block, chỉ JSON thuần\n- JSON phải hợp lệ và có thể parse được ngay"
        messages = [{"role": "system", "content": sp}, {"role": "user", "content": f"Tạo danh sách công việc để đạt được mục tiêu này:\n\n{goal}"}]
        resp = self.llm.create_chat_completion(messages, max_iterations=max_iterations)
        content = resp.get("content") if isinstance(resp, dict) else str(resp)
        
        # Log raw response from LLM
        logger.info(f"Raw LLM response for todos:\n{content}\n{'='*80}")

        # Try to parse JSON - if fails, raise error (no fallback)
        try:
            # Try to extract JSON from markdown code block if present
            json_content = content
            if "```json" in content:
                json_content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_content = content.split("```")[1].split("```")[0].strip()
            else:
                # If no markdown blocks, try to extract JSON array from content
                # Find the first '[' and match it with the corresponding ']'
                start_idx = json_content.find('[')
                if start_idx != -1:
                    # Count brackets to find matching closing bracket
                    bracket_count = 0
                    end_idx = start_idx
                    for i in range(start_idx, len(json_content)):
                        if json_content[i] == '[':
                            bracket_count += 1
                        elif json_content[i] == ']':
                            bracket_count -= 1
                            if bracket_count == 0:
                                end_idx = i + 1
                                break
                    if end_idx > start_idx:
                        json_content = json_content[start_idx:end_idx].strip()
                else:
                    # If no '[' found, try to remove HTML tags and parse
                    # Remove HTML tags
                    json_content = re.sub(r'<[^>]+>', '', json_content)
                    json_content = json_content.strip()
            
            parsed = json.loads(json_content)
            
            if not isinstance(parsed, list):
                raise ValueError(f"Expected JSON array, got {type(parsed).__name__}")
            
            todos = []
            for i, item in enumerate(parsed, start=1):
                if isinstance(item, str):
                    todos.append({"id": str(i), "task": item})
                elif isinstance(item, dict):
                    if "task" not in item:
                        raise ValueError(f"Todo item {i} missing 'task' field: {item}")
                    todo_id = item.get("id", str(i))
                    todos.append({"id": str(todo_id), "task": item.get("task")})
                else:
                    raise ValueError(f"Invalid todo item type at index {i}: {type(item).__name__}")
            
            if not todos:
                raise ValueError("Parsed JSON array is empty")
            
            logger.info(f"Successfully parsed {len(todos)} todos from JSON response")
            return todos
            
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse JSON from LLM response. JSON decode error: {str(e)}\nRaw response:\n{content}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to parse todos from LLM response: {str(e)}\nRaw response:\n{content}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

    def execute_plan(
        self, 
        goal: str, 
        coordinator: Optional[Any] = None, 
        system_prompt: Optional[str] = None, 
        max_iterations: int = 6,
        available_tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Create a todo list for `goal` and execute each todo sequentially.

        Args:
            goal: high-level goal string
            coordinator: optional coordinator to allow function/tool calls within LLM executions
            system_prompt: optional system-level instructions
            max_iterations: max iterations for the adapter per call
            available_tools: list of available tool schemas for the LLM

        Returns:
            dict with keys: 'goal', 'todos' (list of todos with results), 'aggregated' (combined result)
        """
        logger.info(f"Starting plan execution for goal: {goal[:100]}..." if len(goal) > 100 else f"Starting plan execution for goal: {goal}")
        todos = self._ask_for_todos(goal, system_prompt=system_prompt, max_iterations=max_iterations, available_tools=available_tools)

        results = []
        accumulated_context = []  # list of dicts {'id','task','result'}

        for i, todo in enumerate(todos, start=1):
            logger.info(f"Executing todo {i}/{len(todos)}: {todo['task'][:80]}..." if len(todo['task']) > 80 else f"Executing todo {i}/{len(todos)}: {todo['task']}")
            task_prompt = f"Mục tiêu: {goal}\n\nNhiệm vụ: {todo['task']}\n\nKết quả trước đó:\n{json.dumps(accumulated_context, indent=2)}\n\nVui lòng thực hiện nhiệm vụ và trả về kết quả ngắn gọn (dưới dạng văn bản hoặc JSON)."
            messages = []
            if system_prompt:
                # Append tool instructions for this todo execution
                full_system = system_prompt
                if available_tools:
                    # Get collections from coordinator if available
                    collections = None
                    if coordinator and hasattr(coordinator, 'get_available_collections'):
                        collections = coordinator.get_available_collections()
                    full_system += "\n\n" + _build_tool_instructions(available_tools, collections=collections)
                messages.append({"role": "system", "content": full_system})
            messages.append({"role": "user", "content": task_prompt})

            # Use the adapter and allow coordinator-based function calls if provided
            # The GeminiLLM adapter will loop: if tool_calls returned, execute and re-invoke
            # Continue until LLM returns final response without tool_calls
            logger.debug(f"Calling LLM for todo {todo['id']}")
            resp = self.llm.create_chat_completion(messages, coordinator=coordinator, max_iterations=max_iterations)
            content = resp.get("content") if isinstance(resp, dict) else str(resp)

            # Save result and feed forward
            entry = {"id": todo["id"], "task": todo["task"], "result": content}
            results.append(entry)
            accumulated_context.append(entry)
            logger.debug(f"Todo {todo['id']} completed with result length: {len(content)} chars")

        # Optionally synthesize aggregated result
        aggregated = "\n".join([f"[{r['id']}] {r['task']}: {r['result']}" for r in results])
        logger.info(f"Plan execution completed. Processed {len(todos)} todos")
        return {"goal": goal, "todos": results, "aggregated": aggregated}
