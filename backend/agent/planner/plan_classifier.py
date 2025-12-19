"""LLM-based classifier to decide if a prompt should be handled as a plan or non-plan.

The classifier asks the LLM to return 'plan' if the input requires stepwise planning (multi-step, goal, project, research, etc.), or 'non-plan' if it is a direct question, open-ended, or exploratory.
"""
from typing import Any, Literal
from agent.logging import logger


PLAN_CLASSIFIER_PROMPT = (
    """
    Bạn là một bộ phân loại prompt cho một AI agent. Nhiệm vụ của bạn là quyết định xem đầu vào của người dùng có cần lập kế hoạch từng bước (danh sách công việc, kế hoạch nhiều bước, hoặc phân tích dự án) hay có thể xử lý trực tiếp (như một câu hỏi đơn, khám phá mở, hoặc lời gọi hàm trực tiếp).

    QUAN TRỌNG:
    - Nếu câu hỏi có phạm trù tính toán (toán học, số học, tính toán tài chính, công thức, phương trình, v.v.), BẮT BUỘC phải trả lời: plan
    - Nếu đầu vào là mục tiêu, dự án, hoặc nhiệm vụ phức tạp cần được chia thành các bước, trả lời: plan
    - Nếu đầu vào là một câu hỏi dài gồm 3 đến 4 câu trở lên và nhiều ý chi tiết, trả lời: plan
    - Nếu đầu vào là câu hỏi trực tiếp, prompt mở, hoặc có thể được trả lời/thực thi trong một bước, trả lời: non-plan
    - Chú ý là phân loại rất quan trọng.
    - Chỉ trả lời bằng một từ: plan hoặc non-plan. Không thêm giải thích.
    """
)

def classify_prompt(llm_adapter: Any, user_prompt: str, system_prompt: str = PLAN_CLASSIFIER_PROMPT) -> Literal["plan", "non-plan"]:
    """Use the LLM to classify the prompt as 'plan' or 'non-plan'."""
    # Identify which LLM adapter is being used for logging
    adapter_name = getattr(llm_adapter, 'model_name', getattr(llm_adapter, '__class__', {}).__name__ if hasattr(llm_adapter, '__class__') else 'unknown')
    logger.info(f"[PLAN CLASSIFIER] Starting classification with adapter: {adapter_name}")
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Phân loại đầu vào này: {user_prompt}"}
    ]
    resp = llm_adapter.create_chat_completion(messages, max_iterations=2)
    content = resp.get("content") if isinstance(resp, dict) else str(resp)
    raw_content = content.strip()
    content = raw_content.lower()
    
    # Log raw response from LLM
    logger.info(f"[PLAN CLASSIFIER] Question: {user_prompt[:100]}...")
    logger.info(f"[PLAN CLASSIFIER] LLM raw response: {raw_content}")
    
    if "plan" in content and "non-plan" not in content:
        result = "plan"
        logger.info(f"[PLAN CLASSIFIER] Decision: {result} (matched 'plan' in response) - Adapter: {adapter_name}")
        return result
    if "non-plan" in content:
        result = "non-plan"
        logger.info(f"[PLAN CLASSIFIER] Decision: {result} (matched 'non-plan' in response) - Adapter: {adapter_name}")
        return result
    # fallback: if ambiguous, default to non-plan
    result = "non-plan"
    logger.warning(f"[PLAN CLASSIFIER] Decision: {result} (fallback - ambiguous response: {raw_content}) - Adapter: {adapter_name}")
    return result
