"""Question classification utility to determine if a question is a classification task."""
from typing import List, Dict, Any, Optional
from agent.logging import logger


def is_classification_question(question: str, choices: List[str]) -> bool:
    """
    Determine if a question is a classification task.
    
    Classification questions typically:
    - Ask to categorize, classify, or identify type/category
    - Have keywords like "loại", "phân loại", "thuộc", "thuộc loại", "dạng", "kiểu"
    - Ask "là gì", "là ai", "là loại gì"
    - Ask about types, categories, classifications
    
    Args:
        question: The question text
        choices: List of answer choices
    
    Returns:
        True if the question appears to be a classification task, False otherwise
    """
    question_lower = question.lower()
    
    # Keywords that indicate classification tasks
    classification_keywords = [
        "loại",
        "phân loại",
        "thuộc loại",
        "thuộc",
        "dạng",
        "kiểu",
        "nhóm",
        "hạng",
        "cấp",
        "là loại gì",
        "là dạng gì",
        "là kiểu gì",
        "là nhóm gì",
        "thuộc nhóm",
        "thuộc dạng",
        "thuộc kiểu",
        "phân nhóm",
        "xếp loại",
        "phân hạng",
        "phân cấp"
    ]
    
    # Check if question contains classification keywords
    for keyword in classification_keywords:
        if keyword in question_lower:
            logger.debug(f"Question classified as classification (keyword: '{keyword}')")
            return True
    
    # Check if question starts with "là gì", "là ai", "là loại"
    if question_lower.startswith(("là gì", "là ai", "là loại", "là dạng", "là kiểu")):
        logger.debug("Question classified as classification (starts with 'là...')")
        return True
    
    # Check if choices are very short (often classification tasks have short category names)
    # This is a heuristic - classification choices are often single words or short phrases
    if choices:
        avg_choice_length = sum(len(c) for c in choices) / len(choices)
        if avg_choice_length < 20:  # Average choice length less than 20 characters
            # Additional check: if most choices are single words or very short
            short_choices = sum(1 for c in choices if len(c.split()) <= 3)
            if short_choices >= len(choices) * 0.7:  # 70% of choices are short
                logger.debug("Question classified as classification (short choices heuristic)")
                return True
    
    logger.debug("Question classified as non-classification")
    return False


def classify_question_with_llm(
    question: str,
    choices: List[str],
    llm_client: Any,
    use_llm: bool = False
) -> bool:
    """
    Classify question using LLM if use_llm=True, otherwise use rule-based method.
    
    Args:
        question: The question text
        choices: List of answer choices
        llm_client: LLM client instance (VNPTLLM, GeminiLLM, etc.)
        use_llm: If True, use LLM for classification; if False, use rule-based method
    
    Returns:
        True if classification question, False otherwise
    """
    if not use_llm:
        return is_classification_question(question, choices)
    
    # Use LLM for classification
    try:
        prompt = f"""Phân tích câu hỏi sau và xác định xem đây có phải là câu hỏi PHÂN LOẠI (classification) hay không.

Câu hỏi phân loại là câu hỏi yêu cầu xác định loại, dạng, kiểu, nhóm, hạng, cấp của một đối tượng hoặc hiện tượng.

Câu hỏi: {question}

Lựa chọn:
"""
        for i, choice in enumerate(choices):
            prompt += f"{chr(65 + i)}. {choice}\n"
        
        prompt += "\nTrả lời chỉ bằng 'YES' nếu là câu hỏi phân loại, hoặc 'NO' nếu không phải:"
        
        response = llm_client.create_chat_completion([
            {"role": "user", "content": prompt}
        ])
        
        answer = response.get("content", "").strip().upper()
        is_classification = "YES" in answer or "CÓ" in answer or "PHÂN LOẠI" in answer
        
        logger.debug(f"LLM classification result: {is_classification} (response: {answer})")
        return is_classification
    
    except Exception as e:
        logger.warning(f"Error using LLM for classification: {e}, falling back to rule-based")
        return is_classification_question(question, choices)

