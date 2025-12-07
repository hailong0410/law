from typing import TypedDict,List
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    messages: List[BaseMessage]
    context: str # To hold retrive context
    use_rag: bool # Flag to decide if we should use rag or not
    session_id: str
    
    
    