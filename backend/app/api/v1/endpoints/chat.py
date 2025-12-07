from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from app.agent.graph import create_agent_graph
import traceback
import logging

logger = logging.getLogger(__name__)

agent_executer = create_agent_graph()

router = APIRouter()


class ChatRequest(BaseModel):
    session_id: str
    message: str
    use_rag: bool = False


@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Endpoint để xử lý chat requests từ frontend.
    """
    try:
        # Validate input
        if not request.message or not request.message.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message cannot be empty"
            )
        
        if not request.session_id or not request.session_id.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Session ID cannot be empty"
            )
        
        logger.info(f"Processing chat request - Session: {request.session_id}, RAG: {request.use_rag}")
        
        # Pass the use_rag flag into the agent's state
        config = {"configurable": {"session_id": request.session_id}}
        inputs = {
            "messages": [HumanMessage(content=request.message)],
            "use_rag": request.use_rag,
            "session_id": request.session_id
        }
        
        # We will use stream for better UX later, but for now invoke works
        response_state = await agent_executer.ainvoke(inputs, config=config)
        
        if not response_state or "messages" not in response_state or len(response_state["messages"]) == 0:
            logger.error("Agent returned empty response")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Agent returned empty response"
            )
        
        last_message = response_state["messages"][-1]
        
        if not hasattr(last_message, 'content') or not last_message.content:
            logger.error("Last message has no content")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Agent response has no content"
            )
        
        logger.info(f"Successfully processed chat request for session: {request.session_id}")
        return {"response": last_message.content}
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log full error details for debugging
        error_traceback = traceback.format_exc()
        logger.error(f"Error during agent execution: {e}\n{error_traceback}")
        print(f"Error during agent execution: {e}")
        print(f"Full traceback:\n{error_traceback}")
        
        # Return proper HTTP error response
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing your request: {str(e)}"
        )


