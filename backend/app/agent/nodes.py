# from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from app.services import vector_store
from app.tools import web_search
from langchain_core.messages import HumanMessage
from app.core.config import settings

from app.agent.state import AgentState
import os 
from dotenv import load_dotenv


load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")



llm_gemini = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL"), temperature=0)
# llm_groq = ChatGroq(model=os.getenv("GROQ_MODEL"), temperature=0)

llm_gemini_rag = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL"),
    temperature=0.7,
    convert_system_message_to_human=True # Important for Gemini
)


def route_to_llm(state: AgentState) -> str:
    """According to the prompt decide which LLm to use"""
    lastmessage = state["messages"][-1].content
    
    if "creative" in lastmessage.lower() or "write" in lastmessage.lower():
        return "gemini"
    
    if "code" in lastmessage.lower() or "python" in lastmessage.lower():
        return "groq"
    
    return "ginini"



# def generate_gemini(state: AgentState):
#     """Node to call Gimini."""
#     human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)] 
#     response = llm_gemini.invoke(human_messages)
#     return {"messages": [response]}

def generate_gemini(state: AgentState):
    last_msg = state["messages"][-1].content.strip()

    if not last_msg:
        raise ValueError("Last message content is empty — cannot invoke Gemini.")

    # Ensure proper format: list of message objects
    messages = [HumanMessage(content=last_msg)]
    
    response = llm_gemini.invoke(messages)
    return {"messages": [response]}

# def generate_groq(state: AgentState):
#     """Node to call Groq."""
#     response = llm_groq.invoke(state["messages"])
#     return {"messages": [response]}


# ---- similar for other model






# def check_for_rag(state: AgentState):
#     """Decides which tool to use, if any."""
#     last_message = state["messages"][-1].content.lower()
    
#     # Priority 1: Check for explicit RAG flag
#     if state.get("use_rag", False):
#         print("---ROUTING: RAG---")
#         return "rag_retrieval"
    
#     # Priority 2: Check for keywords implying web search
#     search_keywords = ["latest", "what is the current", "search for", "tavily"]
#     if any(keyword in last_message for keyword in search_keywords):
#         print("---ROUTING: Web Search---")
#         return "web_search"
    
#     # Default: No tool needed, go to the LLM router
#     print("---ROUTING: Direct to LLM---")
#     return "__end__" # LangGraph convention for routing to another router

def check_for_rag(state: AgentState):
    """
    Checks the 'use_rag' flag passed from the frontend.
    This node acts as a router to decide the agent's path.
    Its string return value is used by the conditional edge router.
    """
    print(f"--- ROUTER: Checking for RAG. Flag is: {state.get('use_rag')} ---")
    if state.get("use_rag", False):
        # The string "rag_retrieval" will be used for routing.
        # The node itself returns an empty dictionary, as it doesn't modify state.
        return "rag_retrieval"
    else:
        # The string "generate_direct" will be used for routing.
        return "generate_direct"
    
    
    
def retrieve_from_rag(state: AgentState):
    """
    Retrieves relevant documents from the ChromaDB vector store,
    filtered by the session_id from the agent's state.
    """
    
    print("Retrieving context from RAG...")
    user_query = state["messages"][-1].content
    session_id = state.get("session_id")
    
    if not session_id:
        print("Warning: No session_id found in state for RAG retrieval.")
        return {"context": "Error: No session ID provided for document retrieval."}
    
    retriever = vector_store.get_retriever(session_id=session_id)
    
    retrieved_docs = retriever.invoke(user_query)
    
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    print(f"Retrieved context for session {session_id}: {context[:300]}...")
    
    return {"context": context}

# def generate_direct(state: AgentState):
#     """Node to call the primary LLM directly without context."""
#     print("Generating direct response...")
#     # Using Gemini Pro as the default direct LLM
#     response = llm_gemini.invoke(state["messages"])
#     return {"messages": [response]}

def run_web_search(state: AgentState):
    """Node to perform web search."""
    search_tool = web_search.get_web_search_tool()
    search_results = search_tool.invoke({"query": state["messages"][-1].content})
    return {"context": search_results}

def generate_with_context(state: AgentState):
    """
    Generates a response using the LLM with the retrieved context.
    This is the final step in the RAG path.
    """
    print("--- NODE: Generating response with context ---")
    user_query = state["messages"][-1].content
    context = state.get("context", "")

    # --- NEW, IMPROVED PROMPT ---
    prompt = f"""
    Bạn là một trợ lý AI có nhiệm vụ trả lời câu hỏi dựa trên dữ liệu được cung cấp. 
    Bạn phải tuân thủ các quy tắc sau để đảm bảo tính chính xác và an toàn thông tin.

    MỤC TIÊU:
    Cung cấp câu trả lời đúng, súc tích và hoàn toàn dựa trên nguồn dữ liệu được cung cấp.

    QUY TẮC BẮT BUỘC:
    1. Chỉ sử dụng thông tin trong "DOCUMENT SNIPPETS" để trả lời.
    2. Không được sử dụng kiến thức bên ngoài hoặc suy đoán.
    3. Nếu câu hỏi mang tính tổng quan (ví dụ: “tài liệu này nói gì?”, “tóm tắt file”), hãy đưa ra một bản tóm tắt cô đọng.
    4. Nếu câu hỏi yêu cầu thông tin cụ thể, hãy trả lời trực tiếp bằng nội dung trích dẫn.
    5. Nếu tài liệu không chứa thông tin cần thiết, hãy trả lời:
    → “Tài liệu không cung cấp thông tin để trả lời câu hỏi này.”
    6. Không được bịa thông tin, không được giả định.

    DOCUMENT SNIPPETS:
    ---
    {context}
    ---

    USER QUESTION:
    "{user_query}"

    Hãy đưa ra câu trả lời chính xác và tuân thủ các quy tắc trên.
    """
    # --------------------------

    # We replace the user's last message with this new, enriched prompt
    messages_for_llm = state["messages"][:-1] + [
        ( "user", prompt)
    ]

    response = llm_gemini.invoke(messages_for_llm)
    return {"messages": [response]}

def generate_direct(state: AgentState):
    """
    Generates a response using the LLM directly, without any RAG context.
    This is the standard chat path.
    """
    print("--- NODE: Generating direct response (no RAG) ---")
    response = llm_gemini.invoke(state["messages"])
    return {"messages": [response]}


def agent_entry(state: AgentState):
    """A dummy entry point node that does nothing but start the graph."""
    print("--- AGENT ENTRY ---")
    return {} # It's a valid node because it returns a dictionary