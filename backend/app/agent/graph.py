from langgraph.graph import StateGraph, END
from app.agent.state import AgentState
from app.agent.nodes import (
    agent_entry, # <-- Import the new dummy entry node
    check_for_rag,
    retrieve_from_rag,
    generate_with_context,
    generate_direct,
)

def create_agent_graph():
    """
    Creates the LangGraph agent by defining nodes and the edges between them.
    """
    graph = StateGraph(AgentState)

    # 1. Add all the REAL nodes to the graph
    graph.add_node("agent_entry", agent_entry) # <-- Add our entry node
    graph.add_node("retrieve_from_rag", retrieve_from_rag)
    graph.add_node("generate_with_context", generate_with_context)
    graph.add_node("generate_direct", generate_direct)

    # 2. Set the REAL entry point for the graph
    graph.set_entry_point("agent_entry")

    # 3. Define the conditional routing
    #    The routing decision happens *after* the entry point node.
    graph.add_conditional_edges(
        # The source of the routing is our real entry node
        source="agent_entry",
        # The `path` function decides where to go next
        path=check_for_rag,
        # The `path_map` connects the function's output to the next node
        path_map={
            "rag_retrieval": "retrieve_from_rag",
            "generate_direct": "generate_direct"
        }
    )

    # 4. Define the edges for the RAG path
    graph.add_edge("retrieve_from_rag", "generate_with_context")

    # 5. Define the finish points for both paths
    graph.add_edge("generate_with_context", END)
    graph.add_edge("generate_direct", END)

    # 6. Compile the graph into a runnable executor
    return graph.compile()