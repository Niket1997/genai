from typing import Annotated, TypedDict
from typing import Annotated, List, Optional
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.mongodb import MongoDBSaver
from models.schemas import OutputSchema
from graph.nodes import chatbot, determine_flow, handle_tool_call


# Define the state type
class State(TypedDict):
    """Graph state definition."""

    messages: Annotated[List, add_messages]
    llm_output: Optional[OutputSchema] = None


# Create the graph
def create_graph(
    llm_with_tools: BaseChatModel, checkpointer: MongoDBSaver
) -> StateGraph:
    """Create and configure the graph.

    Args:
        llm_with_tools: The LLM instance with bound tools

    Returns:
        A compiled graph instance
    """
    graph = StateGraph(State)

    # Add nodes
    graph.add_node("chatbot", lambda state: chatbot(state, llm_with_tools))
    graph.add_node("tools", handle_tool_call)

    # Define edges
    graph.add_edge(START, "chatbot")
    graph.add_conditional_edges(
        "chatbot", determine_flow, {"tools": "tools", "chatbot": "chatbot", END: END}
    )
    graph.add_edge("tools", "chatbot")

    return graph.compile(checkpointer=checkpointer)
