from typing import Dict, Any
from langchain_core.messages import AIMessage
from langgraph.graph import END

from models.schemas import OutputSchema
from tools.weather import get_weather
from tools.command import execute_command

tools = [get_weather, execute_command]


def chatbot(state: Dict[str, Any], llm_with_tools):
    """Chatbot node that processes messages and generates responses."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)

    if isinstance(response, OutputSchema):
        return {
            "messages": [AIMessage(content=response.content)],
            "llm_output": response,
        }

    return {"messages": [response]}


def determine_flow(state: Dict[str, Any]) -> str:
    """Determine the next node in the flow based on the current state."""
    llm_output = state["llm_output"]
    if not llm_output or llm_output.step == "output":
        return END

    if llm_output.step == "action":
        return "tools"

    if llm_output.step in ["plan", "observe"]:
        return "chatbot"
    return END


def handle_tool_call(state: Dict[str, Any]):
    """Handle tool calls based on the LLM output."""
    llm_output = state["llm_output"]
    if not llm_output or llm_output.step != "action":
        return state

    tool_name = llm_output.tool_name
    tool_input = llm_output.tool_input
    tool_call_id = llm_output.tool_call_id

    for tool in tools:
        if tool.name == tool_name:
            # Use the correct parameter name based on the tool
            param_name = "city" if tool_name == "get_weather" else "command"
            tool_call = tool.invoke(
                {param_name: tool_input, "tool_call_id": tool_call_id}
            )
            return {
                "messages": [tool_call["messages"][0]],
                "llm_output": tool_call,
            }

    return state
