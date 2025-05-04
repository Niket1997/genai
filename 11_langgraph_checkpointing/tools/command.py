import os
from typing import Annotated
from langchain_core.tools import InjectedToolCallId, tool
from langsmith import traceable

from tools.weather import generate_tool_response
from langgraph.types import interrupt


@tool
@traceable
def execute_command(
    command: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """Execute a system command and return its output. The command should be a valid system command.
    For example, you want to create a file, you can use the command "touch file.txt" to create a file named file.txt.

    Args:
        command: The system command to execute
        tool_call_id: The id of the tool call
    """
    print(f"🧰 Tool: Executing command {command}")
    human_response = interrupt({"query": f"Execute the command: {command}"})
    
    # Analyze the response from interrupt
    if human_response and isinstance(human_response, dict):
        response_text = human_response.get("response", "").lower().strip()
        if response_text in ["yes", "y"]:
            response = os.system(command)
            return generate_tool_response(
                response=f"The command {command} returned {response}.",
                tool_name="execute_command",
                tool_input=command,
                tool_call_id=tool_call_id,
            )
        elif response_text in ["no", "n"]:
            return generate_tool_response(
                response=f"Command execution cancelled by user.",
                tool_name="execute_command",
                tool_input=command,
                tool_call_id=tool_call_id,
            )
    
    return generate_tool_response(
        response=f"The command {command} was interrupted.",
        tool_name="execute_command",
        tool_input=command,
        tool_call_id=tool_call_id,
    )
