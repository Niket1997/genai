import requests
from typing import Annotated
from langchain_core.tools import InjectedToolCallId, tool
from langsmith import traceable
from langchain_core.messages import AIMessage

from models.schemas import OutputSchema


def generate_tool_response(
    response: str, tool_name: str, tool_input: str, tool_call_id: str
) -> dict:
    """Generate a standardized tool response."""
    return {
        "messages": [
            AIMessage(content=response, name=tool_name, tool_call_id=tool_call_id)
        ],
        "llm_output": OutputSchema(
            step="observe",
            role="tool",
            content=response,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_call_id=tool_call_id,
        ),
    }


@tool
@traceable
def get_weather(city: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> str:
    """Get the current weather information for a specified city.

    Args:
        city: The name of the city to get weather information for
        tool_call_id: The id of the tool call
    """
    print(f"🧰 Tool: Getting weather for {city}")
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)
    if response.status_code != 200:
        return generate_tool_response(
            response="Sorry, I couldn't get the weather information for that city.",
            tool_name="get_weather",
            tool_input=city,
            tool_call_id=tool_call_id,
        )
    return generate_tool_response(
        response=f"The weather in {city} is {response.text}.",
        tool_name="get_weather",
        tool_input=city,
        tool_call_id=tool_call_id,
    )
