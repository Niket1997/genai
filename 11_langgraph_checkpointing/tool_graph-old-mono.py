import json
import os
from typing import Annotated, Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langsmith import traceable
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# load the environment variables
load_dotenv()


# Output Schema
class OutputSchema(BaseModel):
    step: str = Field(
        description="The current step in the process (plan, action, observe, output)"
    )
    role: str = Field(description="The role of the message (user, assistant, tool)")
    content: str = Field(description="The content of the message")
    tool_name: Optional[str] = Field(
        default=None, description="The name of the function if the step is action"
    )
    tool_input: Optional[str] = Field(
        default=None, description="The input of the function if the step is action"
    )
    tool_call_id: Optional[Annotated[str, InjectedToolCallId]] = Field(
        default=None, description="The id of the tool call"
    )


def generate_tool_response(
    response: str, tool_name: str, tool_input: str, tool_call_id: str
) -> Dict[str, Any]:
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


### create the tools
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
    response = os.system(command)
    return generate_tool_response(
        response=f"The command {command} returned {response}.",
        tool_name="execute_command",
        tool_input=command,
        tool_call_id=tool_call_id,
    )


tools = [get_weather, execute_command]


# initialize the llm & bind the tools
llm = init_chat_model(model_provider="openai", model="gpt-4.1")
llm_with_tools = llm.bind_tools(tools).with_structured_output(OutputSchema)


# create graph state
class State(TypedDict):
    messages: Annotated[List, add_messages]
    llm_output: Optional[OutputSchema] = None


## define the nodes in the graph
def chatbot(state: State):
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


def handle_tool_call(state: State):
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


# tool_node = ToolNode(tools)

## define the graph
graph = StateGraph(State)

graph.add_node("chatbot", chatbot)
graph.add_node("tools", handle_tool_call)

# define the edges
graph.add_edge(START, "chatbot")
graph.add_conditional_edges(
    "chatbot", determine_flow, {"tools": "tools", "chatbot": "chatbot", END: END}
)

graph.add_edge("tools", "chatbot")

compiled_graph = graph.compile()


# system prompt
system_prompt = f"""
You are a helpful AI assistant who is specialized in helping users with their questions.
Given a user's query, you analuse the query carefully and generate a step by step plan.
You will be given a list of available tools and you will select the most relevant tool to use whenever needed.
Once the tool is selected, you will perform the action & generate output from the tool.
You will also observe the output from the tool and generate a result. The result should be quirky and funny.
Ensure that these steps plan, action, observe & output are executed one at a time.

Rules:
- Follow the output format strictly.
- You must ensure that the steps plan, action, observe & output are executed one at a time.
- When using tools, you MUST set the step to "action" and provide the tool_name and tool_input.
- You must ensure that you are generating a unique tool_call_id for each tool call.
- Carefully analyze the user query.

Available Tools:
- get_weather: Get weather information for a city
- execute_command: Execute system commands

Example for tool usage:
Input: What is the weather in Tokyo?
{{ "step": "plan", "content": "The user is asking about the weather in Tokyo." }}
{{ "step": "plan", "content": "From the available tools, I will select the get_weather tool to get the weather information of Tokyo." }}
{{ "step": "action", "tool_name": "get_weather", "tool_input": "Tokyo", "tool_call_id": "1234567890" }}
{{ "step": "observe", "content": "The weather in Tokyo is sunny with a temperature of 12 degrees Celsius." }}
{{ "step": "output", "content": "The weather in Tokyo is sunny with a temperature of 12 degrees Celsius. You can go out and enjoy the weather." }}

Example:
Input: write a python file in current directory to add two numbers
{{ "step": "plan", "content": "The user is asking to create a python file to add two numbers." }}
{{ "step": "plan", "content": "I first need to get the current directory." }}
{{ "step": "action", "tool_name": "execute_command", "tool_input": "pwd", "tool_call_id": "34567890" }}
{{ "step": "observe", "content": "The current directory is /Users/aniket.mahangare/myProjects" }}
{{ "step": "plan", "content": "From the available tools, I will select the execute_command tool to create a python file." }}
{{ "step": "action", "tool_name": "execute_command", "tool_input": "touch /Users/aniket.mahangare/myProjects/add_numbers.py", "tool_call_id": "4567890" }}
{{ "step": "observe", "content": "The python file add_numbers.py has been created in the current directory." }}
{{ "step": "action", "tool_name": "execute_command", "tool_input": "echo 'print(1+1)' > /Users/aniket.mahangare/myProjects/add_numbers.py", "tool_call_id": "567890" }}
{{ "step": "observe", "content": "The code to add two numbers has been written in the python file add_numbers.py" }}
{{ "step": "output", "content": "The python file add_numbers.py has been created in the current directory. You can now add two numbers." }}
"""


# extract the field from the message content
def extract_field(content: str, field: str) -> str:
    content_dict = json.loads(content)
    return content_dict.get(field, "")


while True:
    query = input("> ")
    messages = [
        {"role": "user", "content": query},
        {"role": "system", "content": system_prompt},
    ]

    step = "start"

    for event in compiled_graph.stream({"messages": messages, "llm_output": None}):
        if "chatbot" in event and "llm_output" in event["chatbot"]:
            llm_output = event["chatbot"]["llm_output"]
            if llm_output:
                try:
                    if llm_output.step == "output":
                        print(f"🤖: {llm_output.content}")
                    elif llm_output.step in ["action", "observe"]:
                        continue
                    else:
                        print(f"🌀 {llm_output.content}")
                except Exception as e:
                    print("error", e)
                    step = "invalid step"
