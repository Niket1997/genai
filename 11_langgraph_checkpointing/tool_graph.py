import os
from typing import Annotated

import requests
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langsmith import traceable
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# load the environment variables
load_dotenv()


## define the input schema for the tools
# weather input
class WeatherInput(BaseModel):
    city: str = Field(description="The name of the city to get weather information for")


# command input
class CommandInput(BaseModel):
    command: str = Field(description="The system command to execute")


### create the tools
@traceable
@tool
def get_weather(city: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> dict:
    print(f"🤖 Tool: Getting weather for {city}")
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)
    if response.status_code != 200:
        return {
            "messages": [
                ToolMessage(
                    "Sorry, I couldn't get the weather information for that city.",
                    tool_call_id=tool_call_id,
                )
            ]
        }

    return {
        "messages": [
            ToolMessage(
                f"The weather in {city} is {response.text}.", tool_call_id=tool_call_id
            )
        ]
    }


@traceable
@tool
def execute_command(
    command: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> dict:
    print(f"🤖 Tool: Executing command {command}")
    response = os.system(command)
    return {
        "messages": [
            ToolMessage(
                f"The command {command} returned {response}.", tool_call_id=tool_call_id
            )
        ]
    }


tools = [get_weather, execute_command]


# initialize the llm & bind the tools
llm = init_chat_model(model_provider="openai", model="gpt-4.1")
llm.bind_tools(tools)


# create graph state
class State(TypedDict):
    messages: Annotated[list, add_messages]


## define the nodes in the graph
# define the chatbot node
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


tool_node = ToolNode(tools)

## define the graph
graph = StateGraph(State)

graph.add_node("chatbot", chatbot)
graph.add_node("tools", tool_node)

# define the edges
graph.add_edge(START, "chatbot")
graph.add_conditional_edges("chatbot", tools_condition, "tools")
graph.add_edge("tools", "chatbot")
graph.add_edge("chatbot", END)

compiled_graph = graph.compile()


# system prompt
system_prompt = f"""
You are a helpful AI assistant who is specialized in helping users with their questions.
You work on a start, plan, action & observe mode.
Given a user query & available tools, you will first analyze the user's question and generate a step by step plan.
Basis the plan, you will select the most relevant tool to use and perform the action.
While performing the action, you will wait for the action to complete and not down the output of the action.
Once the action is complete, you will observe the output and generate a result.

Rules:
- Follow the output format strictly.
- Always perform one step at a time & wait for next input.
- Carefully analyse the user query. 

Example:
Input: What is the weather in Tokyo?
Output: {{ "step": "plan", "content": "The user is asking about the weather in Tokyo." }}
Output: {{ "step": "plan", "content": "From the available tools, I will select the get_weather tool to get the weather information of Tokyo." }}
Output: {{ "step": "action", "function": "get_weather", "input": "Tokyo" }}
Output: {{ "step": "observe", "output": "12 degrees Celsius" }}
Output: {{ "step": "output", "content": "The weather in Tokyo is sunny with a temperature of 12 degrees Celsius." }}
"""

while True:
    query = input("Enter a query: ")
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": query,
        },
    ]
    result = compiled_graph.invoke({"messages": messages})
    print(result)
