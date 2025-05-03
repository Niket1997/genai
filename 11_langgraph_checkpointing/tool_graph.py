import os
from typing import Annotated, List, Dict, Any, Optional
import json
import requests
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langsmith import traceable
from langgraph.graph.message import add_messages


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
@tool
@traceable
def get_weather(city: str) -> str:
    """Get the current weather information for a specified city.
    
    Args:
        city: The name of the city to get weather information for
    """
    print(f"🤖 Tool: Getting weather for {city}")
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)
    if response.status_code != 200:
        return "Sorry, I couldn't get the weather information for that city."
    return f"The weather in {city} is {response.text}."


@tool
@traceable
def execute_command(command: str) -> str:
    """Execute a system command and return its output.
    
    Args:
        command: The system command to execute
    """
    print(f"🤖 Tool: Executing command {command}")
    response = os.system(command)
    return f"The command {command} returned {response}."

tools = [get_weather, execute_command]


# initialize the llm & bind the tools
llm = init_chat_model(model_provider="openai", model="gpt-4.1")
llm.bind_tools(tools)


# create graph state
class State(TypedDict):
    messages: Annotated[List, add_messages]
    


## define the nodes in the graph
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

def determine_flow(state: Dict[str, Any]) -> str:
    """Determine the next node in the flow based on the current state.
    
    Args:
        state: The current state of the graph
        
    Returns:
        str: The name of the next node to execute
    """
    messages = state['messages']
    if not messages or len(messages) == 0 or messages[-1].content is None:
        return END

    content = messages[-1].content

    # Handle both string and dictionary content
    if isinstance(content, dict):
        step = content.get("step")
    else:
        # If content is a string, try to parse it as JSON
        try:
            step = extract_field(content, "step")
        except (json.JSONDecodeError, AttributeError):
            # If it's not valid JSON, treat it as a regular message
            return END

    # If we have an output step, end the flow
    if step == 'output':
        return END
    
    # If we have a plan continue with chatbot
    if step in ['plan']:
        return 'chatbot'
                
    # If we have an action step, go to tools
    if step == 'action':
        function = extract_field(content, "function")
        input = extract_field(content, "input")
        print(f"🧰 executing {function} tool with input {input}")
        return 'chatbot'
    
    if step == 'observe':
        output = extract_field(content, "output")
        print(f"🌀 observing output - {output}")
        return 'chatbot'
                
            
    return END

tool_node = ToolNode(tools)

## define the graph
graph = StateGraph(State)

graph.add_node("chatbot", chatbot)
graph.add_node("tools", tool_node)

# define the edges
graph.add_edge(START, "chatbot")
graph.add_conditional_edges("chatbot", determine_flow, {
    "tools": "tools",
    "chatbot": "chatbot",
    END: END
})
graph.add_edge("tools", "chatbot")

compiled_graph = graph.compile()


# system prompt
system_prompt = f"""
You are a helpful AI assistant who is specialized in helping users with their questions.
Given a user's query, you analuse the query carefully and generate a step by step plan.
You will be given a list of available tools and you will select the most relevant tool to use whenever needed.
Once the tool is selected, you will perform the action & generate output from the tool.
You will also observe the output from the tool and generate a result. The result should quirky and funny.



Rules:
- Follow the output format strictly.
- Always perform one step at a time & wait for next input.
- Carefully analyse the user query. 

Example:
Input: What is the weather in Tokyo?
Output: {{ "step": "plan", "content": "The user is asking about the weather in Tokyo." }}
Output: {{ "step": "plan", "content": "From the available tools, I will select the get_weather tool to get the weather information of Tokyo." }}
Output: {{ "step": "action", "function": "get_weather", "input": "Tokyo" }}
Output: {{ "step": "observe", "output": "The weather in Tokyo is sunny with a temperature of 12 degrees Celsius." }}
Output: {{ "step": "output", "content": "The weather in Tokyo is sunny with a temperature of 12 degrees Celsius. You can go out and enjoy the weather." }}
"""

# extract the field from the message content
def extract_field(content: str, field: str) -> str:
    content_dict = json.loads(content)
    return content_dict.get(field, "")

while True:
    query = input("> ")
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

    step = "start"

    for event in compiled_graph.stream({"messages": messages}):
        if 'chatbot' in event and 'messages' in event['chatbot']:
            messages = event['chatbot']['messages']
            if messages and messages[-1].content:
                content = messages[-1].content
                step = extract_field(content, "step")
                content_text = extract_field(content, "content")
                if step == "output":
                    print(f"🤖: {content_text}")
                elif step == "action" or step == "observe":
                    continue
                else:
                    print(f"🌀 {content_text}")

