import json
import os
from openai import OpenAI
from dotenv import load_dotenv
import requests

load_dotenv()

client = OpenAI()


def get_weather(city: str) -> str:
    print(f"🤖 Tool: Getting weather for {city}")
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)
    if response.status_code != 200:
        return "Sorry, I couldn't get the weather information for that city."

    return f"The weather in {city} is {response.text}."


def execute_command(command: str) -> str:
    print(f"🤖 Tool: Executing command {command}")
    return os.system(command)


tools = {
    "get_weather": {
        "fn": get_weather,
        "description": "Get the weather information of a city",
        "parameters": {
            "type": "str",
            "properties": {
                "city": {"type": "string", "description": "The name of the city"}
            },
            "required": ["city"],
        },
    },
    "execute_command": {
        "fn": execute_command,
        "description": "Execute a command",
        "parameters": {
            "type": "str",
            "properties": {
                "command": {"type": "string", "description": "The command to execute"}
            },
            "required": ["command"],
        },
    },
}

# system prompt
system_prompt = f"""
You are a helpful AI assistant who is specialized in helping users with their questions.
You work on a start, plan, action & observe mode.
Given a user query & available tools, you will first analyze the user's question and generate a step by step plan.
Basis the plan, you will select the most relevant tool to use and perform the action.
While performing the action, you will wait for the action to complete and not down the output of the action.
Once the action is complete, you will observe the output and generate a result.

Rules:
- Follow the output JSON format strictly.
- Always perform one step at a time & wait for next input.
- Carefully analyse the user query. 

Output JSON Format:
{{ 
    "step": "string", 
    "content": "string",
    "function": "The name of the function if the step is action",
    "input": "The input of the function if the step is action",
}}

Available Tools:
{', '.join(f"{tool}: {tools[tool]['description']}" for tool in tools)}

Example:
Input: What is the weather in Tokyo?
Output: {{ "step": "plan", "content": "The user is asking about the weather in Tokyo." }}
Output: {{ "step": "plan", "content": "From the available tools, I will select the get_weather tool to get the weather information of Tokyo." }}
Output: {{ "step": "action", "function": "get_weather", "input": "Tokyo" }}
Output: {{ "step": "observe", "output": "12 degrees Celsius" }}
Output: {{ "step": "output", "content": "The weather in Tokyo is sunny with a temperature of 12 degrees Celsius." }}
"""

messages = [
    {"role": "system", "content": system_prompt},
]

while True:
    query = input("> ")
    messages.append({"role": "user", "content": query})

    while True:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=messages,
        )

        response_json = json.loads(response.choices[0].message.content)
        messages.append(
            {"role": "assistant", "content": response.choices[0].message.content}
        )

        if response_json.get("step") == "plan":
            print(f"🤖: {response_json.get('content')}")
            continue

        if response_json.get("step") == "output":
            print(f"👤: {response_json.get('content')}")
            break

        if response_json.get("step") == "action":
            function_name = response_json.get("function")
            if function_name not in tools:
                messages.append(
                    {
                        "role": "system",
                        "content": f"👤: {function_name} is not a valid function",
                    }
                )
                continue

            tool_fn = tools[function_name]["fn"]
            tool_input = response_json.get("input")
            if tool_input is None:
                messages.append(
                    {
                        "role": "system",
                        "content": f"👤: {function_name} is missing input",
                    }
                )
                continue

            tool_response = tool_fn(tool_input)
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps({"step": "observe", "output": tool_response}),
                }
            )
