import json
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from models.schemas import OutputSchema
from graph.nodes import tools
from graph.factory import create_graph
from config.prompts import SYSTEM_PROMPT

# Load environment variables
load_dotenv()

# Initialize the LLM and bind tools
llm = init_chat_model(model_provider="openai", model="gpt-4.1")
# Bind tools first, then add structured output
llm_with_tools = llm.bind_tools(tools, tool_choice="auto").with_structured_output(
    OutputSchema
)


def main():
    """Main function to run the chatbot."""
    graph = create_graph(llm_with_tools)

    while True:
        query = input("> ")
        messages = [
            {"role": "user", "content": query},
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        for event in graph.stream({"messages": messages, "llm_output": None}):
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


if __name__ == "__main__":
    main()
