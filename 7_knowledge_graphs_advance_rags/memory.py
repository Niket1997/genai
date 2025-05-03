import os

from dotenv import load_dotenv
from mem0 import Memory
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"

config = {
    "version": "v1.1",
    "vector_store": {
        "provider": "qdrant",
        "config": {"host": QDRANT_HOST, "port": QDRANT_PORT},
    },
    "llm": {
        "provider": "openai",
        "config": {"api_key": OPENAI_API_KEY, "model": "gpt-4.1"},
    },
    "embedder": {
        "provider": "openai",
        "config": {"api_key": OPENAI_API_KEY, "model": "text-embedding-3-small"},
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": NEO4J_URL,
            "username": NEO4J_USERNAME,
            "password": NEO4J_PASSWORD,
        },
    },
}

memory = Memory.from_config(config)

openai_client = OpenAI(api_key=OPENAI_API_KEY)


def chat(message):
    related_memories = memory.search(query=message, user_id="userId123")
    related_memories_str = "\n".join(
        [memory["memory"] for memory in related_memories.get("results")]
    )

    system_prompt = f"""
    You are a memory-aware fact extraction agent, an advanced AI designed to 
    systematically analyze input content, extract structured knowledge, and 
    maintain an optimal memory store. Your primary function is information distillation, 
    and knowledge preservation with contextual awareness.
    
    Tone: Professional, analytical, precision-focused.
    Here are some memories that are related to the user's message:
    {related_memories_str}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )

    response_content = response.choices[0].message.content

    messages.append({"role": "assistant", "content": response_content})

    memory.add(messages=messages, user_id="userId123")

    return response_content


while True:
    message = input(">> ")
    response = chat(message)
    print(f"🤖: {response}")
