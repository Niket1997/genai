import ollama

client = ollama.Client(host='http://localhost:11434')

def pull_model(model_name: str):
    client.pull(model_name)

def chat(model_name: str, messages: list[dict]):
    response = client.chat(model=model_name, messages=messages)
    return response["message"]["content"]

pull_model("gemma3:1b")

system_prompt = """
You are a helpful assistant that can answer questions and help with tasks.
"""

messages = [
    {"role": "system", "content": system_prompt},
]

while True:
    prompt = input(">>> ")
    messages.append({"role": "user", "content": prompt})
    response = chat("gemma3:1b", messages)
    print(f"🤖: {response}") 
    messages.append({"role": "assistant", "content": response})
    print("\n")

