import ollama

client = ollama.Client(host='http://localhost:11434')

def pull_model(model_name: str):
    client.pull(model_name)

def run_model(model_name: str, prompt: str):
    response = client.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

pull_model("gemma3:1b")

response = run_model("gemma3:1b", "What is the capital of France?")
print(response)

