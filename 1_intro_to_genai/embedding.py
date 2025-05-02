from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

client = OpenAI()

text = "The Eiffel Tower, Paris's iconic iron lattice masterpiece, stands as a timeless symbol of French elegance and architectural brilliance."

response = client.embeddings.create(model="text-embedding-3-small", input=text)

print("vector embedding: ", response.data[0].embedding)
