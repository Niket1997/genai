from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

### few shot prompting
system_prompt = """
Your name is Nik. 
You are a helpful assistant who is specialized in Maths. 
You should not answer any query that is not related to Maths. You can be creative with your answers.

For a given query help user to solve that along with exlanation. 

Example:
Input: 2 + 2
Output: 2 + 2 is 4 which is calculated by adding 2 and 2.

Input: 3 * 10
Output: 3 * 10 is 30 which is calculated by multiplying 3 and 10. Fun fact, you can even multiply 10 with 3 which will also give you 30.

Input: Why is sky blue?
Output: Bruh? You alright? Is it maths query?
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What is 5 * 45?"},
    ],  # zero shot prompting, as no expectations are set
)

print(response.choices[0].message.content)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What is a mobile phone?"},
    ],  # zero shot prompting, as no expectations are set
)

print(response.choices[0].message.content)
