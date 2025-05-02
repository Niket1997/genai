import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# chain of thought prompting
system_prompt = """
You are an AI assistant who is expert in breaking down complex problems and then resolve the user's query.

For the given user input, analyse the input & break down the problem into smaller sub-problems.
At least think 5-6 steps on how to solve the problem before responding to the user.

Once you get the user input, you should follow the below steps:
1. analyse the user's input.
2. think multiple times on how to solve the problem.
3. come up with the output. 
4. validate the output with the user's input.
5. if the output is not as per the user's input, then think again and come up with the correct output.
6. if the output is as per the user's input, then respond to the user with the output.

Rules:
1. Follow the strict JSON output as per Output schema.
2. Always perform one step at a time & wait for next input. 
3. Carefully analyse the user query. 

Output Format:
{{ step:"string", content:"string"}}

Example:
Input: What is 2 + 2.
Output: {{ step:"analyse", content:"Alright! The user is interested in maths query and is asking a basic arithmetic question."}}
Output: {{ step:"think", content:"To perform addition, I must go from left to right & add all the operands."}}
Output: {{ step:"output", content:"The answer to 2 + 2 is 4."}}
Output: {{ step:"validate", content:"Seems like 4 is correct answer for 2 + 2."}}
Output: {{ step:"result", content:"2 + 2 = 4 and that is calculated by adding 2 and 2."}}
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    response_format={"type": "json_object"},
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What is 3 + 4 * 5?"},
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "step": "analyse",
                    "content": "The user is asking a mathematics question involving both addition and multiplication, which requires understanding the order of operations.",
                }
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "step": "think",
                    "content": "To solve the expression 3 + 4 * 5, I need to follow the order of operations (PEMDAS/BODMAS): first evaluating the multiplication before the addition.",
                }
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "step": "output",
                    "content": "First, I will calculate 4 * 5, which equals 20. Then, I will add 3 to that result.",
                }
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "step": "validate",
                    "content": "The result of 3 + 20, where 20 comes from the multiplication, needs to be calculated to confirm accuracy.",
                }
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "step": "result",
                    "content": "3 + 4 * 5 = 3 + 20 = 23, which follows the correct order of operations.",
                }
            ),
        },
    ],
)

print(response.choices[0].message.content)
