from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langsmith.wrappers import wrap_openai
from openai import OpenAI
from pydantic import BaseModel
from typing_extensions import TypedDict

load_dotenv()


# State: defines the state in the graph
class State(TypedDict):
    user_message: str
    ai_message: str
    is_coding_question: bool


# Response: defines the response from the graph
class QueryTypeDetectionResponse(BaseModel):
    is_coding_question: bool


# Response: defines the response from the graph
class CodingQuestionSolverResponse(BaseModel):
    solution: str


# Response: defines the response from the graph
class GenericQuestionSolverResponse(BaseModel):
    solution: str


# create openai client
client = wrap_openai(OpenAI())


# identify the type of the user message
def query_type_detector(state: State):
    # call openai api to identify the type of the user message
    system_prompt = """
        You are a helpful assistant that can identify the type of the user message.
        If the type is coding question, return true.
        If the type is not coding question, return false.
        Return the response in the specified JSON format.
    """

    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state["user_message"]},
        ],
        response_format=QueryTypeDetectionResponse,
    )

    state["is_coding_question"] = response.choices[0].message.parsed.is_coding_question

    return state


# router: defines the router in the graph
def router(state: State):
    if state["is_coding_question"]:
        return "coding_question_solver"
    else:
        return "generic_question_solver"


# coding_question_solver: defines the coding question solver in the graph
def coding_question_solver(state: State):
    # call openai api to solve the coding question
    system_prompt = """
        You are a helpful assistant that can solve the coding question.
        Return the response in the specified JSON format.
    """

    response = client.beta.chat.completions.parse(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state["user_message"]},
        ],
        response_format=CodingQuestionSolverResponse,
    )

    state["ai_message"] = response.choices[0].message.parsed.solution

    return state


# generic_question_solver: defines the generic question solver in the graph
def generic_question_solver(state: State):
    # call openai api to solve the generic question
    system_prompt = """
        You are a helpful assistant that can solve the generic question.
        Return the response in the specified JSON format.
    """

    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state["user_message"]},
        ],
        response_format=GenericQuestionSolverResponse,
    )

    state["ai_message"] = response.choices[0].message.parsed.solution

    return state


# create the graph
graph = StateGraph(State)

# add the nodes to the graph
graph.add_node("query_type_detector", query_type_detector)
graph.add_node("router", router)
graph.add_node("coding_question_solver", coding_question_solver)
graph.add_node("generic_question_solver", generic_question_solver)

# add the edges to the graph
graph.add_edge(START, "query_type_detector")
graph.add_conditional_edges("query_type_detector", router)
graph.add_edge("coding_question_solver", END)
graph.add_edge("generic_question_solver", END)

# compile the graph
compiled_graph = graph.compile()

while True:
    user_message = input("> ")
    result = compiled_graph.invoke({"user_message": user_message})
    print(f"🤖: {result['ai_message']}")
