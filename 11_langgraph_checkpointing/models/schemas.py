from typing import Annotated, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import InjectedToolCallId


class OutputSchema(BaseModel):
    """Schema for LLM output."""

    step: str = Field(
        description="The current step in the process (plan, action, observe, output)"
    )
    role: str = Field(description="The role of the message (user, assistant, tool)")
    content: str = Field(description="The content of the message")
    tool_name: Optional[str] = Field(
        default=None, description="The name of the function if the step is action"
    )
    tool_input: Optional[str] = Field(
        default=None, description="The input of the function if the step is action"
    )
    tool_call_id: Optional[Annotated[str, InjectedToolCallId]] = Field(
        default=None, description="The id of the tool call"
    )
