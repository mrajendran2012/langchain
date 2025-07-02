
from typing import Sequence
from langchain_core.messages import BaseMessage
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages

# Define the state schema for the chatbot
class State(TypedDict):
    """State schema for the chatbot."""
    # Messages have the type "list". The `add_messages` function
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str = "English"
