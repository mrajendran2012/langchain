import os
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import HumanMessage
from typing import Dict, Any

import os
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")


# Define a function to initialize the chat model
def get_chat_model():
    """Initialize and return a chat model."""
    return init_chat_model(model="gpt-4o-mini", model_provider="openai")

''' 
class CustomMessagesState(State):
    messages: list
    remaining_steps: int = StateField(default=10)
'''

def create_search_agent():
    """Create a search agent using TavilySearch."""
    memory = MemorySaver()
    model = get_chat_model()
    #workflow = StateGraph(state_schema=CustomMessagesState)
    #workflow.add_edge(START, "search")
    search = TavilySearch(max_results=5, model=model, memory=memory)
    #workflow.add_node("search", search)
    tools = [search]

    return create_react_agent(
        model=get_chat_model(),
        tools=tools,
        checkpointer=memory,
        #state_schema=CustomMessagesState
    )

agent = create_search_agent()

# Define the configuration for the app. This can be used to pass parameters to the model or control its behavior.
config = {"configurable": {"thread_id": "abc123"}}


input_message = {
    "role": "user",
    "content": "Hi, I'm Michael and I life in NY.",
}
for step in agent.stream(
    {"messages": [input_message]}, config, stream_mode="values"
):
    step["messages"][-1].pretty_print()

input_message = {
    "role": "user",
    "content": "What's the weather where I live?",
}

for step in agent.stream(
    {"messages": [input_message]}, config, stream_mode="values"
):
    step["messages"][-1].pretty_print()
