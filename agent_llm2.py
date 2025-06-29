import os
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch
from typing import Dict, Any

import os
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
langsmit_tracing = os.getenv("LANGSMITH_TRACING", "true")


# Define a function to initialize the chat model
def get_chat_model():
    """Initialize and return a chat model."""
    return init_chat_model(model="gpt-4o-mini", model_provider="openai")

def create_search_agent():
    """Create a search agent using TavilySearch."""
    memory = MemorySaver()
    model = get_chat_model()
    search = TavilySearch(max_results=5, model=model, memory=memory)
    tools = [search]
    model_with_tools = model.bind_tools(tools)

    return create_react_agent(
        model=model_with_tools,
        tools=tools,
        checkpointer=memory,
    )

agent = create_search_agent()

# Define the configuration for the app. This can be used to pass parameters to the model or control its behavior.
config = {"configurable": {
            "thread_id": "abc123"
            }
          }


# Define the query
query = "Hi!"

while True:
    # Check if the user wants to exit
    if query.lower() in ["exit", "quit", "stop"]:
        print("Exiting the chat.")
        break

    # Invoke the app with the query
    query = input("\nYou: ")
    if not query.strip():
        print("Please enter a valid query.")
        continue
    input_message = { "role": "user", "content": query }

    '''  
    for step, metadata in agent.stream(
        {"messages": [input_message]}, config, stream_mode="messages"
    ):
        if metadata["langgraph_node"] == "agent" and (text := step.text()):
            #print(text, end="|") # tokenized
            print(text, end="") 
    '''
    for step in agent.stream(
        {"messages": [input_message]}, config, stream_mode="values"
    ):
        step["messages"][-1].pretty_print()  # output contains all messages in state