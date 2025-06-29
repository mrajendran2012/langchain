import os
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import HumanMessage

import os
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Define a function to initialize the chat model
def get_chat_model():
    """Initialize and return a chat model."""
    return init_chat_model(model="gpt-4o-mini", model_provider="openai")

# Define the function that calls the model
def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}

# Initialize the chat model
model = get_chat_model()

# Define a new graph
workflow = StateGraph(state_schema=MessagesState)

# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Define the configuration for the app. This can be used to pass parameters to the model or control its behavior.
config = {"configurable": {"thread_id": "abc123"}}

# Define the query
query = "Hi!"

while True:
    # Check if the user wants to exit
    if query.lower() in ["exit", "quit", "stop"]:
        print("Exiting the chat.")
        break

    # Invoke the app with the query
    query = input("You: ")
    if not query.strip():
        print("Please enter a valid query.")
        continue
    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()  # output contains all messages in state

