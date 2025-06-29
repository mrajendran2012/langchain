import os
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import HumanMessage

import os
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
langsmit_tracing = os.getenv("LANGSMITH_TRACING", "true")


from typing import Annotated

from typing_extensions import TypedDict
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.message import add_messages


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


# Define a function to initialize the chat model
def get_chat_model():
    """Initialize and return a chat model."""
    return init_chat_model(model="gpt-4o-mini", model_provider="openai")

# Initialize the chat model
llm = get_chat_model()

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)

# Add an entry point to the graph
graph_builder.add_edge(START, "chatbot")

# Compile the graph into an app
memory = MemorySaver()
app = graph_builder.compile(checkpointer=memory)

# Define the configuration for the app. This can be used to pass parameters to the model or control its behavior.
config = {"configurable": {
            "thread_id": "abc123"
            }
          }

# Visualize the graph
from IPython.display import Image, display

try:
    display(Image(app.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass


def stream_graph_updates(user_input: str):
    for event in app.stream({"messages": [{"role": "user", "content": user_input}]}, config, stream_mode="values"):
        for value in event.values():
            #print("Assistant:", value["messages"][-1].content)
            #print("Assistant:", value["messages"][-1].content, end="", flush=True)
            #print(value)
            print(value)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break