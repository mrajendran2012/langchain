""" Chatbot using LangGraph and LangChain with OpenAI's GPT-4o-mini model.
This script sets up a simple chatbot that can respond to user queries in a specified language.
Ashynchronously processes streamed messages from the chatbot.
"""
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langchain_core.messages import HumanMessage
from langchain_core.messages import trim_messages
from typing import Sequence
from langchain_core.messages import BaseMessage
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
import asyncio  # Add this import here to ensure asyncio is defined
import os
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
langsmit_tracing = os.getenv("LANGSMITH_TRACING", "true")

# Define the state schema for the chatbot
class State(TypedDict):
    """State schema for the chatbot."""
    # Messages have the type "list". The `add_messages` function
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str = "English"

# Define a function to initialize the chat model
def get_chat_model():
    """Initialize and return a chat model."""
    return init_chat_model(model="gpt-4o-mini", model_provider="openai")

# Create a prompt template for the chatbot
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Initialize the chat model
llm = get_chat_model()
   
# Define the function that calls the model
def call_model(state: State):
    trimmed_messages = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke(
        {"messages": trimmed_messages, "language": state["language"]}
    )
    response = llm.invoke(prompt)
    # Ensure the response is wrapped in a list as expected by the state schema
    return {"messages": trimmed_messages + [response], "language": state["language"]}

# Define a new graph
def visualize_graph(app):
    """Visualize the graph."""
    from IPython.display import Image, display
    try:
        display(Image(app.get_graph().draw_mermaid_png()))
    except Exception:
        # This requires some extra dependencies and is optional
        pass

# Define the configuration for the app. This can be used to pass parameters to the model or control its behavior.
config = {"configurable": {
            "thread_id": "abc1234"
            }
          }
language = config["configurable"]["language"] = "English"

# Create a new state graph
workflow = StateGraph(state_schema=State)

# Add an entry point to the graph
workflow.add_edge(START, "model")

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=llm,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

workflow.add_node("model", call_model)

# Compile the graph into an app
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

#visualize_graph(app)   

async def process_stream_messages(stream):
    """Process streamed messages from the chatbot."""

    async for chunk, metadata in stream:
        if isinstance(chunk, AIMessage):  # Filter to just model responses
            #print(chunk.content, end="|")  # To print each token separately
            print(chunk.content, end="")

# Start the chatbot interaction
print("Welcome to the LangGraph Chatbot! Type 'quit' to exit.")
while True:
    try:
        query = input("\nUser: ")
        if query.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        input_messages =  [HumanMessage(query)]
        stream = app.astream(
            {"messages": input_messages, "language": language}, 
            config, 
            stream_mode="messages"
        )
        # Process the stream of messages asynchronously   
        try:
            loop = asyncio.get_running_loop()
            # If we're already in an event loop, create a task
            task = loop.create_task(process_stream_messages(stream))
            loop.run_until_complete(task)
        except RuntimeError:
            # No running event loop, safe to use asyncio.run
            asyncio.run(process_stream_messages(stream))
    except KeyboardInterrupt:
        print("\nExiting the chat.")
        break
    except:
        print("\nException occurred") 
        break