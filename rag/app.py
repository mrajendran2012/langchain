# Create a LLM agent application that can handle user queries and provide responses using a chat model and a search tool.
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langchain_core.messages import trim_messages
from typing import Sequence
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_core.prompts import ChatPromptTemplate
import asyncio  # Add this import here to ensure asyncio is defined
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch
from typing import Dict, Any
from langchain_core.messages import AIMessage
# Environment variables for API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
langsmit_tracing = os.getenv("LANGSMITH_TRACING", "true")

from models.state import State 
language_preference = "en"
from models.chat import ChatModel
from models.call import CallModel 
from models.state import State 

llm = ChatModel(
    model_name="gpt-4o-mini",
    model_provider="openai"
)

prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
            ),
        MessagesPlaceholder(variable_name="messages"),
        ]
    )

state: State = {
    "messages": [HumanMessage(content="Hello, how can I help you?")],
    "language": "English"
}
# ✅ Now pass the valid state to your model
_call_model = CallModel(llm, state=state)
workflow = StateGraph(state_schema=State)
workflow.add_edge(START, "model")

def model_node(state: State) -> State:
    # Recreate CallModel with the incoming state
    dynamic_model = CallModel(llm, state=state)
    return dynamic_model.call()

workflow.add_node("model", model_node)

config = _call_model.config
language = _call_model.language

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
            # If we're already in an event loop, create a task and wait for it
            task = loop.create_task(process_stream_messages(stream))
            loop.run_until_complete(task)
        except RuntimeError:
            # No running event loop, safe to use asyncio.run
            asyncio.run(process_stream_messages(stream))
            pass
    except KeyboardInterrupt:
        print("\nExiting the chat.")
        break
    except Exception as e:
        print(f"\nException occurred: {e}") 
        break