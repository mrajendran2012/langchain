# Create a REST API for the chatbot application
from fastapi import FastAPI, HTTPException
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langchain_core.messages import trim_messages
from typing import Sequence
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
import asyncio  # Add this import here to ensure asyncio is defined
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch
from typing import Dict, Any
from models.state import State
from models.chat import ChatModel
from models.call import CallModel
from models.state import State
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
# Environment variables for API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
langsmit_tracing = os.getenv("LANGSMITH_TRACING", "true")
# Initialize FastAPI app
app = FastAPI()
# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity; adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Initialize the chat model
llm = ChatModel(
    model_name="gpt-4o-mini",
    model_provider="openai"
)
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
# Initialize the state
state: State = {
    "messages": [HumanMessage(content="Hello, how can I help you?")],
    "language": "English"
}
# Create a CallModel instance with the initial state
_call_model = CallModel(llm, state=state)
# Create a StateGraph for the workflow
workflow = StateGraph(state_schema=State)
workflow.add_edge(START, "model")
def model_node(state: State) -> State:
    """Node function to call the model with the current state."""
    # Recreate CallModel with the incoming state
    dynamic_model = CallModel(llm, state=state)
    return dynamic_model.call()
workflow.add_node("model", model_node)

config = _call_model.config
language = _call_model.language


# Compile the workflow with a memory saver
memory = MemorySaver()
compiled_workflow = workflow.compile(checkpointer=memory)


@app.post("/chat")
# Add this route to swagger documentation
async def chat_endpoint(message: str, language: str = "English"):
    """Endpoint to handle chat messages."""
    # Update the state with the new message and language
    state["messages"].append(HumanMessage(content=message))
    state["language"] = language
    
    input_messages = state["messages"]

    ''' Review streaming functionality
    stream = compiled_workflow.astream(
            {"messages": input_messages, "language": language}, 
            config, 
            stream_mode="messages"
        )
    '''
    # Call the model with the updated state
    response_state = await compiled_workflow.invoke(state)

    # Extract the AI response from the state
    ai_message = response_state["messages"][-1]
    
    if isinstance(ai_message, AIMessage):
        return JSONResponse(content={"response": ai_message.content})
    else:
        raise HTTPException(status_code=500, detail="Invalid response from model")  
    

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/config")
async def get_config():
    """Endpoint to retrieve the configuration."""
    return {
        "configurable": {
            "thread_id": _call_model.config["configurable"]["thread_id"],
            "language": _call_model.language
        }
    }

@app.get("/visualize")
async def visualize_graph():
    """Endpoint to visualize the graph."""
    from IPython.display import Image, display
    try:
        graph_image = compiled_workflow.get_graph().draw_mermaid_png()
        return JSONResponse(content={"graph": graph_image})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/messages")
async def get_messages():
    """Endpoint to retrieve the current messages."""
    return {"messages": [msg.content for msg in state["messages"]]}

@app.post("/messages")
async def add_message(message: str, sender: str = "human"):
    """Endpoint to add a new message."""
    if sender.lower() == "human":
        state["messages"].append(HumanMessage(content=message))
    elif sender.lower() == "ai":
        state["messages"].append(AIMessage(content=message))
    else:
        raise HTTPException(status_code=400, detail="Invalid sender type")
    
    return {"status": "success", "message": message}

@app.get("/stream")
async def stream_messages():
    """Endpoint to stream messages from the chatbot."""
    async def message_stream():
        for msg in state["messages"]:
            if isinstance(msg, AIMessage):
                yield msg.content + "\n"
            await asyncio.sleep(0.5)
    return JSONResponse(content={"stream": message_stream()})

@app.get("/reset")
async def reset_chat():
    """Endpoint to reset the chat state."""
    state["messages"] = [HumanMessage(content="Hello, how can I help you?")]
    state["language"] = "English"
    return {"status": "success", "message": "Chat reset successfully"}

@app.get("/configurable")
async def get_configurable():
    """Endpoint to retrieve configurable parameters."""
    return {
        "thread_id": _call_model.config["configurable"]["thread_id"],
        "language": _call_model.language
    }

@app.post("/configurable")
async def update_configurable(thread_id: str = None, language: str = None):
    """Endpoint to update configurable parameters."""
    if thread_id:
        _call_model.config["configurable"]["thread_id"] = thread_id
    if language:
        _call_model.language = language
    
    return {
        "status": "success",
        "thread_id": _call_model.config["configurable"]["thread_id"],
        "language": _call_model.language
    }

# Add routes for the swagger 

@app.get("/swagger")
async def get_swagger():
    """Endpoint to retrieve Swagger documentation."""
    return JSONResponse(content={"swagger": "This is the Swagger documentation for the chatbot API."})
@app.get("/openapi.json")
async def get_openapi():
    """Endpoint to retrieve OpenAPI schema."""
    return JSONResponse(content=app.openapi())

# Run the FastAPI app using uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
# This code creates a FastAPI application that serves as a chatbot interface using LangGraph and LangChain.
# It includes endpoints for chatting, health checks, configuration retrieval, and message streaming.
# This code creates a FastAPI application that serves as a chatbot interface using LangGraph and LangChain.
# It includes endpoints for chatting, health checks, configuration retrieval, and message streaming.
