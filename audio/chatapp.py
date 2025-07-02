from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from typing import Sequence
import asyncio
import os
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from models.state import State
from models.chat import ChatModel
from models.call import CallModel
import base64
from models.prompt_template import PromptTemplate
from langchain_community.utilities import SQLDatabase
from models.sqlquery import QueryOutput


# Environment variables for API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
langsmit_tracing = os.getenv("LANGSMITH_TRACING", "true")

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the chat model
llm = ChatModel(
    model_name="gpt-4o-mini",
    model_provider="openai"
)

# Create a prompt template
prompt_template = PromptTemplate() 

# Initialize the state
state: State = {
    "messages": [HumanMessage(content="Hello, how can I help you?")],
    "language": "English"
}
#db = SQLDatabase.from_uri("sqlite:///Chinook.db")
agent = []
# Create a CallModel instance
_call_model = CallModel(llm, agent, state=state)

graph_builder = StateGraph(state_schema=State)
graph_builder.add_edge(START, "write_query")

def model_node(state: State) -> State:
    dynamic_model = CallModel(llm, agent, state=state)  
    return dynamic_model.call()

graph_builder.add_node("write_query", model_node)
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# Pydantic model for chat input
class ChatInput(BaseModel):
    message: str
    language: str = "English"


from models.speech import PygameAudio



@app.post("/speech")
async def speech_endpoint():
    """Endpoint to handle speech input and return AI response."""
    try:
        speech = PygameAudio()  # Assuming PygameAudio is defined elsewhere
        source_lang_code = "en-US"  # Input speech language (e.g., English)
        target_lang_code = "ta"     # Output language (e.g., Tamil)
        # Assign the language based on lang_code or default to English
        
        target_language = "Tamil"
        source_language = "English"

        original_text = speech.recognize_speech(lang_code=source_lang_code)

        #if original_text:
        #    translated_text = translate_text(original_text, dest_lang=target_lang_code)
        #if translated_text:
        #    speak_text(translated_text, lang=target_lang_code)
        # Update the state with the correct key and attribute
        state["messages"].append(HumanMessage(content=original_text))
        state["language"] = source_language
        # Ensure the state is updated with the translated text
        
        # Call the model
        response_state = await graph.ainvoke(state, config=_call_model.config)
        
        # Extract AI response
        ai_message = response_state["messages"][-1]
        
        from deep_translator import GoogleTranslator

        if isinstance(ai_message, AIMessage):
            # Speak the AI response. Not needed to translate as the LLM will return in the dest language
            #translated_text = speech_input.translate_text(ai_message.content, dest_lang=target_lang_code)
            # translated_text = GoogleTranslator(source='auto', target=speech.target_lang_code).translate(ai_message.content)
            # Speak the AI response in the target language and return audio bytes
            audio_bytes = speech.speak_text(text=ai_message.content, lang=target_lang_code)
            return StreamingResponse(audio_bytes, media_type="audio/wav")
            #if translated_text:
            #    return speech_input.speak_text(translated_text=ai_message.content, lang=target_lang_code)
        else:
            raise HTTPException(status_code=500, detail="Invalid response from model")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing speech: {str(e)}")

@app.post("/chat")
async def chat_endpoint(chat_input: ChatInput):
    """Endpoint to handle chat messages."""
    try:
        # Update the state with the correct key and attribute
        state["messages"].append(HumanMessage(content=chat_input.message))
        
        # Call the model
        response_state = await graph.ainvoke(state, config=_call_model.config)
        
        # Extract AI response
        ai_message = response_state["messages"][-1]
        
        if isinstance(ai_message, AIMessage):
            return JSONResponse(content={"response": ai_message.content})
        else:
            raise HTTPException(status_code=500, detail="Invalid response from model")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")


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
    try:
        graph_image = graph.get_graph().draw_mermaid_png()
        # Convert bytes to base64 for JSON response
        graph_b64 = base64.b64encode(graph_image).decode('utf-8')
        return JSONResponse(content={"graph": graph_b64})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating graph: {str(e)}")

@app.get("/messages")
async def get_messages():
    """Endpoint to retrieve the current messages."""
    return {"messages": [msg.content for msg in state["messages"]]}

@app.post("/messages")
async def add_message(message: str, sender: str = "human"):
    """Endpoint to add a new message."""
    try:
        if sender.lower() == "human":
            state["messages"].append(HumanMessage(content=message))
        elif sender.lower() == "ai":
            state["messages"].append(AIMessage(content=message))
        else:
            raise HTTPException(status_code=400, detail="Invalid sender type")
        
        return {"status": "success", "message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding message: {str(e)}")

@app.get("/stream")
async def stream_messages():
    """Endpoint to stream messages from the chatbot."""
    async def message_stream():
        for msg in state["messages"]:
            if isinstance(msg, AIMessage):
                yield f"data: {msg.content}\n\n"
            await asyncio.sleep(0.5)
    
    return StreamingResponse(message_stream(), media_type="text/event-stream")

@app.get("/reset")
async def reset_chat():
    """Endpoint to reset the chat state."""
    try:
        state["messages"] = [HumanMessage(content="Hello, how can I help you?")]
        state["language"] = "English"
        return {"status": "success", "message": "Chat reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting chat: {str(e)}")

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
    try:
        if thread_id:
            _call_model.config["configurable"]["thread_id"] = thread_id
        if language:
            _call_model.language = language
        
        return {
            "status": "success",
            "thread_id": _call_model.config["configurable"]["thread_id"],
            "language": _call_model.language
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating config: {str(e)}")

@app.get("/swagger")
async def get_swagger():
    """Endpoint to retrieve Swagger documentation."""
    return JSONResponse(content={"swagger": "This is the Swagger documentation for the chatbot API."})

@app.get("/openapi.json")
async def get_openapi():
    """Endpoint to retrieve OpenAPI schema."""
    return JSONResponse(content=app.openapi())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")