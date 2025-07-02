# Write python code to transcribe audio files using a language model.
from langchain.audio import AudioTranscriber
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
import os
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
langsmit_tracing = os.getenv("LANGSMITH_TRACING", "true")
def get_chat_model():
    """Initialize and return a chat model."""
    return init_chat_model(model="gpt-4o-mini", model_provider="openai")
# Initialize the chat model
llm = get_chat_model()  
# Create an audio transcriber using the chat model
transcriber = AudioTranscriber.from_chat_model(llm) 
# Define a function to transcribe audio files
def transcribe_audio(file_path: str) -> str:
    """Transcribe audio from a file."""
    # Use   the transcriber to transcribe the audio file
    transcription = transcriber.transcribe(file_path)
    return transcription.text   

# Example usage
if __name__ == "__main__":
    audio_file = "path/to/your/audio/file.wav"  # Replace with your audio file path
    transcription = transcribe_audio(audio_file)
    print(f"Transcription: {transcription}")    
# Note: Ensure that the audio file exists at the specified path before running the script.
# This script initializes a chat model, creates an audio transcriber, and defines a function to transcribe audio files using the transcriber. The transcription is printed to the console.
# Ensure the response is wrapped in a list as expected by the state schema
    return {"messages": trimmed_messages + [response], "language": state["language"]}