import os
from langchain.chat_models import init_chat_model

import os
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

def get_chat_model():
    """Initialize and return a chat model."""
    return init_chat_model(model="gpt-4o-mini", model_provider="openai")

model = get_chat_model()
response = model.invoke("Hello, how are you?")
print(response.content)
