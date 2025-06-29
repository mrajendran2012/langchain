import os
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

def prompt_template(language: str, text: str) -> ChatPromptTemplate:
    """Create a prompt template for translation.""" 
    system_template = "Translate the following from English into {language}"

    template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{text}")]
    )
    return template.format(language=language, text=text)


model = get_chat_model()
prompt = prompt_template("Hindi", "Hello, how are you?")   
response = model.invoke(prompt)
print(response.content)
print(response)