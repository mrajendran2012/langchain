from langchain.chat_models import init_chat_model

class ChatModel:
    """
    A class representing a chat model for RAG (Retrieval-Augmented Generation).
    
    This class is designed to handle chat interactions, including initialization of the model,
    invoking the model with messages, and managing the state of the chat.
    """

    def __init__(self, model_name: str, model_provider: str):
        """
        Initialize the chat model with the specified model name and provider.
        
        Args:
            model_name (str): The name of the chat model to use.
            model_provider (str): The provider of the chat model (e.g., 'openai').
        """
        self.model = init_chat_model(model=model_name, model_provider=model_provider)


    def invoke(self, messages: list):
        """
        Invoke the chat model with a list of messages.
        
        Args:
            messages (list): A list of messages to send to the chat model.
        
        Returns:
            The response from the chat model.
        """
        return self.model.invoke(messages)
    
    def bind_tools(self, tools: list):
        """
        Bind tools to the chat model.
        
        Args:
            tools (list): A list of tools to bind to the chat model.
        
        Returns:
            The chat model with the bound tools.
        """
        return self.model.bind_tools(tools) 
# Example usage:
# chat_model = ChatModel(model_name="gpt-4o-mini", model_provider="openai")