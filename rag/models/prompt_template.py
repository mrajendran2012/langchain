from models.prompt_template import  ChatPromptTemplate, MessagesPlaceholder
class PromptTemplate:
    """A class to represent a prompt template for RAG (Retrieval-Augmented Generation) models."""

    def __init__(self, language: str = "English"):
        """
        Initializes the PromptTemplate with a given template string.

        Args:
            template (str): The prompt template string.
        """
        """  
        if template is None or not isinstance(template, str):
            raise ValueError("Template must be a non-empty string.")
        if not template.strip():
            raise ValueError("Template cannot be an empty string or whitespace.")
        if not isinstance(template, str):
            raise TypeError("Template must be a string.")
        """
        language = "English"  # Default language, can be overridden in the format method
        # Create a chat prompt template with a system message and a placeholder for messages
        prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
            ),
        MessagesPlaceholder(variable_name="messages"),
        ]
        )
        # Will assign the template to the instance variable. For now its hardcoded
        self.template = prompt_template
        self.language = language

    def format(self, **kwargs) -> str:
        """
        Formats the prompt template with the provided keyword arguments.

        Args:
            **kwargs: Keyword arguments to format the template.

        Returns:
            str: The formatted prompt.
        """
        return self.template.format(**kwargs)   
    
    def __call__(self, **kwargs) -> str:
        """
        Calls the format method to get the formatted prompt.

        Args:
            **kwargs: Keyword arguments to format the template.

        Returns:
            str: The formatted prompt.
        """
        return self.format(**kwargs)
    
# Example usage:
# prompt_template = PromptTemplate("Translate the following from English into {language}: {text}")
# response = prompt_template(language="Hindi", text="Hello, how are you?")
# print(response)  # Output: Translate the following from English into Hindi: Hello, how are you?   