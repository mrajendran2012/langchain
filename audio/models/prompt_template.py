from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class PromptTemplate:
    """A class to represent a prompt template for RAG (Retrieval-Augmented Generation) models."""

    def __init__(self, language: str = "English"):
        """
        Initializes the PromptTemplate with a given template string.

        Args:
            template (str): The prompt template string.
        """
        # If customization is needed, it can be passed as a parameter
        #self.dialect = dialect
        #self.top_k = top_k
        #self.table_info = table_info

        language = "English"  # Default language, can be overridden in the format method
        #self.input = input
    
        system_message = """
            You are an helpful agent designed to respond with most appropriate response.
            Given an input question, create a story from bible, look at the result and
            then return the answer. Unless the user specifies a specific number of examples 
            they wish to obtain, always limit your query to at most {top_k} results.

            You can order the results by a relevance to return the most interesting
            examples. 
        """.format(
                top_k=2,
        )

        user_prompt = "Question: {self.input}"

        prompt_template = ChatPromptTemplate(
            [("system", system_message), MessagesPlaceholder(variable_name="messages"),]
        )
       
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
