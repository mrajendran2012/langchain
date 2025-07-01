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
            You are an agent designed to interact with a SQL database.
            Given an input question, create a syntactically correct {dialect} query to run,
            then look at the results of the query and return the answer. Unless the user
            specifies a specific number of examples they wish to obtain, always limit your
            query to at most {top_k} results.

            You can order the results by a relevant column to return the most interesting
            examples in the database. Never query for all the columns from a specific table,
            only ask for the relevant columns given the question.

            You MUST double check your query before executing it. If you get an error while
            executing a query, rewrite the query and try again.

            DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
            database.

            To start you should ALWAYS look at the tables in the database to see what you
            can query. Do NOT skip this step.

            Then you should query the schema of the most relevant tables.
        """.format(
                dialect="SQLite",
                top_k=5,
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
