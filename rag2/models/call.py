from langchain_core.messages import HumanMessage
from langchain_core.messages import trim_messages
from langchain_core.messages import BaseMessage
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from models.state import State
#from models.prompt_template import query_prompt_template
from langchain_core.messages import AIMessage
from langchain_community.agent_toolkits import SQLDatabaseToolkit


class CallModel:
    """Base class for call models."""
    def __init__(self, llm, db, state: State):
        self.config = {"configurable": {
            "thread_id": "abc1234"
            }
          }
        self.language = self.config["configurable"]["language"] = "English"
        self.state = state 
        self.llm = llm
        self.db = db
        self.trimmer = trim_messages(
        max_tokens=256,
        strategy="last",
        token_counter=self.token_counter,
        include_system=True,
        allow_partial=False,
        start_on="human",
    )

    # Use a token counting function or a model that implements get_num_tokens_from_messages()
    ''' 
    def token_counter(self):
        print("[DEBUG] Raw messages in state:", self.state["messages"])
        for i, msg in enumerate(self.state["messages"]):
            print(f"[DEBUG] Message {i}: type={type(msg)}, value={msg}")

        return sum(len(msg.content.split()) for msg in self.state["messages"])
    '''
     
    def token_counter(self, messages: list[BaseMessage]) -> int:
        return sum(len(msg.content.split()) for msg in messages)
    '''
    def write_query(self):
        """Write a query based on the database and state."""
        """Generate SQL query to fetch information."""
        prompt = query_prompt_template.invoke(
            {
            "dialect": self.calldb.dialect,
            "top_k": 10,
            "table_info": self.db.get_table_info(),
            "input": self.state["question"],
            }
        )
        return prompt

    def execute_query(self):
        """Execute SQL query."""
        execute_query_tool = QuerySQLDatabaseTool(db=self.db)
        return {"result": execute_query_tool.invoke(self.state["query"])}

    def generate_answer(self):
        """Answer question using retrieved information as context."""
        prompt = (
            "Given the following user question, corresponding SQL query, "
            "and SQL result, answer the user question.\n\n"
            f'Question: {self.state["question"]}\n'
            f'SQL Query: {self.state["query"]}\n'
            f'SQL Result: {self.state["result"]}'
        )
        response = self.llm.invoke(prompt)
        return {"answer": response.content}
    '''

    def create_agent(self):
        """Create a SQL database agent."""
        toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm.model)  # Access underlying model

        #toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        tools = toolkit.get_tools()
        model_with_tools = self.llm.bind_tools(tools)

        return create_react_agent(
            model=model_with_tools,
            tools=tools
        )

    def call(self) -> dict:
        """Call the model with the given arguments."""
        # Filter out any objects that are not instances of BaseMessage
        valid_messages = [msg for msg in self.state["messages"] if isinstance(msg, BaseMessage)]

        trimmed_messages = self.trimmer.invoke(valid_messages)
        if not trimmed_messages:
            print("[WARN] Trimmed messages were empty. Using original messages.")
            trimmed_messages = valid_messages

        # Debugging output to see the messages being sent to the model
        ''' 
        print("\n[DEBUG] Messages sent to model:")
        for msg in trimmed_messages:
            sender = "User" if isinstance(msg, HumanMessage) else "Assistant"
            print(f"{sender}: {msg.content}")
        '''
        agent = self.create_agent()
        try:
            response = agent.invoke({"messages": trimmed_messages})
        except Exception as e:
            print(f"[ERROR] Agent invoke failed: {e}")
            raise

        # Message for debugging
        # This will print the response from the model
        #print("\n[DEBUG] Model response:")
        #print(response)

        if isinstance(response, BaseMessage):
            final_messages = trimmed_messages + [response]
        else:
            final_messages = trimmed_messages + [AIMessage(content=str(response))]

        return {"messages": final_messages, "language": self.state["language"]}