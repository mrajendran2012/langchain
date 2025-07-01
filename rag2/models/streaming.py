from langchain_core.messages import AIMessage
class StreamingModel:
    """A base class for streaming models."""

    def __init__(self, stream):
        pass

    async def process_stream_messages(stream):
        """Process streamed messages from the chatbot."""

        async for chunk, metadata in stream:
            if isinstance(chunk, AIMessage):  # Filter to just model responses
                #print(chunk.content, end="|")  # To print each token separately
                print(chunk.content, end="")
            """Stream the response for a given prompt."""
        
    def close(self):
        """Close any resources used by the model."""
        pass