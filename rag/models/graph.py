from IPython.display import Image, display
        
class GraphModel:
    """A base class for graph-based models in RAG (Retrieval-Augmented Generation)."""

    def __init__(self, graph):
        """
        Initialize the GraphModel with a graph.

        Args:
            graph: The graph to be used by the model.
        """
        self.graph = graph

    def query(self, query_text):
        """
        Query the graph with the given text.

        Args:
            query_text: The text to query the graph with.

        Returns:
            The results of the query.
        """
        raise NotImplementedError("Subclasses should implement this method.")    

    # Define a new graph
    def visualize_graph(app):
        """Visualize the graph."""
        try:
            display(Image(app.get_graph().draw_mermaid_png()))
        except Exception:
            # This requires some extra dependencies and is optional
            pass
   