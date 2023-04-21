import numpy as np

def generate_adjacency_matrix(n: int, graph: str = 'complete', self_connections: bool = False, **kwargs) -> np.ndarray:
    """Generate an adjacency matrix for a graph (network) defining the environment of agents, with vertices representing agents, and edges representing communication.
    
    Args:
        n: the number of vertices (agents) in the graph

        graph: the kind of graph to generate {'complete', 'random'}

        self_connections: whether to allow an agent to communicate with itself (weird)
    """
    graph = graph_map[graph](n, **kwargs)
    if not self_connections:
        graph -= np.eye(n)
    return graph


def complete_graph(n: int, **kwargs) -> np.ndarray:
    return np.ones((n, n))

def random_graph(n: int, **kwargs) -> np.ndarray:
    raise NotImplementedError

graph_map = {
    "complete": complete_graph,
    "random": random_graph,
}