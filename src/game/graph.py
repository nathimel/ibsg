import numpy as np


def generate_adjacency_matrix(
    n: int, graph: str = "complete", self_connections: bool = False, **kwargs
) -> np.ndarray:
    """Generate an adjacency matrix for a graph (network) defining the environment of agents, with vertices representing agents, and edges representing communication.

    Args:
        n: the number of vertices (agents) in the graph

        graph: the kind of graph to generate {'complete', 'random'}

        self_connections: whether to allow an agent to communicate with itself (weird)
    """
    if graph == "complete":
        graph = np.ones((n, n))

    elif graph == "random":
        # generate a random UNWEIGHTED graph
        graph = np.zeros(n, n)
        while not (graph - np.eye(n, n)).any():  # at least one irreflexive connection
            graph = np.array(np.random.randn(n, n) > 0, dtype=int)

    else:
        raise ValueError("Argument `graph` must be str 'complete' or 'random'.")

    if not self_connections:
        graph -= np.eye(n)
    return graph
