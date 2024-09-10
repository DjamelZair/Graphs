"""
Conversions between adjacency list and adjacency matrix.

@author: R.J. Both <r.j.both@uva.nl> and Vlad Niculae <v.niculae@uva.nl>
Djamel Zair 11015934

"""

import numpy as np
from typing import List


# The time complexity for the lists_to_matrix function is
# O(n) because you can see it has to (in the worst case)
# traverse al the nodes one time and traverse all the edges per
# node.


class AdjListsGraph:
    """Graph implementation as adjacency lists"""

    _adj: List[List[int]] = []

    def __init__(self, n_nodes: int = 1):
        """Create a graph with n_nodes nodes and no edges."""
        self.n_nodes = n_nodes
        self._adj = [[] for _ in range(n_nodes)]

    def has_edge(self, u: int, v: int):
        """Returns True if an edge u->v exists, else False."""
        return v in self._adj[u]

    def add_edge(self, u: int, v: int, check_if_exists=True):
        """Add an edge u->v unless one exists."""
        if check_if_exists and v in self._adj[u]:
            raise ValueError("Edge already exists.")
        self._adj[u].append(v)

    def del_edge(self, u: int, v: int):
        """Delete the edge u->v if it exists."""
        if v in self._adj[u]:
            self._adj[u].remove(v)

    def add_node(self):
        """Add a new node with no incident edges.

        The node shall have the next available index."""
        self.n_nodes += 1
        self._adj.append([])

    def del_node(self, u: int):
        """Remove node u and all edges incident to it.

        The indices of other nodes might change.
        """
        # first, delete u's list
        del self._adj[u]
        self.n_nodes -= 1

        # next, delete all edges pointing into u
        for v in range(self.n_nodes):
            self.del_edge(v, u)

        # now, go through all the nodes and update the indices.
        for v in range(self.n_nodes):
            for k in range(len(self._adj[v])):
                if self._adj[v][k] > u:
                    self._adj[v][k] -= 1

    def __repr__(self):
        """Create a readable string representation of the graph."""
        edges = [(u, v) for u in range(self.n_nodes) for v in self._adj[u]]
        edges = sorted(edges)
        return "Graph with edges: " + ", ".join(
            "({}, {})".format(u, v) for u, v in edges
        )


class AdjMatrixGraph:
    """Graph implementation as an adjacency matrix"""

    def __init__(self, n_nodes: int = 1):
        """Create a graph with n_nodes nodes and no edges."""
        self.n_nodes = n_nodes
        self._M = np.zeros((n_nodes, n_nodes), dtype=np.bool_)

    def add_edge(self, u: int, v: int):
        """Add an edge u->v unless one exists."""
        self._M[u, v] = True

    def has_edge(self, u: int, v: int) -> bool:
        """Returns True if an edge u->v exists, else False."""
        return self._M[u, v]

    def del_edge(self, u: int, v: int):
        """Delete the edge u->v if it exists."""
        self._M[u, v] = False

    def add_node(self):
        """Add a new node with no incident edges."""
        new_M = np.zeros((self.n_nodes + 1, self.n_nodes + 1), dtype=np.bool_)
        new_M[: self.n_nodes, : self.n_nodes] = self._M
        self._M = new_M
        self.n_nodes += 1

    def del_node(self, u: int):
        """Remove node u and all edges incident to it.

        The indices of other nodes might change.
        """
        mask = np.ones(self.n_nodes, dtype=np.bool_)
        mask[u] = 0
        self._M = self._M[mask][:, mask]
        self.n_nodes -= 1

    def __repr__(self):
        """Create a readable string representation of the graph."""
        edges = [
            (u, v)
            for u in range(self.n_nodes)
            for v in range(self.n_nodes)
            if self.has_edge(u, v)
        ]

        return "Graph with edges: " + ", ".join(
            "({}, {})".format(u, v) for u, v in edges
        )


def lists_to_matrix(graph: AdjListsGraph) -> AdjMatrixGraph:
    """Given a adjacency list structure, create the corresponding graph."""
    # Initialize a matrix
    result = AdjMatrixGraph(graph.n_nodes)
    print(result)
    for i in range(graph.n_nodes):
        for j in graph._adj[i]:
            result.add_edge(i, j)
    print(f"This the result {result}")
    return result


def matrix_to_lists(graph: AdjMatrixGraph) -> AdjListsGraph:
    """Given a matrix structure, create the corresponding adjacency graph."""
    result = AdjListsGraph(graph.n_nodes)
    for i in range(graph.n_nodes):
        for j in range(graph.n_nodes):
            if graph.has_edge(i, j):
                result.add_edge(i, j)
    return result


def length_two_paths(graph: AdjListsGraph) -> AdjListsGraph:
    """Compute the length-two paths graph of the given graph.

    Return the graph where u->v if there is a length-two path between u and
    v in the original graph."""
    matrix_graph = lists_to_matrix(graph)

    # Compute the matrix multiplication of the adjacency matrix with itself
    matrix_square = np.dot(matrix_graph._M, matrix_graph._M)
    np.fill_diagonal(matrix_square, 0)
    print(matrix_square)

    # Convert the resulting numpy ndarray back to an adjacency list graph
    result = AdjListsGraph(graph.n_nodes)
    for i in range(graph.n_nodes):
        for j in range(graph.n_nodes):
            if matrix_square[i, j]:
                result.add_edge(i, j)

    return result


def main():
    # Test graph
    gr = AdjMatrixGraph(5)
    gr.add_edge(0, 1)
    gr.add_edge(1, 2)
    gr.add_edge(2, 3)
    gr.add_edge(3, 4)
    gr.add_edge(4, 0)
    print(gr)

    gr_list = matrix_to_lists(gr)
    print(gr_list)

    # Adjusting test graph. Note: The indices will be renumbered.
    gr.del_node(1)
    print(gr)

    gr_list = matrix_to_lists(gr)
    print(gr_list)

    print("Length-two paths: ", length_two_paths(gr_list))


if __name__ == "__main__":
    main()
