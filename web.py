"""
Nearly Complete Graphs

@author: Vlad Niculae <v.niculae@uva.nl>

# Djamel Zair 11015934

"""
import numpy as np
from typing import List
from conversions import AdjListsGraph

# Question 1
# Because we are talking about a nearly complete graph
# this means that almost every node has almost every other
# node as their "neighbour".
# This means that the list representation that normally
# has a complexity of O(n+n-k) will approach that of
# O(n^2) which is the same as the complexity of the adjecent matrix
# when talking about the space complexity.
# There are however matrix representation possible where not all the
# possible edges (so including also non-edges) are stored but because
# the graph is nearly complete these will also approach an O(n^2) complexity

# Question 2
# I am considering a non adj list data structure
# O(n+k)


class NearlyCompleteGraph:
    """Graph implementation for nearly complete graphs.

    Implement the methods provided.

    Upon initialization, all edges are considered present, but your
    object should not take up significant space.

    Example behavior:

    >>> g = NearlyCompleteGraph(10)
    >>> g.has_edge(2, 3)
    True
    >>> g.del_edge(2, 3)
    >>> g.has_edge(2, 3)
    False

    """

    graph: List[List[int]] = []

    def __init__(self, n_nodes: int = 1):
        """Initialize a complete graph,

        All edges are understood to be present.
        """
        # _adj: List[List[int]] = []

        self.n_nodes = n_nodes
        self.graph = [[] for _ in range(n_nodes)]

    def has_edge(self, u: int, v: int):
        """Returns True if an edge u->v exists, else False."""
        if u <= self.n_nodes and v <= self.n_nodes:
            return u != v and v not in self.graph[u]
        else:
            return False

    def add_edge(self, u: int, v: int):
        """Add an edge u->v unless one exists."""
        if u != v and v in self.graph[u]:
            self.graph[u].remove(v)

    def del_edge(self, u: int, v: int):
        """Delete the edge u->v if it exists."""
        if u != v and v not in self.graph[u]:
            self.graph[u].append(v)

    def add_node(self):
        """Add a new node with incident edges to all existing nodes.

        The node shall have the next available index."""
        self.n_nodes += 1
        self.graph.append([])

    def del_node(self, u: int):
        """Delete the node u and all incident edges, if exists."""
        # first, delete u's list
        del self.graph[u]
        self.n_nodes -= 1

        # next, delete all edges pointing into u
        for v in range(self.n_nodes):
            self.add_edge(v, u)

        # going through all the nodes and update the indices.
        for v in range(self.n_nodes):
            for k in range(len(self.graph[v])):
                if self.graph[v][k] > u:
                    self.graph[v][k] -= 1

    def to_adj_lists(self) -> AdjListsGraph:
        result = AdjListsGraph(self.n_nodes)
        for u in range(self.n_nodes):
            for v in range(self.n_nodes):
                if self.has_edge(u, v):
                    result.add_edge(u, v)
        print("To list:")
        print(result)
        return result


def build_links_graph():
    """In this function, we show how Graph data structure is used.

    Use your class to construct and return a graph with 10 nodes and all edges
    except 2->3 and 2->4.
    """
    g = NearlyCompleteGraph(10)
    g.del_edge(2, 3)
    g.del_edge(2, 4)
    print(g)
    return g


def main():
    g = build_links_graph()
    print(g.to_adj_lists())
    g = NearlyCompleteGraph(10)
    print(g.graph)
    print(g.has_edge(0, 8))
    g.add_node()
    g.del_edge(1, 2)
    print(g.graph)


if __name__ == "__main__":
    main()
