"""
Shortest Paths

@author: Vlad Niculae <v.niculae@uva.nl>
Djamel Zair 11015934

"""

from typing import List, Optional, Tuple
import heapq

INF = float("inf")


def reconstruct_path(predecessor: List[Optional[int]], v: int) -> List[int]:
    """Computes the path from root to v in the given arborescence.

    `predecessor` is the predecessor list representation of an arborescence
    rooted at a source node, such that `predecessor[v] == u` if the edge u -> v
    is in the arborescence, and `predecessor[v] is None` if v is the root or if
    there is no path to v.

    If there is no path to v, return the single-node path [v].


    """
    if predecessor[v] is None:
        return [v]

    path_list = []

    while v is not None:
        path_list.append(v)
        v = predecessor[v]
    recon_path = path_list[::-1]
    print("the path list = ", path_list)
    print("reversed is the recon_path, ", recon_path)
    return recon_path


class AdjListsWeightedGraph:

    _adj: List[List[Tuple[int, float]]] = []

    def __init__(self, n_nodes: int):
        self.n_nodes = n_nodes
        self._adj = [[] for _ in range(n_nodes)]

    def get_edge(self, u: int, v: int) -> Optional[float]:
        """If an edge u->v exists, return the weight of this edge.
        Otherwise, return None."""

        for vp, w in self._adj[u]:
            if vp == v:
                return w

        return None

    def add_edge(self, u: int, v: int, weight: float):
        """If an edge u->v does not exist, add it, with the given weight."""
        if self.get_edge(u, v) is None:
            self._adj[u].append((v, weight))
        else:
            raise ValueError("Edge already exists.")

    def del_edge(self, u: int, v: int):
        """If an edge u->v exists, delete it; otherwise, do nothing."""
        self._adj[u] = [
            (vp, weight) for (vp, weight) in self._adj[u] if vp != v
        ]

    def dijkstra(self, source: int) -> Tuple[List[Optional[int]], List[float]]:
        """Compute shortest paths from the given source to all other nodes.

        Returns:

        predecessors: a list of length `n_nodes`, containing, for each node,
        the id of its predecessor, or None if it is the root. (This list is the
        first argument to `reconstruct_path`).

        weights: a list of length `n_nodes`, containing, for each node, the
        length (total weight) of the path from the source to that node.
        (or INF for nodes that cannot be reached.)

        """

        unvisited = [True] * self.n_nodes  # all nodes are initially unvisited
        dist = [INF] * self.n_nodes
        pred = [None] * self.n_nodes  # predecessor list
        dist[source] = 0  # Creating  [0, INF, INF, etc]

        # Initialize priority queue with source node, zero cost
        pq = [(0, source)]

        while pq:
            # Get node with smallest distance from priority queue the min-heap way
            (_, u) = heapq.heappop(pq)

            # Check if we already found the shortest path to u
            if not unvisited[u]:
                continue

            # Mark node as visited
            unvisited[u] = False

            # Relax edges going out from u
            for (v, w) in self._adj[u]:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    pred[v] = u
                    heapq.heappush(pq, (dist[v], v))

        return (pred, dist)


def main():
    gr = AdjListsWeightedGraph(n_nodes=5)
    gr.add_edge(0, 1, 10)
    gr.add_edge(1, 2, 1)
    gr.add_edge(0, 3, 5)
    gr.add_edge(3, 4, 2)
    gr.add_edge(4, 0, 7)
    gr.add_edge(1, 3, 2)
    gr.add_edge(3, 1, 3)
    gr.add_edge(2, 4, 4)
    gr.add_edge(4, 2, 6)

    pred, dist = gr.dijkstra(source=0)

    print("Distances: ", dist)
    print("Paths:")

    for v in range(gr.n_nodes):
        print(reconstruct_path(pred, v))


if __name__ == "__main__":
    main()
