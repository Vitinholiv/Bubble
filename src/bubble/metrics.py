"""Metrics for measuring filter-bubble phenomena."""

import networkx as nx


def cross_group_connectivity(
    G: nx.Graph,
    words_per_node: tuple[int, int],
) -> float:
    """Fraction of realised cross-group edges relative to the maximum possible.

    This serves as a proxy for "bubble burst": higher values indicate
    more interaction across group boundaries.

    Parameters
    ----------
    G : nx.Graph
        The social-network graph.  Each node must have a ``'label'``
        attribute (0 or 1).
    words_per_node : tuple[int, int]
        Number of words for label-0 and label-1 respectively.

    Returns
    -------
    float
        Ratio in [0, 1].
    """
    counts = {0: 0, 1: 0}
    for _, attrs in G.nodes(data=True):
        label = attrs.get("label")
        counts[label] += 1

    total_possible = counts[0]*counts[1]
    if total_possible == 0:
        return 0.0

    cross_edges = sum( 1 for u,v in G.edges() if G.nodes[u]["label"] != G.nodes[v]["label"])

    return cross_edges / total_possible
