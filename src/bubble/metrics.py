"""Metrics for measuring filter-bubble phenomena."""
import warnings
import networkx as nx
import math
from networkx.algorithms.community import modularity

"""
    Parameters
    ----------
    G : nx.Graph
        The social-network graph.  Each node must have a ``'label'``
        attribute (0 or 1).

    words_per_node : tuple[int, int]
        Number of words for label-0 and label-1 respectively.
"""

def cross_group_connectivity(
    G: nx.Graph,
    words_per_node: tuple[int, int]
) -> float:
    """Fraction of realized cross-group edges relative to the maximum possible.

    This serves as a proxy for "bubble burst": higher values indicate
    more interaction across group boundaries.

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

    cross_edges = sum(1 for u, v in G.edges() if G.nodes[u]["label"] != G.nodes[v]["label"])

    return float(cross_edges) / total_possible

def modularity_value(
    G: nx.Graph,
    words_per_node: tuple[int, int]
) -> float:
    """Calculates the modularity of the graph based on the group labels.

    This serves as a measure of structural segregation: higher values indicate 
    that the groups are isolated and internally dense, while lower values
    indicate a well-mixed network.

    Returns
    -------
    float
        Modularity value.
    """

    # Safeguard to avoid division by zero
    if G.number_of_edges() == 0:
        return 0.0

    communities = [[], []]
    for n, d in G.nodes(data=True):
        communities[int(d.get('label'))].append(n)

    current_modularity = modularity(G, communities)

    return current_modularity

def assortativity_value(
    G: nx.Graph,
    words_per_node: tuple[int, int]
) -> float:
    """Calculates the attribute assortativity coefficient for the labels.

    This serves as a proxy for "bubble burst": higher values indicate
    more interaction between members of the same group, where lower values
    indicate cross-group relations.

    Returns
    -------
    float
        Assortativity value.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        current_assortativity = nx.attribute_assortativity_coefficient(G, "label")
    
    if math.isnan(current_assortativity):
        return 1.0
    
    return current_assortativity
