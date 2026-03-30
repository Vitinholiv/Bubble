"""Metrics for measuring filter-bubble phenomena."""

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

    initial_modularity : float
        The modularity of the graph at stage 0, used as a baseline for calculating change in modularity."""

def cross_group_connectivity(
    G: nx.Graph,
    words_per_node: tuple[int, int],
    initial_metric_value: float
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

def modularity_change(
    G: nx.Graph,
    words_per_node: tuple[int, int],
    initial_metric_value: float
) -> float:
    """Change in modularity relative to the initial stage.

    This serves as a proxy for "bubble burst": higher values indicate
    more interaction across group boundaries.

    Returns
    -------
    float
        Difference between the initial modularity and the current modularity.
    """

    # Safeguard to avoid division by zero
    if G.number_of_edges() == 0:
        return 0.0

    # Get the current modularity
    communities = [[], []]
    for n, d in G.nodes(data=True):
        communities[d.get('label')].append(n)

    current_modularity = modularity(G, communities)

    # Calculate the change in modularity
    modularity_change_value =  initial_metric_value - current_modularity

    return modularity_change_value

def assortativity_change(
    G: nx.Graph,
    words_per_node: tuple[int, int],
    initial_metric_value: float = 1.0
) -> float:
    """Change in assortativity relative to the initial stage.

    This serves as a proxy for "bubble burst": higher values indicate
    more interaction across group boundaries.

    Returns
    -------
    float
        Difference between the initial assortativity 
        and the current assortativity.
    """

    # Get the current assortativity
    current_assortativity = nx.attribute_assortativity_coefficient(G, "label")

    # Safeguard for nan values
    if math.isnan(current_assortativity):
        current_assortativity = 1.0

    # Calculate the change in assortativity
    assortativity_change_value = initial_metric_value - current_assortativity

    return assortativity_change_value
