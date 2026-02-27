"""Influencer selection strategies."""

import networkx as nx
import random


def select_by_max_degree(G: nx.Graph, num_influencers: int) -> list[int]:
    """Select influencers by highest degree, balanced across labels.

    Half of the influencers are drawn from label-0 nodes and the other
    half from label-1 nodes (by highest degree).

    Parameters
    ----------
    G : nx.Graph
        The social-network graph.  Each node must carry a ``'label'``
        attribute (0 or 1).
    num_influencers : int
        Total number of influencers to select.

    Returns
    -------
    list[int]
        Node ids of the selected influencers.
    """
    num_label0 = num_influencers // 2
    num_label1 = num_influencers - num_label0

    top_label0: list[tuple[int, int]] = [(-1, -1)] * num_label0
    top_label1: list[tuple[int, int]] = [(-1, -1)] * num_label1

    # Iterate through all nodes to find the top-degree nodes for each label
    # (This is more efficient than sorting all nodes by degree.)
    for node_id, attrs in G.nodes(data=True):
        degree = G.degree(node_id)
        label = attrs.get("label")

        if label == 0:
            if degree > top_label0[0][1]:
                top_label0[0] = (node_id, degree)
                top_label0.sort()
        elif label == 1:
            if degree > top_label1[0][1]:
                top_label1[0] = (node_id, degree)
                top_label1.sort()

    # return the node ids of the selected influencers, excluding any placeholders
    result_list = [node for node,_ in top_label0 if node != -1] + [
        node for node,_ in top_label1 if node != -1
    ]
    return result_list


def select_randomly(G: nx.Graph, num_influencers: int = 2)-> list[int]:
    nodes = list(G.nodes)
    k =  min(num_influencers, len(nodes))
    return random.sample(nodes, k = k)