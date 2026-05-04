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


def select_intermediation(G: nx.Graph, num_influencers: int) -> list[int]:
    """Select influencers by highest betweenness centrality, balanced across labels.

    Half of the influencers are drawn from label-0 nodes and the other
    half from label-1 nodes (by highest betweenness centrality).

    Parameters
    ----------
    G : nx.Graph
        The social-network graph. Each node must carry a ``'label'``
        attribute (0 or 1).
    num_influencers : int
        Total number of influencers to select.

    Returns
    -------
    list[int]
        Node ids of the selected influencers.
    """
    centrality = nx.betweenness_centrality(G)

    num_label0 = num_influencers // 2
    num_label1 = num_influencers - num_label0

    nodes_label0 = []
    nodes_label1 = []

    for node_id, attrs in G.nodes(data=True):
        label = attrs.get("label")
        score = centrality[node_id]
        
        if label == 0:
            nodes_label0.append((score, node_id))
        elif label == 1:
            nodes_label1.append((score, node_id))

    nodes_label0.sort(key=lambda x: x[0], reverse=True)
    nodes_label1.sort(key=lambda x: x[0], reverse=True)

    selected_label0 = [node_id for score, node_id in nodes_label0[:num_label0]]
    selected_label1 = [node_id for score, node_id in nodes_label1[:num_label1]]

    return selected_label0 + selected_label1

def select_unbalanced_intermediation(G: nx.Graph, num_influencers: int) -> list[int]:
    """Select influencers by highest absolute betweenness centrality across the entire graph.

    This strategy ignores node labels and simply picks the nodes with the 
    highest betweenness centrality, regardless of which community they belong to.

    Parameters
    ----------
    G : nx.Graph
        The social-network graph.
    num_influencers : int
        Total number of influencers to select.

    Returns
    -------
    list[int]
        Node ids of the selected influencers.
    """
    centrality = nx.betweenness_centrality(G)
    sorted_nodes = sorted(centrality.items(), key=lambda item: item[1], reverse=True)
    selected = [node_id for node_id, score in sorted_nodes[:num_influencers]]
    
    return selected