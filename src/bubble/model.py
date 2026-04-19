"""Core BubbleModel simulation class."""

from __future__ import annotations
from typing import Any, Callable

import networkx as nx
import numpy as np
from networkx.algorithms.community import modularity

from bubble.visualization import plot_edge_counts as _plot_edge_counts, plot_bubble_burst as _plot_bubble_burst


class BubbleModel:
    """Simulate filter-bubble dynamics on a social-network graph.

    Parameters
    ----------
    hp : dict
        Hyper-parameter dictionary.  Expected keys:

        - ``num_nodes`` (int): Number of nodes in the graph.
        - ``words_per_node`` (tuple[int, int]): Words for label-0 / label-1.
        - ``affinity_level`` (float): Threshold for edge creation.
        - ``labels`` (array-like): Optional pre-assigned labels.
        - ``alphaL`` (array-like | None): Dirichlet parameter for label-0.
        - ``alphaR`` (array-like | None): Dirichlet parameter for label-1.
        - ``beta`` (float | array-like): Influencer susceptibility.
        - ``gamma`` (float | array-like): User susceptibility.
        - ``affinity`` (callable): ``(u, v) -> float`` affinity function.
        - ``influencer_selection`` (callable): ``(G, n) -> list[int]``.
        - ``influencer_scaling`` (callable): ``(n) -> int``.
        - ``bubble_burst_metric`` (callable): ``(G, (u,v)) -> int``.
        - ``bubble_burst_metric_name`` (string): ``Name of the bubble burst metric``.
    """

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #

    def __init__(self, hp: dict[str, Any]) -> None:
        self.stage: int = 0
        self.num_nodes: int = hp["num_nodes"]
        self.words_per_node: tuple[int, int] = tuple(hp["words_per_node"])  # type: ignore[arg-type]
        self.affinity_level: float = hp["affinity_level"]
        self.edge_count = np.zeros((1, 4), dtype=int)
        self.burst_metric_values = np.zeros((1), dtype=float)
        self.burst_metric_name = hp["bubble_burst_metric_name"]
        self.cross_edge_time = -1

        # --- Labels ---------------------------------------------
        if len(hp["labels"]) == self.num_nodes:
            self.labels = (np.array(hp["labels"]) != hp["labels"][0]).astype(bool)
        else:
            self.labels = np.random.randint(0, 2, self.num_nodes).astype(bool)

        # --- Dirichlet alphas -------------------------------------
        total_words = sum(self.words_per_node)

        if hp["alphaR"] is None or len(hp["alphaR"]) != total_words:
            self.alphaR = np.ones(self.words_per_node[1])
        else:
            self.alphaR = np.maximum(hp["alphaR"], np.zeros(self.words_per_node[1]))

        if hp["alphaL"] is None or len(hp["alphaL"]) != total_words:
            self.alphaL = np.ones(self.words_per_node[0])
        else:
            self.alphaL = np.maximum(hp["alphaL"], np.zeros(self.words_per_node[0]))

        # --- Building initial profile vectors ----------------------------
        self.p = np.zeros((self.num_nodes, total_words))
        for i in range(self.num_nodes):
            if self.labels[i] == 0:
                self.p[i] = np.concatenate(
                    [np.random.dirichlet(self.alphaL), np.zeros(self.words_per_node[1])]
                )
            else:
                self.p[i] = np.concatenate(
                    [np.zeros(self.words_per_node[0]), np.random.dirichlet(self.alphaR)]
                )

        # --- Building the Graph (Affinity Network) -----------------------------------------------
        self.G = nx.Graph()
        self.G.add_nodes_from(
            [(i, {"p": self.p[i], "label": self.labels[i]}) for i in range(self.num_nodes)]
        )

        # --- Scalar / vectorized parameters ------------------
        self.beta = (
            np.full(self.num_nodes, hp["beta"])
            if np.isscalar(hp["beta"])
            else np.asarray(hp["beta"])
        )
        self.gamma = (
            np.full(self.num_nodes, hp["gamma"])
            if np.isscalar(hp["gamma"])
            else np.asarray(hp["gamma"])
        )

        # --- Strategy callables ---------------------------------------
        self.affinity: Callable = hp["affinity"]
        self.influencer_selection: Callable = hp["influencer_selection"]
        self.bubble_burst_metric: Callable = hp['bubble_burst_metric']
        self.influencer_scaling = hp['influencer_scaling']
        self.num_influencers: int = self.influencer_scaling(self.num_nodes)

        # --- Initial edges --------------------------------------------
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                self.update_edge(i, j)

    # ------------------------------------------------------------- #
    #  Edge management
    # ------------------------------------------------------------ #

    def update_edge(self, node1: int, node2: int) -> None:
        """Add or remove the edge between *node1* and *node2* based on affinity."""
        v1 = self.G.nodes[node1]["p"]
        v2 = self.G.nodes[node2]["p"]
        aff = self.affinity(v1, v2)

        if aff >= self.affinity_level and not self.G.has_edge(node1, node2):
            self.G.add_edge(node1, node2)
            target = self.edge_count[self.stage]
            # Maps (label1,label2) to column index of edge_count: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
            target[self.labels[node1] + 2 * self.labels[node2]] += 1
            if self.cross_edge_time == -1 and self.labels[node1] != self.labels[node2]:
                self.cross_edge_time = self.stage

        elif aff < self.affinity_level and self.G.has_edge(node1, node2):
            self.G.remove_edge(node1, node2)
            target = self.edge_count[self.stage]
            target[self.labels[node1] + 2 * self.labels[node2]] -= 1

    # ---------------------------------------------------------- #
    #  Simulation loop
    # -------------------------------------------------------------- #

    def iteration(self, msg0: np.ndarray, msg1: np.ndarray) -> None:
        """Execute a single simulation iteration.

        Parameters
        ----------
        msg0 : np.ndarray
            Message broadcast to label-0 influencers.
        msg1 : np.ndarray
            Message broadcast to label-1 influencers.
        """
        total_words = int(np.sum(self.words_per_node))

        # Step 1: influencer update
        self.influencer_nodes: list[int] = self.influencer_selection(
            self.G, self.num_influencers
        )
        for i in self.influencer_nodes:
            msg = msg0 if self.G.nodes[i]["label"] == 0 else msg1
            self.G.nodes[i]["p"] = (1 - self.beta[i]) * self.G.nodes[i]["p"] + self.beta[i] * msg

        # Step 2: user profile update
        for i in range(self.num_nodes):
            if i in self.influencer_nodes:
                continue
            neighbours = [j for j in self.influencer_nodes if self.G.has_edge(i, j)]
            if neighbours:
                profile_sum = np.zeros(total_words)
                for k in neighbours:
                    profile_sum += self.G.nodes[k]["p"] / len(neighbours)
            else:
                profile_sum = self.G.nodes[i]["p"]
            self.G.nodes[i]["p"] = (
                (1 - self.gamma[i]) * self.G.nodes[i]["p"] + self.gamma[i] * profile_sum
            )

        # Step 3: edge update
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                self.update_edge(i, j)

    def run(
        self,
        n: int,
        msg0: np.ndarray,
        msg1: np.ndarray,
    ) -> nx.Graph:
        """Run *n* iterations of the simulation.

        Parameters
        ----------
        n : int
            Number of iterations.
        msg0, msg1 : np.ndarray
            Messages for label-0 and label-1 influencers respectively.

        Returns
        -------
        nx.Graph
            The graph after all iterations.
        """
        # Initialize edge count tracking for all stages as a 2D array of shape (n+1 (time), 4 (edge type)) to store counts for each edge type at each stage
        temp_edge_count = self.edge_count[0].copy()
        self.edge_count = np.zeros((n+1, 4), dtype=int)
        self.edge_count[0] = temp_edge_count

        temp_burst = self.burst_metric_values[0]
        self.burst_metric_values = np.zeros(n+1, dtype=float)
        self.burst_metric_values[0] = temp_burst

        self.burst_metric_values[0] = self.bubble_burst_metric(self.G, self.words_per_node)

        for i in range(n):
            self.stage = i + 1
            self.edge_count[self.stage] = self.edge_count[i].copy()
            self.iteration(msg0, msg1)
            self.burst_metric_values[self.stage] = self.bubble_burst_metric(self.G, self.words_per_node)

        return self.G

    # ------------------------------------------------------------------ #
    #  Visualization helpers
    # ------------------------------------------------------------------ #

    def plot_edge_counts(self, **kwargs) -> None:
        """Plot the edge-count evolution (delegates to :func:`bubble.visualization.plot_edge_counts`)."""
        _plot_edge_counts(self.edge_count, **kwargs)

    def plot_bubble_burst(self, **kwargs) -> None:
        """Plot the bubble burst metric progression (delegates to :func:`bubble.visualization.plot_bubble_burst`)."""
        _plot_bubble_burst(self.burst_metric_values, **kwargs)