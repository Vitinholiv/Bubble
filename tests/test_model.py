"""Smoke tests for BubbleModel."""

import numpy as np

from bubble import BubbleModel, DEFAULT_CONFIG, create_model
from bubble.affinity import cosine_similarity, dot_product
from bubble.messages import opposite_uniform_message, opposite_unique_message
from bubble.metrics import cross_group_connectivity
from bubble.selection import select_by_max_degree


def test_default_model_runs():
    """BubbleModel initializes and runs without errors using default config."""
    np.random.seed(0)
    model = BubbleModel(DEFAULT_CONFIG)
    wpn = model.words_per_node
    graph = model.run(5, opposite_uniform_message(wpn, 0), opposite_uniform_message(wpn, 1))
    assert len(graph.nodes) == DEFAULT_CONFIG["num_nodes"]


def test_create_model_override():
    """create_model correctly overrides defaults."""
    cfg = create_model({"num_nodes": 10, "beta": 0.5})
    assert cfg["num_nodes"] == 10
    assert cfg["beta"] == 0.5
    assert cfg["gamma"] == DEFAULT_CONFIG["gamma"]  # unchanged


def test_cosine_similarity():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_similarity(a, b) == 0.0
    assert cosine_similarity(a, a) == 1.0


def test_dot_product():
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    assert dot_product(a, b) == 11.0


def test_uniform_message_shape():
    wpn = (4, 4)
    msg = opposite_uniform_message(wpn, 0)
    assert msg.shape == (8,)
    assert np.isclose(msg.sum(), 1.0)


def test_unique_message_shape():
    np.random.seed(42)
    wpn = (4, 4)
    msg = opposite_unique_message(wpn, 1)
    assert msg.shape == (8,)
    assert np.isclose(msg.sum(), 1.0)
    assert np.count_nonzero(msg) == 1


def test_select_by_max_degree():
    """Selection returns at most num_influencers nodes."""
    np.random.seed(0)
    model = BubbleModel(DEFAULT_CONFIG)
    influencers = select_by_max_degree(model.G, 3)
    assert len(influencers) <= 3


def test_cross_group_connectivity_range():
    np.random.seed(0)
    model = BubbleModel(DEFAULT_CONFIG)
    cg = cross_group_connectivity(model.G, model.words_per_node, 1.0)
    assert 0.0 <= cg <= 1.0


def test_edge_count_shape():
    np.random.seed(0)
    model = BubbleModel(DEFAULT_CONFIG)
    wpn = model.words_per_node
    model.run(3, opposite_uniform_message(wpn, 0), opposite_uniform_message(wpn, 1))
    assert model.edge_count.shape == (4, 4)  # n+1 rows x 4 columns


def test_burst_metric_values_shape():
    """burst_metric_values has one entry per stage (n+1) after run."""
    np.random.seed(0)
    model = BubbleModel(DEFAULT_CONFIG)
    wpn = model.words_per_node
    n = 3
    model.run(n, opposite_uniform_message(wpn, 0), opposite_uniform_message(wpn, 1))
    assert model.burst_metric_values.shape == (n + 1,)


def test_initial_profiles_normalized():
    """Each node's initial profile vector sums to 1.0."""
    np.random.seed(0)
    model = BubbleModel(DEFAULT_CONFIG)
    for i in range(model.num_nodes):
        assert np.isclose(model.p[i].sum(), 1.0)
