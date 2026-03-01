"""Default configuration and model factory."""

import copy
from typing import Any

import numpy as np

from bubble.affinity import cosine_similarity
from bubble.metrics import cross_group_connectivity
from bubble.selection import select_by_max_degree

DEFAULT_CONFIG: dict[str, Any] = {
    "num_nodes": 20,
    "words_per_node": (4, 4),
    "affinity_level": 0.1,
    "labels": [],
    "alphaL": None,
    "alphaR": None,
    "beta": 0.3,
    "gamma": 0.2,
    "affinity": cosine_similarity,
    "influencer_selection": select_by_max_degree,
    "num_influencers": 4,
    "bubble_burst": cross_group_connectivity,
}


def create_model(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a copy of :data:`DEFAULT_CONFIG` with *overrides* applied.

    Parameters
    ----------
    overrides : dict, optional
        Key-value pairs that replace the defaults.

    Returns
    -------
    dict
        A new configuration dictionary ready to pass to
        :class:`~bubble.model.BubbleModel`.
    """
    config = copy.deepcopy(DEFAULT_CONFIG)
    if overrides:
        config.update(overrides)
    return config
