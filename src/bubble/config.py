"""Default configuration and model factory."""

import copy
from typing import Any

import numpy as np

from bubble.affinity import cosine_similarity
from bubble.metrics import cross_group_connectivity, modularity_change, assortativity_change
from bubble.selection import select_by_max_degree
from bubble.scaling import constant_scaling, linear_scaling, log_scaling, power_law_scaling, sqrt_scaling

metric_options = {
    'cross_group_connectivity': cross_group_connectivity,
    'modularity_change': modularity_change,
    'assortativity_change': assortativity_change
}

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
    "influencer_scaling": constant_scaling(4),
    "bubble_burst_metric_name": 'cross_group_connectivity',
    "bubble_burst_metric": metric_options['cross_group_connectivity']
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
        if 'bubble_burst_metric_name' in overrides:
            new_name = overrides['bubble_burst_metric_name']
            config['bubble_burst_metric'] = metric_options.get(
                new_name, 
                config['bubble_burst_metric']
            )
    return config
