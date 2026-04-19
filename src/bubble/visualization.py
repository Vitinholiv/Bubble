"""Plotting utilities for Bubble simulations."""


import matplotlib.pyplot as plt
import numpy as np

from matplotlib.axes import Axes


def plot_edge_counts(
    edge_counts: np.ndarray,
    *,
    title: str = "Edges x Iteration",
    ax: Axes | None = None,
    show: bool = True,
) -> Axes:
    """Plot edge-count evolution across simulation steps.

    Parameters
    ----------
    edge_counts : np.ndarray
        Array of shape ``(n_steps+1, 4)`` where columns represent
        ``[L0↔L0, L0→L1, L1→L0, L1↔L1]``.
    title : str, optional
        Plot title.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created when *None*.
    show : bool, optional
        Whether to call ``plt.show()`` at the end.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    steps = np.arange(len(edge_counts))
    ax.plot(steps, edge_counts[:, 0], label=r"$L_0 \leftrightarrow L_0$", color="royalblue")
    ax.plot(steps, edge_counts[:, 3], label=r"$L_1 \leftrightarrow L_1$", color="crimson")
    ax.plot(
        steps,
        edge_counts[:, 1] + edge_counts[:, 2],
        label=r"$L_0 \leftrightarrow L_1$",
        color="indigo",
        linestyle="--",
    )

    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total Edges")
    ax.legend()

    if show:
        plt.show()

    return ax

def plot_bubble_burst(
    burst_values: np.ndarray,
    burst_metric_name: str = "Bubble Burst Metric",
    *,
    title: str = "Bubble Burst x Iteration",
    ax: Axes | None = None,
    show: bool = True,
    label: str | None = None,
    color: str | None = None
) -> Axes:
    """Plot bubble burst metric progression across simulation steps.

    Parameters
    ----------
    burst_values : np.ndarray
        Array of shape ``(n_steps+1)`` where each entry represents the value 
        of the burst metric at a given iteration.
    burst_metric_name : str, optional
        Name of the metric used, used for the label if 'label' is None.
    title : str, optional
        Plot title.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created when *None*.
    show : bool, optional
        Whether to call ``plt.show()`` at the end.
    label : str, optional
        Label for the plot line. Defaults to burst_metric_name.
    color : str, optional
        Color for the plot line. Defaults to "royalblue".

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    plot_label = label if label is not None else burst_metric_name
    plot_color = color if color is not None else "royalblue"

    steps = np.arange(len(burst_values))
    ax.plot(steps, burst_values, label=plot_label, color=plot_color)

    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Metric Value")
    
    if plot_label:
        ax.legend()

    if show:
        plt.show()

    return ax