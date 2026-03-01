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
