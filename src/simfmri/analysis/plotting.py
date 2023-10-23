"""Plotting utilities."""
from __future__ import annotations
from typing import Mapping, Any, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from simfmri.simulation import SimData

from matplotlib.lines import Line2D
from matplotlib.offsetbox import (
    AnchoredOffsetbox,
    TextArea,
    HPacker,
    VPacker,
    AuxTransformBox,
    PaddedBox,
)


def init_xy_plot(
    xlabel: str, ylabel: str, diag: bool = True, ax: Axes = None
) -> tuple[Figure, Axes]:
    """Initialize a square plot, with value in [0,1].

    Ideal for ROC curve.
    """
    if ax is None:
        fig, ax = plt.subplots()
    eps = 0.05
    ax.set_xlim(-eps, 1 + eps)
    ax.set_ylim(-eps, 1 + eps)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid("on")
    if diag:
        ax.plot([[0, 0], [1, 1]], c="gray", ls="dashed", lw=1)
    return fig, ax


def table_legend(
    ax: Axes,
    handles: list[Artist],
    colnames: list[str],
    rows: list[list[str]],
    **kwargs: Mapping[str, Any],
) -> Artist:
    """Create a multicolumn legend.

    The first colums is the marker/line, the rest are provided in rows.

    Parameters
    ----------
    ax: plt.Axes
        Figure Axis
    handles: list[Artist]
        List of handles, if None, use ax.get_legend_handles_label.
    cols: list[str]
        List of column name
    rows: list[list[Any]]
        list of row, each row being a list of string.
    **kwargs: Mapping[str, any]
        Extra arguments for the Anchor Box (similar to plt.legend)
    """
    table = []
    table.append(
        [TextArea("")]
        + [TextArea(colname, textprops={"fontweight": "bold"}) for colname in colnames]
    )
    for row in rows:
        table.append([None] + [TextArea(r) for r in row])
    for i, h in enumerate(handles):
        x = AuxTransformBox(ax.transAxes)
        x.add_artist(
            Line2D([0, 0.08], [0, 0], color=h.get_color(), ls=h.get_linestyle())
        )
        # Add a padding around the line, to match the row height.
        table[i + 1][0] = PaddedBox(x, pad=table[i + 1][1]._text.get_fontsize() // 2)

    # Vpack the column elemement, and stack them horizontally together.
    # This ensure proper alignemnent per column.
    packer = HPacker(
        children=[
            VPacker(
                children=[table[i][j] for i in range(len(table))], sep=2, align="top"
            )
            for j in range(len(table[0]))
        ],
        sep=2,
    )
    artist = ax.add_artist(AnchoredOffsetbox(child=packer, **kwargs))
    return artist


def plot_ts_roi(
    arr: np.ndarray,
    sim: SimData,
    roi_idx: int,
    ax: Axes | None = None,
    center: bool = False,
    **kwargs: Mapping[str, Any],
) -> Axes:
    """Plot a time series of a given ROI.

    Parameters
    ----------
    arr: np.ndarray
        Array to plot
    sim: SimData
        Simulation object
    roi_idx: int
        Index of the ROI voxel to plot.
    ax: plt.Axes
        Figure Axis
    center: bool
        If True, center the time series.
    **kwargs: Mapping[str, any]
        Extra arguments for the plot function.
    """
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_xlabel("time (s)")
    N = len(arr)
    time_samples = np.arange(N, dtype=np.float64)
    time_samples *= (
        sim.sim_tr if N == len(sim.data_ref) else sim.extra_infos["TR_ms"] / 1000
    )

    arr_ = np.abs(arr) if np.iscomplexobj(arr) else arr
    arr_ = arr_[:, sim.roi > 0.5]

    if roi_idx is None:
        ts = np.mean(arr_, axis=-1)
        if center:
            mean_val = np.mean(ts, axis=0)
            ts -= mean_val
            ax.plot(time_samples, ts, **kwargs)
    else:
        ts = arr_[:, roi_idx]
        if center:
            ts -= np.mean(ts, axis=0)
        ax.plot(time_samples, ts, **kwargs)
    return ax


def plot_ts_roi_many(
    arr_dict: Mapping[str, np.ndarray],
    sim: SimData,
    roi_idx: int,
    ax: Axes = None,
    center: bool = False,
    **kwargs: Mapping[str, Any],
) -> Axes:
    """Plot many time series of a given ROI.

    Parameters
    ----------
    arr_dict: dict[str, np.ndarray]
        Dictionary of arrays to plot. The key is the label.
    sim: SimData
        Simulation object
    roi_idx: int
        Index of the ROI voxel to plot.
    ax: plt.Axes
        Figure Axis
    center: bool
        If True, center the time series.
    **kwargs: Mapping[str, any]
        Extra arguments for the plot function.
    """
    if ax is None:
        fig, ax = plt.subplots()
    for label, arr in arr_dict.items():
        if label == "acquired":
            kwargs2 = kwargs.copy()
            kwargs2["c"] = "gray"
            kwargs2["alpha"] = 0.5
            kwargs2["zorder"] = 0
            plot_ts_roi(arr, sim, roi_idx, ax=ax, label=label, center=center, **kwargs2)
        else:
            plot_ts_roi(arr, sim, roi_idx, ax=ax, label=label, center=center, **kwargs)
    ax.legend()

    return ax
