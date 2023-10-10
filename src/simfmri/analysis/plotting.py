"""Plotting utilities."""
from __future__ import annotations
from typing import Mapping, Any, TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

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
