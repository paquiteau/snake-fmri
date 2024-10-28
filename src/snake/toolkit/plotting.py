"""Plotting utilities for the project."""

import matplotlib
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from mpl_toolkits.axes_grid1.axes_divider import Size, make_axes_locatable
from skimage.measure import find_contours
from matplotlib.cm import ScalarMappable


def get_coolgraywarm(thresh: float = 3, max: float = 7) -> matplotlib.colorbar.Colorbar:
    """Get a cool-warm colorbar, with gray inside the threshold."""
    coolwarm = matplotlib.colormaps["coolwarm"].resampled(256)
    newcolors = coolwarm(np.linspace(0, 1, 256))
    gray = np.array([0.8, 0.8, 0.8, 1])
    minthresh = int(128 + (thresh / max) * 128)
    maxthresh = int(128 - (thresh / max) * 128)
    newcolors[minthresh:maxthresh, :] = gray
    cool_gray_warm = matplotlib.colors.ListedColormap(newcolors)
    return cool_gray_warm


# %%
def get_axis_properties(
    array_bg: NDArray,
    cuts: tuple[int, ...],
    width_inches: float,
    cbar: bool = True,
    arr_pad: int = 4,
) -> tuple[
    NDArray,
    NDArray,
    tuple[tuple[slice, slice], ...],
    tuple[tuple[Any, Any, Any], ...],
]:
    """Generate mplt toolkit axes dividers."""
    slices = (np.s_[cuts[0], :, :], np.s_[:, cuts[1], :], np.s_[:, :, cuts[2]])
    bbox: list[tuple] = [(None, None), (None, None), (None, None)]
    for i in range(3):
        cut = array_bg[slices[i]]
        if cut.dtype != "bool":
            mask = abs(cut) > 0.5 * np.percentile(abs(cut), 95)
        else:
            mask = cut
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        rmin = max(0, rmin - arr_pad)
        rmax = min(rmax + arr_pad, mask.shape[0])
        cmin = max(0, cmin - arr_pad)
        cmax = min(cmax + arr_pad, mask.shape[1])
        bbox[i] = (slice(rmin, rmax), slice(cmin, cmax))
    hdiv, vdiv = _get_hdiv_vdiv(array_bg, bbox, slices, width_inches, cbar=cbar)

    return hdiv, vdiv, tuple(bbox), slices


def _get_hdiv_vdiv(
    array_bg: NDArray,
    bbox: tuple[tuple[slice]],
    slices: tuple[slice],
    width_inches: float,
    cbar: bool = False,
) -> tuple[NDArray, NDArray]:
    sizes = np.array([(bb.stop - bb.start) for b in bbox for bb in b])

    sizes = tuple(array_bg[s][b].shape for s, b in zip(slices, bbox, strict=False))
    alpha1 = sizes[1][1] / sizes[2][1]
    update_sizes = [[0, 0], [0, 0], [0, 0]]
    update_sizes[2][0] = sizes[2][0]
    update_sizes[2][1] = sizes[2][1]
    alpha1 = sizes[2][1] / sizes[1][1]
    update_sizes[1][0] = sizes[1][0] * alpha1
    update_sizes[1][1] = sizes[1][1] * alpha1
    alpha2 = (update_sizes[2][0] + update_sizes[1][0]) / sizes[0][0]
    update_sizes[0][0] = sizes[0][0] * alpha2
    update_sizes[0][1] = sizes[0][1] * alpha2

    aspect = update_sizes[0][0] / (update_sizes[0][1] + update_sizes[1][1])
    split_lr = update_sizes[0][1] / (update_sizes[1][1] + update_sizes[0][1])
    split_tb = update_sizes[1][0] / (update_sizes[1][0] + update_sizes[2][0])
    hdiv = [
        width_inches * split_lr,
        width_inches * (1 - split_lr),
    ]

    if cbar:
        hdiv.extend(
            [
                0.02 * hdiv[0],
                0.02 * hdiv[0],
            ]
        )
    np.array(hdiv)
    height_inches = width_inches * aspect
    vdiv = np.array([height_inches * split_tb, height_inches * (1 - split_tb)])
    return hdiv, vdiv


def get_mask_cuts_mask(mask: NDArray) -> tuple[int, ...]:
    """Get the optimal cut that expose maximum number of voxel in mask."""
    max_cuts = [0] * len(mask.shape)
    for i in range(len(max_cuts)):
        max_cuts[i] = int(np.argmax(np.sum(mask, axis=tuple(np.array([-2, -1]) + i))))
    return tuple(max_cuts)


def plot_frames_activ(
    background: NDArray,
    z_score: NDArray,
    roi: NDArray | None,
    ax: plt.Axes,
    slices: tuple[Any, ...],
    bbox: tuple[Any, ...],
    z_thresh: float = 3,
    z_max: float = 11,
    bg_cmap: str = "gray",
) -> tuple[plt.Axes, matplotlib.image.AxesImage]:
    """Plot activation maps and background.

    Parameters
    ----------
    background: 3D array
    z_score: 3D array
    roi: 3D array
    ax: plt.Axes

    """
    bg = background[slices][bbox].squeeze()
    im = ax.imshow(
        bg,
        vmin=np.min(background),
        vmax=np.max(background),
        cmap=bg_cmap,
        origin="lower",
        aspect="equal",
    )
    if z_score is not None:
        masked_z = z_score[slices][bbox].squeeze()
        masked_z[abs(masked_z) < z_thresh] = np.NaN
        im = ax.imshow(
            masked_z,
            alpha=1,
            cmap=get_coolgraywarm(z_thresh, max=z_max),
            vmin=-z_max,
            vmax=z_max,
            aspect="equal",
            interpolation="nearest",
            origin="lower",
        )
    if roi is not None:
        roi_cut = roi[slices][bbox].squeeze()
        contours = find_contours(roi_cut)
        for c in contours:
            ax.plot(c[:, 1], c[:, 0] - 0.5, c="cyan", label="ground-truth", linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax, im


def axis3dcut(
    fig: plt.Figure,
    ax: plt.Axes,
    background: NDArray,
    z_score: NDArray,
    gt_roi: NDArray | None = None,
    width_inches: float = 7,
    cbar: bool = True,
    cuts: tuple[int, ...] | None = None,
    bbox: tuple[tuple[Any, Any], ...] | None = None,
    slices: tuple[tuple[Any, Any, Any], ...] | None = None,
    bg_cmap: str = "gray",
    vmin_vmax: tuple[float] = None,
) -> tuple[plt.Figure, plt.Axes, tuple[int, ...]]:
    """Display a 3D image with zscore and ground truth ROI."""
    #    ax.axis("off")
    if cuts is None and gt_roi is not None:
        cuts_ = get_mask_cuts_mask(gt_roi)
        gt_roi_ = gt_roi
    elif cuts is not None and gt_roi is not None:
        cuts_ = cuts
        gt_roi_ = gt_roi
    elif cuts is None and gt_roi is None:
        raise ValueError("Missing gt_roi to compute ideal cuts.")
    elif cuts is not None and gt_roi is None:
        cuts_ = cuts
        gt_roi_ = None

    if bbox is None and slices is None:
        hdiv, vdiv, bbox_, slices_ = get_axis_properties(
            background, cuts_, width_inches, cbar=cbar
        )
    elif bbox is not None and slices is not None:
        hdiv, vdiv = _get_hdiv_vdiv(background, bbox, slices, width_inches, cbar=cbar)
        bbox_ = bbox
        slices_ = slices
    else:
        raise ValueError("Missing either bbox or slices.")
    divider = make_axes_locatable(ax)
    divider.set_horizontal([Size.Fixed(s) for s in hdiv])
    divider.set_vertical([Size.Fixed(s) for s in vdiv])
    axG: list[plt.Axes] = [None, None, None]
    for i, (nx, ny, ny1) in enumerate([(0, 0, 2), (1, 0, 1), (1, 1, 2)]):
        axG[i] = plt.Axes(fig, ax.get_position(original=True))
        axG[i].set_axes_locator(divider.new_locator(nx=nx, ny=ny, ny1=ny1))
        fig.add_axes(axG[i])
    for i in range(3):
        plot_frames_activ(
            background,
            z_score,
            gt_roi_,
            axG[i],
            slices_[i],
            bbox_[i],
            bg_cmap=bg_cmap,
        )

    if cbar:
        cax = type(ax)(fig, ax.get_position(original=True))
        cax.set_axes_locator(divider.new_locator(nx=3, ny=0, ny1=-1))
        if z_score is not None:
            im = ScalarMappable(norm="linear", cmap=get_coolgraywarm())
            im.set_clim(-11, 11)
            matplotlib.colorbar.Colorbar(cax, im, orientation="vertical")
            cax.set_ylabel("z-scores", labelpad=-20)
            cax.set_yticks(np.concatenate([-np.arange(3, 12, 2), np.arange(3, 12, 2)]))
        else:
            # use the background image
            if vmin_vmax is None:
                vmin, vmax = (np.min(background), np.max(background))
            else:
                vmin, vmax = vmin_vmax
            im = ScalarMappable(norm="linear", cmap=bg_cmap)
            im.set_clim(vmin=vmin, vmax=vmax)
            matplotlib.colorbar.Colorbar(cax, im, orientation="vertical")
        fig.add_axes(cax)

    ax.set_axes_locator(divider.new_locator(nx=0, ny=0, ny1=-1, nx1=-1))
    ax.set_zorder(10)
    ax.axis("off")
    # ax.set_xticks([])
    # ax.set_yticks([])
    return fig, ax, cuts_
