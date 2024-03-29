"""
Simulation and Reconstruction
=============================

This example shows how to simulate and reconstruct a simple 2D vds simulation.
"""

import matplotlib.pyplot as plt
import numpy as np
from snkf import H, SimData, list_handlers
from snkf.reconstructors import (
    ZeroFilledReconstructor,
    SequentialReconstructor,
    LowRankPlusSparseReconstructor,
)
from fmri.viz.images import tile_view

# %%
# Create The simulation
# ---------------------
# We create a simple simulation with 4 coils and a 64x64 image.
# ``lazy=True`` means that the data will be generated on the fly when needed.
# This is useful to avoid storing large data in memory.

sim = SimData.from_params(
    shape=(64, 64),
    sim_tr=0.1,
    sim_time=300,
    fov=(0.192, 0.192),
    n_coils=4,
    rng=42,
    lazy=True,
)

# %%
# Initialize the simulator
# ------------------------
# We initialize the simulator from the availables handlers

print(list_handlers())

simulator = (
    H["phantom-big"](roi_index=None)
    >> H["phantom-roi"]()
    >> H["activation-block"](
        block_on=20, block_off=20, duration=300, bold_strength=0.02
    )
    >> H["noise-gaussian"](snr=30)
    >> H["acquisition-vds"](
        acs=0.1,
        accel=2,
        accel_axis=1,
        order="center-out",
        shot_time_ms=25,
        constant=False,
    )
)

# run the simulation
sim = simulator(sim)

# %%
# Let's show the various sampling patterns
fig = tile_view(sim.smaps, axis=0, axis_label="c")
fig.suptitle("Smaps")

# %%
fig2 = tile_view(sim.kspace_mask, cmap="viridis", samples=0.1, axis=0)
fig2.suptitle("kspace mask")

# %%
# Reconstruction
# --------------
# Zero-Filled Reconstruction
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# Simple adjoint fourier transform.

adj_data = ZeroFilledReconstructor().reconstruct(sim)

fig3 = tile_view(abs(adj_data), samples=0.1, axis=0)

# %%
# Sequential Reconstruction
# ~~~~~~~~~~~~~~~~~~~~~~~~~

seq_data = SequentialReconstructor(
    max_iter_per_frame=20, threshold="sure", compute_backend="numpy"
).reconstruct(sim)
fig4 = tile_view(abs(seq_data), samples=0.1, axis=0)

# %%
# LowRank + Sparse Reconstruction
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
lr_s = LowRankPlusSparseReconstructor(
    lambda_l=0.1, lambda_s="sure", algorithm="otazo_raw", max_iter=20
).reconstruct(sim)

fig5 = tile_view(abs(lr_s), samples=0.1, axis=0)
plt.show()


# %%
# Compare the reconstructions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We define some utility plotting function to show time serie in voxel
def plot_ts_roi(arr, sim, roi_idx, ax=None, center=False, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_xlabel("time (s)")
    N = len(arr)
    time_samples = np.arange(N, dtype=np.float64)
    time_samples *= (
        sim.sim_tr if N == len(sim.data_ref) else sim.extra_infos["TR_ms"] / 1000
    )

    arr_ = np.abs(arr) if np.iscomplexobj(arr) else arr
    arr_ = arr_[:, sim.roi]

    ts = arr_[:, roi_idx] if roi_idx is not None else np.mean(arr_, axis=-1)
    if center:
        ts -= np.mean(ts, axis=0)
    ax.plot(time_samples, ts, **kwargs)
    return ax


def plot_ts_roi_many(arr_dict, sim, roi_idx, ax=None, center=False, **kwargs):
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
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), borderaxespad=0.0)

    return ax


# %%
# Time serie of the ROI
# ~~~~~~~~~~~~~~~~~~~~~
#

fig, ax = plt.subplots()
ax.set_title("Time serie of the ROI")
plot_ts_roi_many(
    {"acquired": sim.data_acq, "zero-filled": adj_data, "sequential": seq_data},
    sim,
    roi_idx=0,
    ax=ax,
    center=True,
)
