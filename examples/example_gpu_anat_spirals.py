# %%
"""
Compare Fourier Model and T2* Model for Stack of Spirals trajectory
===========================================

This examples walks through the elementary components of SNAKE.

Here we proceed step by step and use the Python interface. A more integrated
alternative is to use the CLI ``snake-main``

"""

# %%

# Imports
import numpy as np
from snake.core.simulation import SimConfig, default_hardware, GreConfig
from snake.core.phantom import Phantom
from snake.core.smaps import get_smaps
from snake.core.sampling import StackOfSpiralSampler

from mrinufft import get_operator


# For faster computation, try to use the GPU

NUFFT_BACKEND = "stacked-gpunufft"
COMPUTE_BACKEND = "cupy"

try:
    import cupy as cp

    if not cp.cupy.cuda.runtime.getDeviceCount():
        raise ValueError("No CUDA Device found")

    get_operator("stacked-gpunufft")
except Exception:
    try:
        get_operator("stacked-finufft")
    except ValueError as e:
        raise ValueError("No NUFFT backend available") from e

    NUFFT_BACKEND = "stacked-finufft"
    COMPUTE_BACKEND = "numpy"


# %%

sim_conf = SimConfig(
    max_sim_time=1,
    seq=GreConfig(TR=50, TE=22, FA=12),
    hardware=default_hardware,
    fov_mm=(181, 217, 181),
    shape=(60, 72, 60),
)
sim_conf.hardware.n_coils = 1  # Update to get multi coil results.
sim_conf.hardware.field_strength = 7
phantom = Phantom.from_brainweb(sub_id=4, sim_conf=sim_conf, tissue_file="tissue_7T")


# %%
# Setting up Acquisition Pattern and Initializing Result file.
# ------------------------------------------------------------

# The next piece of simulation is the acquisition trajectory.
# Here nothing fancy, we are using a stack of spiral, that samples a 3D
# k-space, with an acceleration factor AF=4 on the z-axis.

sampler = StackOfSpiralSampler(
    accelz=4,
    acsz=0.1,
    orderz="top-down",
    nb_revolutions=12,
    obs_time_ms=30,
    constant=True,
)

smaps = None
if sim_conf.hardware.n_coils > 1:
    smaps = get_smaps(sim_conf.shape, n_coils=sim_conf.hardware.n_coils)


# %%
# The acquisition trajectory looks like this
traj = sampler.get_next_frame(sim_conf)
print(traj.shape)
from mrinufft.trajectories.display import display_3D_trajectory

display_3D_trajectory(traj)

# %%
# Adding noise in Image
# ----------------------

from snake.core.handlers.noise import NoiseHandler

noise_handler = NoiseHandler(variance=0.01)

# %%
# Acquisition with Cartesian Engine
# ---------------------------------
#
# The generated file ``example_EPI.mrd`` does not contains any k-space data for
# now, only the sampling trajectory. let's put some in. In order to do so, we
# need to setup the **acquisition engine** that models the MR physics, and get
# sampled at the specified k-space trajectory.
#
# SNAKE comes with two models for the MR Physics:
#
# - model="simple" :: Each k-space shot acquires a constant signal, which is the
#   image contrast at TE.
# - model="T2s" :: Each k-space shot is degraded by the T2* decay induced by
#   each tissue.

# Here we will use the "simple" model, which is faster.
#
# SNAKE's Engine are capable of simulating the data in parallel, by distributing
# the shots to be acquired to a set of processes. To do so , we need to specify
# the number of jobs that will run in parallel, as well as the size of a job.
# Setting the job size and the number of jobs can have a great impact on total
# runtime and memory consumption.
#
# Here, we have a single frame to acquire with 60 frames (one EPI per slice), so
# a single worker will do.

from snake.core.engine import NufftAcquisitionEngine

engine = NufftAcquisitionEngine(model="simple", snr=30000)

engine(
    "example_spiral.mrd",
    sampler,
    phantom,
    sim_conf,
    handlers=[noise_handler],
    smaps=smaps,
    worker_chunk_size=60,
    n_workers=1,
    nufft_backend=NUFFT_BACKEND,
)
engine_t2s = NufftAcquisitionEngine(model="T2s", snr=30000)

engine_t2s(
    "example_spiral_t2s.mrd",
    sampler,
    phantom,
    sim_conf,
    handlers=[noise_handler],
    worker_chunk_size=60,
    n_workers=1,
    nufft_backend=NUFFT_BACKEND,
)

# %%
# Simple reconstruction
# ---------------------
#
# Getting k-space data is nice, but
# SNAKE also provides rudimentary reconstruction tools to get images (and check
# that we didn't mess up the acquisition process).
# This is available in the companion package ``snake.toolkit``.
#
# Loading the ``.mrd`` file to retrieve all information can be done using the
# ``ismrmd`` python package, but SNAKE provides convient dataloaders, which are
# more efficient, and take cares of managing underlying files access. As we are
# showcasing the API, we will do things manually here, and use only core SNAKE.

from snake.mrd_utils import NonCartesianFrameDataLoader
from snake.toolkit.reconstructors import (
    SequentialReconstructor,
    ZeroFilledReconstructor,
)

zer_rec = ZeroFilledReconstructor(
    nufft_backend=NUFFT_BACKEND, density_compensation=None
)
seq_rec = SequentialReconstructor(
    nufft_backend=NUFFT_BACKEND,
    density_compensation=None,
    max_iter_per_frame=30,
    threshold=2e-6,
    optimizer="fista",
    compute_backend=COMPUTE_BACKEND,
)
with NonCartesianFrameDataLoader("example_spiral.mrd") as data_loader:
    adjoint_spiral = abs(zer_rec.reconstruct(data_loader, sim_conf)[0])
    cs_spiral = abs(seq_rec.reconstruct(data_loader, sim_conf)[0])
with NonCartesianFrameDataLoader("example_spiral_t2s.mrd") as data_loader:
    adjoint_spiral_T2s = abs(zer_rec.reconstruct(data_loader, sim_conf)[0])
    cs_spiral_T2s = abs(seq_rec.reconstruct(data_loader, sim_conf)[0])


# %%
# Plotting the result
# -------------------

import matplotlib.pyplot as plt
from snake.toolkit.plotting import axis3dcut

fig, axs = plt.subplots(
    2,
    3,
    figsize=(19, 10),
    gridspec_kw=dict(wspace=0, hspace=0),
)


for ax, img, title in zip(
    axs[0],
    (adjoint_spiral, adjoint_spiral_T2s, abs(adjoint_spiral - adjoint_spiral_T2s)),
    ("simple", "T2s", "diff"),
):
    axis3dcut(fig, ax, img.T, None, None, cbar=True, cuts=(40, 40, 40), width_inches=4)
    ax.set_title(title)


for ax, img, title in zip(
    axs[1],
    (cs_spiral, cs_spiral_T2s, abs(cs_spiral - cs_spiral_T2s)),
    ("simple", "T2s", "diff"),
):
    axis3dcut(fig, ax, img.T, None, None, cbar=True, cuts=(40, 40, 40), width_inches=4)
    ax.set_title(title + " CS")


plt.show()

# %%

# %%
