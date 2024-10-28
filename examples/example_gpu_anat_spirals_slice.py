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

    get_operator("gpunufft")
except Exception:
    try:
        get_operator("finufft")
    except ValueError as e:
        raise ValueError("No NUFFT backend available") from e

    NUFFT_BACKEND = "finufft"
    COMPUTE_BACKEND = "numpy"

print(
    f"Using NUFFT backend: {NUFFT_BACKEND}", f"Using Compute backend: {COMPUTE_BACKEND}"
)
# %%

sim_conf = SimConfig(
    max_sim_time=3,
    seq=GreConfig(TR=50, TE=22, FA=12),
    hardware=default_hardware,
    fov_mm=(181, 217, 181),
    shape=(60, 72, 60),
)
sim_conf.hardware.n_coils = 1  # Update to get multi coil results.
sim_conf.hardware.field_strength = 7
phantom = Phantom.from_brainweb(sub_id=4, sim_conf=sim_conf, tissue_file="tissue_7T")


# %%
phantom.masks.shape

# %%
# Setting up Acquisition Pattern and Initializing Result file.
# ------------------------------------------------------------

# The next piece of simulation is the acquisition trajectory.
# Here nothing fancy, we are using a stack of spiral, that samples a 3D
# k-space, with an acceleration factor AF=4 on the z-axis.

sampler = StackOfSpiralSampler(
    accelz=1,
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

# engine = NufftAcquisitionEngine(model="simple", snr=30000, slice_2d=True)

# engine(
#     "example_spiral_2D.mrd",
#     sampler,
#     phantom,
#     sim_conf,
#     handlers=[noise_handler],
#     smaps=smaps,
#     worker_chunk_size=60,
#     n_workers=1,
#     nufft_backend=NUFFT_BACKEND,
# )
engine_t2s = NufftAcquisitionEngine(model="T2s", snr=30000, slice_2d=True)

engine_t2s(
    "example_spiral_t2s_2D.mrd",
    sampler,
    phantom,
    sim_conf,
    handlers=[noise_handler],
    worker_chunk_size=60,
    n_workers=1,
    nufft_backend=NUFFT_BACKEND,
)

# %%

from snake.mrd_utils import NonCartesianFrameDataLoader

with NonCartesianFrameDataLoader("example_spiral_t2s_2D.mrd") as data_loader:
    traj, kspace_data = data_loader.get_kspace_frame(0, shot_dim=True)

# %%
kspace_data = kspace_data.squeeze()

# %%
kspace_data.shape
traj[0].shape

# %%
kspace_data.shape

# %%
traj[0]

# %%
data_loader.shape

# %%
shot_debug = np.load("../debug_traj18.npy")
ksp_debug = np.load("../debug_ksp18.npy")

# %%
shot_debug.shape

# %%
shot_debug

# %%
nufft2 = get_operator(NUFFT_BACKEND)(samples=shot_debug[:,:2]*2*np.pi, shape=(60,72), density="voronoi", n_batchs=4)
adj_debug = nufft2.adj_op(ksp_debug)
plt.imshow(abs(adj_debug[0])) 

# %%
shot=traj[18].copy()
print(shot)
nufft = get_operator(NUFFT_BACKEND)(samples=shot[:,:2], shape=data_loader.shape[:-1], density=None, n_batchs=len(kspace_data))
nufft.samples = shot[:,:2]
image = nufft.adj_op(kspace_data)
nufft.shape

# %%

# %%
print(image.shape)
import numpy as np
image = np.moveaxis(image,0,-1)
image.shape

# %%

import matplotlib.pyplot as plt
from snake.toolkit.plotting import axis3dcut

# %%
from fmri.viz.images import tile_view

# %%
tile_view(image, samples=10)

# %%

# %%

# %%
