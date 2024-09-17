# %%
"""
Compare Fourier Model and T2* Model for EPI
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
from snake.core.sampling import EPI3dAcquisitionSampler
from snake.mrd_utils import make_base_mrd

# %%

sim_conf = SimConfig(
    max_sim_time=6,
    seq=GreConfig(TR=100, TE=30, FA=3),
    hardware=default_hardware,
    fov_mm=(181, 217, 181),
    shape=(60, 72, 60),
)
sim_conf.hardware.n_coils = 8

phantom = Phantom.from_brainweb(sub_id=4, sim_conf=sim_conf, tissue_file="tissue_7T")


# %%
# Setting up Acquisition Pattern and Initializing Result file.
# ------------------------------------------------------------

# The next piece of simulation is the acquisition trajectory.
# Here nothing fancy, we are using a EPI (fully sampled), that samples a 3D
# k-space (this akin to the 3D EPI sequence of XXXX)

sampler = EPI3dAcquisitionSampler(accelz=1, acsz=0.1, orderz="top-down")

smaps = None
if sim_conf.hardware.n_coils > 1:
    smaps = get_smaps(sim_conf.shape, n_coils=sim_conf.hardware.n_coils)


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

from snake.core.engine import EPIAcquisitionEngine

engine = EPIAcquisitionEngine(model="simple")

make_base_mrd("example_EPI.mrd", sampler, phantom, sim_conf, smaps=smaps)
make_base_mrd("example_EPI_t2s.mrd", sampler, phantom, sim_conf, smaps=smaps)

engine("example_EPI.mrd", worker_chunk_size=60, n_workers=1)
engine_t2s = EPIAcquisitionEngine(model="T2s")

engine_t2s("example_EPI_t2s.mrd", worker_chunk_size=60, n_workers=1)

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

from snake.mrd_utils import CartesianFrameDataLoader
from scipy.fft import ifftn, ifftshift, fftshift


def reconstruct_frame(filename):
    with CartesianFrameDataLoader(filename) as data_loader:
        mask, kspace_data = data_loader.get_kspace_frame(0)
    axes = (-3, -2, -1)
    image_data = ifftshift(
        ifftn(fftshift(kspace_data, axes=axes), axes=axes, norm="ortho"), axes=axes
    )

    # Take the square root sum of squares to get the magnitude image (SSOS)
    image_data = np.sqrt(np.sum(np.abs(image_data) ** 2, axis=0))

    return image_data.squeeze().T


image_simple = reconstruct_frame("example_EPI.mrd")
image_T2s = reconstruct_frame("example_EPI_t2s.mrd")


# %%
# Plotting the result
# -------------------

import matplotlib.pyplot as plt
from snake.toolkit.plotting import axis3dcut

fig, axs = plt.subplots(1, 3, figsize=(30, 10))

for ax, img, title in zip(
    axs,
    (image_simple, image_T2s, abs(image_simple - image_T2s)),
    ("simple", "T2s", "diff"),
):
    axis3dcut(fig, ax, img, None, None, cbar=True, cuts=(40, 40, 40), width_inches=4)
    ax.set_title(title)

plt.show()

# %%

# %%
