# %%
"""
Creating motion artifacts on an anatomical EPI with SNAKE-fMRI
==============================================================

This examples walks through the elementary components of SNAKE and demonstrates how
to add motion artifacts to the simulation.

"""

# %%

# Imports
import numpy as np
import matplotlib.pyplot as plt

from snake.core.phantom import Phantom
from snake.core.sampling import EPI3dAcquisitionSampler
from snake.core.simulation import GreConfig, SimConfig, default_hardware

# %%
# Setting up the base simulation Config.  This configuration holds all key
# parameters for the simulation, describing the scanner parameters.

sim_conf = SimConfig(
    max_sim_time=6,
    seq=GreConfig(TR=50, TE=30, FA=3),
    hardware=default_hardware,
)
sim_conf.hardware.n_coils = 8
sim_conf.fov.res_mm = (3, 3, 3)
sim_conf

# %%
# Creating the base Phantom
# -------------------------
#
# The simulation acquires
# the data describe in a phantom. A phantom consists of fuzzy segmentation of
# head tissue, and their MR intrinsic parameters
# (density, T1, T2, T2*,  magnetic susceptibilities)
#
# Here we use Brainweb reference mask and values for convenience.

phantom = Phantom.from_brainweb(
    sub_id=4, sim_conf=sim_conf, output_res=1, tissue_file="tissue_7T"
)

# Here are the tissue availables and their parameters
phantom.affine


# %%
# Adding motion to the Phantom
# ----------------------------
# The motion is added to the phantom by applying a transformation to the
# simulation's FOV configuration (as if the phantom was moving in the scanner).

from snake.core.handlers import RandomMotionImageHandler

motion = RandomMotionImageHandler(
    ts_std_mms=[1, 1, 1],
    rs_std_degs=[0.1, 0.1, 0.1],
)

motion2 = RandomMotionImageHandler(
    ts_std_mms=[3, 3, 3],
    rs_std_degs=[0.5, 0.5, 0.5],
)

# %%
# Setting up Acquisition Pattern and Initializing Result file.
# ------------------------------------------------------------

# The next piece of simulation is the acquisition trajectory.
# Here nothing fancy, we are using a EPI (fully sampled), that samples a 3D
# k-space (this akin to the 3D EPI sequence of XXXX)

sampler = EPI3dAcquisitionSampler(accelz=1, acsz=0.1, orderz="top-down")


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

engine(
    "example_nomotion.mrd",
    sampler=sampler,
    phantom=phantom,
    handlers=[],
    sim_conf=sim_conf,
    worker_chunk_size=16,
    n_workers=4,
)

engine(
    "example_motion.mrd",
    sampler=sampler,
    phantom=phantom,
    handlers=[motion],
    sim_conf=sim_conf,
    worker_chunk_size=16,
    n_workers=4,
)

engine(
    "example_motion2.mrd",
    sampler=sampler,
    phantom=phantom,
    handlers=[motion2],
    sim_conf=sim_conf,
    worker_chunk_size=16,
    n_workers=4,
)

# %%
# Simple reconstruction
# ---------------------

from snake.mrd_utils import CartesianFrameDataLoader
from snake.toolkit.reconstructors import ZeroFilledReconstructor


with CartesianFrameDataLoader("example_nomotion.mrd") as data_loader:
    rec = ZeroFilledReconstructor(n_jobs=1)
    rec_nomotion = rec.reconstruct(data_loader).squeeze()

with CartesianFrameDataLoader("example_motion.mrd") as data_loader:
    rec = ZeroFilledReconstructor(n_jobs=1)
    rec_motion = rec.reconstruct(data_loader).squeeze()
    motion = data_loader.get_dynamic(0)

with CartesianFrameDataLoader("example_motion2.mrd") as data_loader:
    rec = ZeroFilledReconstructor(n_jobs=1)
    rec_motion2 = rec.reconstruct(data_loader).squeeze()
    motion2 = data_loader.get_dynamic(0)


# %%
with CartesianFrameDataLoader("example_motion2.mrd") as data_loader:
    sim_conf.max_sim_time

# %%
rec_motion.shape

# %%
# Show the motion
# ---------------
fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
time = np.arange(len(motion.data[0])) * sim_conf.seq.TR / 1000
TR_vol = 3  # We generated 2 frames in 6 seconds
axs[0].axvline(TR_vol, c="gray")
axs[1].axvline(TR_vol, c="gray")


axs[0].plot(time, motion.data[:3, :].T)
axs[1].plot(time, motion.data[3:, :].T)
axs[0].set_prop_cycle(None)  # reset color cycling
axs[1].set_prop_cycle(None)
axs[0].plot(time, motion2.data[:3, :].T, linestyle="dashed")
axs[1].plot(time, motion2.data[3:, :].T, linestyle="dashed")

axs[1].set_xlabel("time (s)")
axs[0].set_ylabel("Translation (mm)")
axs[1].set_ylabel("Rotation (deg)")

# %%
# Visualizing the reconstructed data
# ----------------------------------

import matplotlib.pyplot as plt

from snake.toolkit.plotting import axis3dcut

fig, axs = plt.subplots(1, 3, figsize=(26, 5))

axis3dcut(
    rec_motion[0].T,
    None,
    None,
    cbar=False,
    cuts=(0.5, 0.5, 0.5),
    ax=axs[0],
)
axis3dcut(
    rec_motion2[0].T,
    None,
    None,
    cbar=False,
    cuts=(0.5, 0.5, 0.5),
    ax=axs[1],
)
axis3dcut(rec_nomotion[0].T, None, None, cbar=False, cuts=(0.5, 0.5, 0.5), ax=axs[2])
axs[0].set_title("Small motion")
axs[1].set_title("Big motion")
axs[2].set_title("No motion")
plt.show()


# %%
sim_conf.seq.TR_eff


# %%
