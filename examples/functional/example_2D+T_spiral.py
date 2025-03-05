#!/usr/bin/env python
# %%
from snake.core.engine import NufftAcquisitionEngine
from snake.core.handlers import BlockActivationHandler
from snake.core.phantom import Phantom
from snake.core.sampling import StackOfSpiralSampler
from snake.core.simulation import SimConfig, default_hardware, GreConfig
from snake.core.transform import apply_affine
from snake.mrd_utils import NonCartesianFrameDataLoader
from numpy.typing import NDArray
import numpy as np
from snake.toolkit.plotting import axis3dcut
from mrinufft.trajectories.display import display_3D_trajectory, display_2D_trajectory


# For reconstruction
import cupy as cp
import matplotlib.pyplot as plt
import mrinufft
from tqdm.auto import tqdm



# %%
import logging
logging.getLogger("snake").setLevel(level=logging.DEBUG)

# %%

# +
## Configuration for 2D trajectories

sim_conf = SimConfig(
    max_sim_time=300,
    seq=GreConfig(TR=2000, TE=40, FA=10),
    hardware=default_hardware,
)


sim_conf.hardware.n_coils = 8  # Update to get multi coil results.
sim_conf.hardware.field_strength = 7
sim_conf.fov.res_mm = (1,1,1) # 1 mm isotropic resolution
sim_conf.fov.size = (181,217,1) # Single slice  (in mm)
sim_conf.fov.offset = (-90,-125, -10) # Located in the center of the cortex
sim_conf.fov.angles = (0,0,0)
# -

phantom = Phantom.from_brainweb(sub_id=4, sim_conf=sim_conf, tissue_file="tissue_7T", output_res=1)
phantom = phantom.resample(new_affine=sim_conf.fov.affine, new_shape=sim_conf.fov.shape, use_gpu=True) # already select the phantom here, to reduce computational burden.

contrast = phantom.contrast(sim_conf=sim_conf)

plt.imshow(contrast.squeeze().T, origin="lower", cmap="gray")
plt.axis("off")

activation_handler = BlockActivationHandler(
    block_off=20,
    block_on=20, 
    duration=300, 
    atlas="hardvard-oxford__cort-maxprob-thr50-1mm",
    atlas_label = 48 # occipital cortex
)

example_phantom = activation_handler.get_static(phantom.copy(), sim_conf)

roi = example_phantom.masks[example_phantom.labels_idx["ROI"]]

roi_resampled = apply_affine(roi,new_affine=sim_conf.fov.affine, old_affine=phantom.affine,new_shape=  sim_conf.fov.shape)

plt.imshow(contrast.squeeze().T, origin="lower", cmap="gray")
plt.axis("off")
plt.contour(roi_resampled.T.squeeze(), levels=(0.01, 0.5),origin="lower", colors=["tab:blue", "tab:orange"])

# +
# %%

sampler = StackOfSpiralSampler(
    accelz=1,
    acsz=0.1,
    orderz="top-down",
    spiral_name="archimedes",
    nb_revolutions=16,
    obs_time_ms=30,
    constant=True,
)

# +
traj = sampler.get_next_frame(sim_conf)
print(traj.shape)

display_3D_trajectory(traj)


# %%
# Defining Acquisition model
#--------------------------


# Here we assume an almost infinite SNR and a T2* decay signal model
# in the modeling of the fMRI signal

engine_t2s = NufftAcquisitionEngine(
    model="T2s",  
    snr=30000, 
    slice_2d=True, # Does not matter, we are in 2D anyway (: 
)

engine_t2s(
    "example_2D+T.mrd",  
    sampler,  
    phantom,  
    sim_conf, 
    handlers=[activation_handler],  
    worker_chunk_size=20,  
    n_workers=1,  
    nufft_backend="finufft", 
)




# %%

from snake.toolkit.reconstructors import ZeroFilledReconstructor, SequentialReconstructor
with NonCartesianFrameDataLoader("example_2D+T.mrd") as data_loader:
    rec_zer = ZeroFilledReconstructor(nufft_backend="finufft", density_compensation=None).reconstruct(data_loader)
    dyn_datas = data_loader.get_all_dynamic()
    phantom_dl = data_loader.get_phantom()
    sim_conf = data_loader.get_sim_conf()


# %%
plt.imshow(abs(rec_zer[0, ...]), cmap="gray", origin='lower')

# %%

from snake.toolkit.analysis.stats import contrast_zscore, get_scores

+
waveform_name = "activation-block_on"
good_d = None
for d in dyn_datas:
    print(d.name)
    if d.name == waveform_name:
        good_d = d
if good_d is None:
    raise ValueError("No dynamic data found matching waveform name")

bold_signal = good_d.data[0]
bold_sample_time = np.arange(len(bold_signal)) * sim_conf.seq.TR / 1000
del phantom
del dyn_datas

# -

TR_vol = sim_conf.seq.TR
z_score = contrast_zscore(
    rec_zer, TR_vol, bold_signal, bold_sample_time, "block-on"
)
stats_results = get_scores(z_score, roi_mask, 0.5)
np.save(data_zscore_file, z_score)


rec_zer.shape

# %%

# %%
