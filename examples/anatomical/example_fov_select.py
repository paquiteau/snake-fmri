# %%
"""
Select the FOV and target resolution of the phantom
===================================================
"""

from snake.core import Phantom, SimConfig, GreConfig
from snake.core.engine.utils import get_ideal_phantom
from snake.toolkit.plotting import axis3dcut

# %%
# shape = (90, 110,90)
# Original FOV And size of the phantom
TR = 50
TE = 30
FA = 40  # np.degrees(np.arccos(np.exp((-TR/2000))))
field = "7T"  # "1T5"

sim_conf = SimConfig(max_sim_time=3, seq=GreConfig(TR=TR, TE=TE, FA=FA))
sim_conf.hardware.n_coils = 32  # Update to get multi coil results.
sim_conf.hardware.field_strength = 7

# %%
sim_conf.res_mm

# %%
phantom = Phantom.from_brainweb(sub_id=4, sim_conf=sim_conf, tissue_file="tissue_1T5")

# %%
sim_conf

# %%
from snake.core.handlers.fov import FOVHandler


# the center and size target_res are all in millimeters.
fov_handler = FOVHandler(
    center=(90, 110, 100),
    size=(192, 192, 128),
    angles=(5, 0, 0),
    target_res=(2.0, 2.0, 2.0),
)

new_phantom = fov_handler.get_static(phantom, sim_conf)

# %%
new_contrast = get_ideal_phantom(new_phantom, sim_conf)

# %%
new_contrast.shape

# %%
axis3dcut(new_contrast.T, z_score=None, cuts=(0.5, 0.6, 0.5))
