# %%
"""
Select the FOV and target resolution of the phantom
===================================================
"""

from snake.core import Phantom, SimConfig, GreConfig
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
sim_conf.fov.res_mm = (3, 3, 3)

# %%
phantom = Phantom.from_brainweb(
    sub_id=4, sim_conf=sim_conf, tissue_file="tissue_1T5", output_res=1
)

# %%
contrast_resampled = phantom.contrast(sim_conf=sim_conf, resample=True, aggregate=True)
contrast = phantom.contrast(sim_conf=sim_conf, resample=False, aggregate=True)

# %%
axis3dcut(contrast.T, z_score=None, cuts=(0.5, 0.6, 0.5))

# %%
sim_conf.fov.res_mm = (3, 3, 3)
sim_conf.fov.size = (192, 192, 128)
sim_conf.fov.offset = (-90, -110, -20)
sim_conf.fov.angles = (-5, 0, 0)
contrast_resampled = phantom.contrast(sim_conf=sim_conf, resample=True, aggregate=True)


# %%
axis3dcut(contrast_resampled.T, z_score=None, cuts=(0.5, 0.6, 0.5))

# %%
# Any customization
import matplotlib.pyplot as plt
from snake.core.transform import apply_affine
from snake.core.simulation import FOVConfig
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from ipywidgets import HBox, VBox

contrast = phantom.contrast(sim_conf=sim_conf, resample=False, aggregate=True)
affine_orig = phantom.affine
# Function to resample and display the image


def resample_and_display(vx, vy, vz, sx, sy, sz, ox, oy, oz, rx, ry, rz):
    # Modify the affine matrix based on the original one
    # validation
    sx = max(sx, vx)
    sy = max(sy, vy)
    sz = max(sz, vz)
    fov_conf = FOVConfig(
        res_mm=(vx, vy, vz), angles=(rx, ry, rz), offset=(ox, oy, oz), size=(sx, sy, sz)
    )

    new_contrast = apply_affine(
        contrast,
        new_affine=fov_conf.affine,
        old_affine=affine_orig,
        new_shape=fov_conf.shape,
        use_gpu=True,
    ).squeeze()
    # cleanup previous axes
    fig, ax = plt.subplots()

    if new_contrast.ndim == 2:
        ax.imshow(abs(new_contrast.T), origin="lower", cmap="gray")
        ax.axis("off")
    else:
        axis3dcut(new_contrast.T, z_score=None, cuts=(0.5, 0.5, 0.5), ax=ax)
    fig.canvas.draw_idle()


# Interactive widgets
voxel_sliders = VBox(
    [
        widgets.FloatSlider(min=1, max=5, step=0.5, value=3, description="Voxel X"),
        widgets.FloatSlider(min=1, max=5, step=0.5, value=3, description="Voxel Y"),
        widgets.FloatSlider(min=1, max=5, step=0.5, value=3, description="Voxel Z"),
    ]
)

shape_sliders = VBox(
    [
        widgets.IntSlider(min=1, max=250, step=1, value=192, description="Shape X"),
        widgets.IntSlider(min=1, max=250, step=1, value=192, description="Shape Y"),
        widgets.IntSlider(min=1, max=250, step=1, value=128, description="Shape Z"),
    ]
)

offset_sliders = VBox(
    [
        widgets.FloatSlider(min=-200, max=0, step=1, value=-90, description="Offset X"),
        widgets.FloatSlider(
            min=-200, max=0, step=1, value=-110, description="Offset Y"
        ),
        widgets.FloatSlider(min=-200, max=0, step=1, value=-20, description="Offset Z"),
    ]
)

rotation_sliders = VBox(
    [
        widgets.FloatSlider(
            min=-180, max=180, step=5, value=0, description="Rotation X"
        ),
        widgets.FloatSlider(
            min=-180, max=180, step=5, value=0, description="Rotation Y"
        ),
        widgets.FloatSlider(
            min=-180, max=180, step=5, value=0, description="Rotation Z"
        ),
    ]
)

ui = HBox([voxel_sliders, shape_sliders, offset_sliders, rotation_sliders])

interactive_widget = widgets.interactive_output(
    resample_and_display,
    dict(
        vx=voxel_sliders.children[0],
        vy=voxel_sliders.children[1],
        vz=voxel_sliders.children[2],
        sx=shape_sliders.children[0],
        sy=shape_sliders.children[1],
        sz=shape_sliders.children[2],
        ox=offset_sliders.children[0],
        oy=offset_sliders.children[1],
        oz=offset_sliders.children[2],
        rx=rotation_sliders.children[0],
        ry=rotation_sliders.children[1],
        rz=rotation_sliders.children[2],
    ),
)
display(ui, interactive_widget)

# %%
