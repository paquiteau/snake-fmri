"""
Generate a Phantom and visualize the contrast at different TE values
====================================================================

Example for generating a phantom and visualizing the contrast at different TE values.
"""

from snake.core.phantom import Phantom
from snake.core.simulation import SimConfig, GreConfig


# %%
# This notebook has interactive widgets, run it in google colab or locally to
# enjoy it fully
#

# %%
TR = 100
TE = 25
FA = 3
field = "7T"  # "1T5"

sim_conf = SimConfig(seq=GreConfig(TR=TR, TE=TE, FA=FA))
phantom = Phantom.from_brainweb(sub_id=4, sim_conf=sim_conf, output_res=1)

phantom

contrast_at_TE = phantom.contrast(resample=True, sim_conf=sim_conf)

from snake.toolkit.plotting import axis3dcut
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
axis3dcut(contrast_at_TE.T, None, None, cuts=(0.5, 0.5, 0.5), ax=ax, width_inches=5)
fig

from ipywidgets import interact


# %%
fig = plt.figure()
sim_conf = SimConfig(seq=GreConfig(TR=TR, TE=TE, FA=FA))

phantom1T5 = Phantom.from_brainweb(
    sub_id=4, sim_conf=sim_conf, tissue_file="tissue_1T5", output_res=1
)
phantom7T = Phantom.from_brainweb(
    sub_id=4, sim_conf=sim_conf, tissue_file="tissue_7T", output_res=1
)


def live_contrast(TE: float = 10, TR: float = 100, FA: float = 3, tissue_field="1T5"):
    [ax.remove() for ax in fig.get_axes()]
    ax = fig.subplots()
    sim_conf.seq.TE = TE
    sim_conf.seq.TR = TR
    sim_conf.seq.FA = FA
    if tissue_field == "1T5":
        phantom = phantom1T5
    elif tissue_field == "7T":
        phantom = phantom7T
    contrast_at_TE = phantom.contrast(sim_conf=sim_conf, resample=True)
    axis3dcut(contrast_at_TE.T, None, cuts=(0.5, 0.5, 0.5), width_inches=5, ax=ax)
    fig.canvas.draw_idle()
    # if len(fig.get_axes()) >=5:
    #     fig.get_axes()[-1].remove()


live_contrast(TE=TE, TR=TR, FA=FA, tissue_field="1T5")

# Only for notebook interactive mode
# sphx
# interact(
#     live_contrast,
#     TE=(0, 100, 1),
#     TR=(0, 1000, 1),
#     FA=(0, 90, 1),
#     tissue_field=["1T5", "7T"],
# )
