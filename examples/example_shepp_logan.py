"""
Generate a 3d+t Simple Phantom
=================================

In this example we simulate a 3D phantom and add a simple activation pattern.
"""

import matplotlib.pyplot as plt

from snkf.simulation import SimData
from snkf.handlers import H

# %%
# We are going to simulate a 2D+t fMRI scan of a phantom with activations. .
shape = (64, 64, 64)
sim_tr = 1.0
n_coils = 1

# %%
# Moreover, for the acquisition we are going to specify the acceleration factor
# and the signal to noise ratio

accel = 2
snr = 100

# %%
# This data is used to createa the main SimData object, which gather
# all the data related to this simulation

sim_data = SimData.from_params(
    shape=shape, sim_time=50, sim_tr=1, n_coils=1, fov=(0.192, 0.192, 0.192)
)

print(sim_data)

# %%
#
# Then we are going to build the simulator from elementary steps,
# which all *handles* a particular aspect of the simulation.
#
# The handlers can be chained easily by using the `>>` operator (or by setting the
# ``next`` attribute of an Handler)
# Some handlers comes with preset function to ease their creation.
#

simulator = (
    # Create a shepp logan
    H["phantom-shepp_logan"]()
    # Add activations (and creates a time dimension)
    >> H["activation-block"](block_on=3, block_off=3, duration=50)
    # simulate the acquisition
    >> H["acquisition-vds"](
        acs=24,
        accel=accel,
        accel_axis=1,
        order="center-out",
        shot_time_ms=50,
        constant=True,
        smaps=False,
    )
    # add noise to the kspace
    >> H["noise-kspace"](snr=snr)
)

print(simulator)
# %% The simulation can then be run easily:


def print_callback(old_sim, new_sim):
    print(old_sim)
    print("->")
    print(new_sim)


simulator.add_callback_to_all(print_callback)
sim_data = simulator(sim_data)
print(sim_data)

plt.imshow(abs(sim_data.data_ref[0][32]))
plt.axis("off")
plt.show()
