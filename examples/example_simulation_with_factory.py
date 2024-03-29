#!/usr/bin/env python3
"""
Using a simulation configuration file
=====================================

This example present how to use a simulation config file.

"""

import io  # to simulate a file access.

from snkf import load_from_yaml

# %%
# Describing all the step of a simulation can be tedious.
# In order to have easy reproducible simulation, the configuration can be extracted to
# an external file, using the YAML syntax.
# We then leverage hydra's instantiate method to do the heavy lifting for us by creating
# all the required building block for the.simulation.
#
# For instance to create a 2D+T

sim_config = """
### Simulation for a Shepp Logan phantom with activation.

defaults: #adding default is best practice to get full type checking.
 - handlers: [phantom-shepp_logan, phantom-slicer, activation-block]

sim_params:
    sim_time: 100
    shape: [128, 128, 128]
    fov: [0.192,0.192,0.192]
    sim_tr: 1.0
    n_coils: 1

handlers:
    phantom-shepp_logan: {}
    phantom-slicer:
        axis: 0
        index: 58
    activation-block:
        event_name: block_on
        block_on: 3
        block_off: 3
        duration: 100
        offset: 0
 """


# %%
# .. note::
#     In the handlers section, the name of each handler matter!
#     it is the key for finding the related handler object.
#     They will be processed in order.

# %%
# Launching the simulation
# -----------------------
#
# Thanks to the detailled configuration it is as simple as:

# equivalent to
# with open("your_config.yaml") as config_file:
with io.StringIO(sim_config) as config_file:

    simulator, sim = load_from_yaml(config_file)

sim = simulator(sim)
print(sim)
