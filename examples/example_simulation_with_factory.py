#!/usr/bin/env python3
"""
Using a simulation configuration file
=====================================

In this example we are going to describe how to use the simulation factory interface available in the runner package.

"""

import io  # to simulate a file access.

from hydra.utils import instantiate
from omegaconf import OmegaConf

# %%
# Describing all the step of a simulation can be tedious.
# In order to have easy reproducible simulation, the configuration can be extracted to
# an external file, using the YAML syntax.
# We then leverage hydra's instantiate method to do the heavy lifting for us by creating
# all the required building block for the simulator.
#
# For instance to create a 2D+T

sim_config = """
### Simulation for a Shepp Logan phantom with activation.
_target_: simfmri.simulator.SimulationDataFactory
checkpoints: false
sim_params:
  _target_: simfmri.simulator.SimulationParams
  n_frames: 100
  shape: [128, 128, 128]
  sim_tr: 1.0
  n_coils: 1

handlers:
  generator:
    _target_: simfmri.simulator.SheppLoganGeneratorHandler
  slicer:
    _target_: simfmri.simulator.SlicerHandler
    axis: 0
    index: 58
  activation:
    _target_: simfmri.simulator.ActivationHandler.from_block_design
    event_name: block_on
    block_on: 3
    block_off: 3
    duration: 100
    offset: 0
"""


# %%
# .. note::
#     In the handlers section, the name of each handler does not matter.
#     They will be processed in order.
#
# .. tip::
#     For more information on Hydra and its configuration be sure to check the sister
#     package ``simfmri_runner``.
#

# %%
# Launching the simulator
# -----------------------
#
# Thanks to the detailled configuration it is as simple as:

# equivalent to
# with open("your_config.yaml") as config_file:
with io.StringIO(sim_config) as config_file:
    factory = instantiate(OmegaConf.load(config_file))

sim_data = factory.simulate()

print(sim_data)
