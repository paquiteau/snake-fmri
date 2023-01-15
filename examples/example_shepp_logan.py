"""Example of simulation generation."""


from simfmri.simulator import SimulationData


#%%
# We are going to simulate a 2D+t fMRI scan of a phantom with activations. .
shape = (64, 64, 64)
n_frames = 50
TR = 1.0
n_coils = 1

#%%
# Moreover, for the acquisition we are going to specify the acceleration factor
# and the signal to noise ratio

accel = 2
snr = 100

#%%
# This data is used to createa the main SimulationData object, which gather
# all the data related to this simulation

sim_data = SimulationData(shape=shape, n_frames=n_frames, TR=1, n_coils=1)

print(sim_data)

#%%
#
# Then we are going to build the simulator from elementary steps,
# which all *handles* a particular aspect of the simulation.

from simfmri.simulator import (
    SheppLoganGeneratorHandler,
    ActivationHandler,
    AcquisitionHandler,
    KspaceNoiseHandler,
)

#%%
# The handlers can be chained easily by using the `@` operator (or by setting the
# ``next`` attribute of an Handler)
# Some handlers comes with preset function to ease their creation.
#

simulator = (
    # Create a shepp logan phantom
    SheppLoganGeneratorHandler()
    # Add activations (and creates a time dimension)
    @ ActivationHandler.from_block_design(3, 3, n_frames)
    # simulate the acquisition
    @ AcquisitionHandler.vds(acs=24, accel=accel, constant=True, gen_smaps=False)
    # add noise to the kspace
    @ KspaceNoiseHandler(snr=snr)
)

#%% The simulation can then be run easily:

sim_data = simulator(sim_data)

#%%
# Now lets see the effect of the simulation

print(sim_data)
