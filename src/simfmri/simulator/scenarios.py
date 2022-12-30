"""Example of scenarios for the simulator."""

from simfmri.utils import Shape2d3d

from .simulation import Simulation
from .handlers import (
    SheppLoganPhantomGeneratorHandler,
    ActivationHandler,
    AcquisitionHandler,
)


def sl_block_vds(n_frames: int, shape: Shape2d3d, snr: float, accel: int) -> Simulation:
    """Generate a simulation scenario using shepp-logan and block activation.

    The TR is fixed to 1s.

    Parameters
    ----------
    n_frames
        number of frames
    shape
        Shape of the simulated volume
    snr
        Signal to noise ratio to target
    accel
        Acceleration factor for the acquisition.

    Returns
    -------
    Simulation
        The simulation object associated with the data.

    See Also
    --------
    simfmri.simulator.simulation.Simulation
        The data container for the simulation.
    """
    sim_data = Simulation(shape=shape, n_frames=n_frames, TR=1, n_coils=1)

    simulator = (
        SheppLoganPhantomGeneratorHandler()
        @ ActivationHandler.from_block_design(3, 3, n_frames)
        @ AcquisitionHandler.vds(acs=24, accel=accel, constant=True, gen_smaps=False)
        @ KspaceNoiseHandler(snr=snr)
    )

    return simulator(sim_data)
