"""Example of scenarios for the simulator."""

from simfmri.utils import Shape2d3d

from .simulation import SimulationData, SimulationParams
from .handlers import (
    AbstractHandler,
    SheppLoganGeneratorHandler,
    ActivationHandler,
    AcquisitionHandler,
    KspaceNoiseHandler,
)


class SimulationDataFactory:
    """Simulation Factory.

    The Simulation data is generated throught the `simulate` function.

    Parameters
    ----------
    sim_params
        Essential parameters for a simulation
    handlers
        list of handler to apply to get the fully generated simulation data.

    Notes
    -----
    This is best used with hydra configuration files. see `conf/simulations/`
    """

    def __init__(self, sim_params: SimulationParams, handlers: list[AbstractHandler]):
        self.sim_params = sim_params
        self.handlers = handlers

    def simulate(self, debug=True):
        """Build the simulation data."""
        sim = SimulationData.from_params(self.sim_params)

        print(sim)
        for handler in self.handlers:
            sim = handler.handle(sim)
            print(sim)
        return sim


def sl_block_vds(
    n_frames: int, shape: Shape2d3d, snr: float, accel: int
) -> SimulationData:
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
    sim_data = SimulationData(shape=shape, n_frames=n_frames, TR=1, n_coils=1)

    simulator = (
        SheppLoganGeneratorHandler()
        @ ActivationHandler.from_block_design(3, 3, n_frames)
        @ AcquisitionHandler.vds(acs=24, accel=accel, constant=True, gen_smaps=False)
        @ KspaceNoiseHandler(snr=snr)
    )

    return simulator(sim_data)
