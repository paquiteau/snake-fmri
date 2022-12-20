from .base import AbstractHandler
from ..simulation import Simulation


class NoiseHandler(AbstractHandler):
    """Add noise to the data"""

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def _handle(self, sim: Simulation):
        pass


class GaussianNoiseHandler(AbstractHandler):
    def _handle(self, sim: Simulation):
        pass


class RicianNoiseHandler(AbstractHandler):
    def _handle(self, sim: Simulation):
        pass
