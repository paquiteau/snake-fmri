"""Samplers generate kspace trajectories."""

import ismrmrd as mrd
import numpy as np
from numpy.typing import NDArray

from ..simulation import SimConfig
from .base import BaseSampler
from .factories import (
    AngleRotation,
    VDSorder,
    VDSpdf,
    stack_spiral_factory,
    stacked_epi_factory,
)


class NonCartesianAcquisitionSampler(BaseSampler):
    """Base class for non-cartesian acquisition samplers."""

    def add_all_acq_mrd(
        self,
        dataset: mrd.Dataset,
        sim_conf: SimConfig,
    ) -> mrd.Dataset:
        """Generate all mrd_acquisitions."""
        single_frame = self._single_frame(sim_conf)
        n_shots_frame = single_frame.shape[0]
        n_samples = single_frame.shape[1]
        TR_vol_ms = sim_conf.seq.TR * single_frame.shape[0]
        n_ksp_frames_true = sim_conf.max_sim_time * 1000 / TR_vol_ms
        n_ksp_frames = int(n_ksp_frames_true)

        self.log.info("Generating %d frames", n_ksp_frames)
        self.log.info("Frame have %d shots", n_shots_frame)
        self.log.info("Shot have %d samples", n_samples)
        self.log.info("volume TR: %f ms", TR_vol_ms)

        if n_ksp_frames == 0:
            raise ValueError(
                "No frame can be generated with the current configuration"
                " (TR/shot too long or max_sim_time too short)"
            )
        if n_ksp_frames != n_ksp_frames_true:
            self.log.warning(
                "Volumic TR does not align with max simulation time, "
                "last incomplete frame will be discarded."
            )
        self.log.info("Start Sampling pattern generation")
        counter = 0
        kspace_data_vol = np.zeros(
            (n_shots_frame, sim_conf.hardware.n_coils, n_samples),
            dtype=np.complex64,
        )

        hdr = mrd.xsd.CreateFromDocument(dataset.read_xml_header())
        hdr.encoding[0].encodingLimits = mrd.xsd.encodingLimitsType(
            kspace_encoding_step_0=mrd.xsd.limitType(0, n_samples, n_samples // 2),
            kspace_encoding_step_1=mrd.xsd.limitType(
                0, n_shots_frame, n_shots_frame // 2
            ),
            repetition=mrd.xsd.limitType(0, n_ksp_frames, 0),
        )

        for i in range(n_ksp_frames):
            kspace_traj_vol = self._single_frame(sim_conf)

            for j in range(n_shots_frame):
                acq = mrd.Acquisition.from_array(
                    data=kspace_data_vol[j, :, :], trajectory=kspace_traj_vol[j, :]
                )
                acq.scan_counter = counter
                acq.sample_time_us = self.obs_time_ms * 1000 / n_samples
                acq.center_sample = n_samples // 2 if self.in_out else 0
                acq.idx.repetition = i
                acq.idx.kspace_encode_step_1 = j
                acq.idx.kspace_encode_step_2 = 1

                # Set flags: # TODO: upstream this in the acquisition handler.
                if j == 0:
                    acq.setFlag(mrd.ACQ_FIRST_IN_ENCODE_STEP1)
                    acq.setFlag(mrd.ACQ_FIRST_IN_REPETITION)
                if j == n_shots_frame - 1:
                    acq.setFlag(mrd.ACQ_LAST_IN_ENCODE_STEP1)
                    acq.setFlag(mrd.ACQ_LAST_IN_REPETITION)

                dataset.append_acquisition(acq)
                counter += 1
        return dataset


class StackOfSpiralSampler(NonCartesianAcquisitionSampler):
    """
    Spiral 2D Acquisition Handler to generate k-space data.

    Parameters
    ----------
    acsz: float | int
        Number/ proportion of lines to be acquired in the center of k-space.
    accelz: int
        Acceleration factor for the rest of the lines.
    directionz: Literal["center-out", "random"]
        Direction of the acquisition. Either "center-out" or "random".
    pdfz: Literal["gaussian", "uniform"]
        Probability density function of the sampling. Either "gaussian" or "uniform".
    obs_ms: int
        Time spent to acquire a single shot
    nb_revolutions: int
        Number of revolutions of the spiral.
    in_out: bool
        If true, the spiral is acquired with a double join pattern from/to the periphery
    **kwargs:
        Extra arguments (smaps, n_jobs, backend etc...)
    """

    __sampler_name__ = "stack-of-spiral"

    acsz: float | int
    accelz: int
    orderz: VDSorder = VDSorder.TOP_DOWN
    nb_revolutions: int = 10
    spiral_name: str = "archimedes"
    pdfz: VDSpdf = VDSpdf.GAUSSIAN
    constant: bool = False
    rotate_angle: AngleRotation = AngleRotation.ZERO
    obs_time_ms: int = 30
    n_shot_slices: int = 1

    def _single_frame(self, sim_conf: SimConfig) -> NDArray:
        """Generate the sampling pattern."""
        n_samples = int(self.obs_time_ms / sim_conf.hardware.dwell_time_ms)
        trajectory = stack_spiral_factory(
            shape=sim_conf.shape,
            accelz=self.accelz,
            acsz=self.acsz,
            n_samples=n_samples,
            nb_revolutions=self.nb_revolutions,
            pdfz=self.pdfz,
            orderz=self.orderz,
            spiral=self.spiral_name,
            rotate_angle=self.rotate_angle,
            in_out=self.in_out,
            n_shot_slices=self.n_shot_slices,
            rng=sim_conf.rng,
        )
        self._n_shot_frames = trajectory.shape[0]
        self._n_samples_shot = trajectory.shape[1]
        return trajectory


class EPI3dAcquisitionSampler(BaseSampler):
    """Sampling pattern for EPI-3D."""

    __sampler_name__ = "epi-3d"
    is_cartesian = True
    in_out = True

    acsz: float | int
    accelz: int
    orderz: VDSorder = VDSorder.CENTER_OUT

    def _single_frame(self, sim_conf: SimConfig) -> NDArray:
        """Generate the sampling pattern."""
        epi_coords = stacked_epi_factory(
            shape=sim_conf.shape,
            accelz=self.accelz,
            acsz=self.acsz,
            orderz=self.orderz,
            rng=sim_conf.rng,
        )
        return epi_coords

    def add_all_acq_mrd(
        self,
        dataset: mrd.Dataset,
        sim_conf: SimConfig,
    ) -> mrd.Dataset:
        """Create the acquisitions associated with this sampler."""
        single_frame = self._single_frame(sim_conf)
        n_shots_frame = single_frame.shape[0]
        n_samples = single_frame.shape[1]

        TR_vol_ms = sim_conf.seq.TR * single_frame.shape[0]
        n_ksp_frames_true = sim_conf.max_sim_time * 1000 / TR_vol_ms
        n_ksp_frames = int(n_ksp_frames_true)

        self.log.info("Generating %d frames", n_ksp_frames)
        self.log.info("Frame have %d shots", n_shots_frame)
        self.log.info("Shot have %d samples", n_samples)
        self.log.info("volume TR: %f ms", TR_vol_ms)

        if n_ksp_frames == 0:
            raise ValueError(
                "No frame can be generated with the current configuration"
                " (TR/shot too long or max_sim_time too short)"
            )
        if n_ksp_frames != n_ksp_frames_true:
            self.log.warning(
                "Volumic TR does not align with max simulation time, "
                "last incomplete frame will be discarded."
            )
        self.log.info("Start Sampling pattern generation")
        counter = 0
        zero_data = np.zeros(
            (sim_conf.hardware.n_coils, sim_conf.shape[2]), dtype=np.complex64
        )

        # Update the encoding limits.
        # step 0 : frequency (readout directionz)
        # step 1 : phase encoding (blip epi)
        #
        hdr = mrd.xsd.CreateFromDocument(dataset.read_xml_header())
        hdr.encoding[0].encodingLimits = mrd.xsd.encodingLimitsType(
            kspace_encoding_step_0=mrd.xsd.limitType(
                0, sim_conf.shape[2], sim_conf.shape[2] // 2
            ),
            kspace_encoding_step_1=mrd.xsd.limitType(
                0, sim_conf.shape[1], sim_conf.shape[1] // 2
            ),
            slice=mrd.xsd.limitType(0, sim_conf.shape[2], sim_conf.shape[2] // 2),
            repetition=mrd.xsd.limitType(0, n_ksp_frames, 0),
        )

        dataset.write_xml_header(mrd.xsd.ToXML(hdr))  # write the updated header back
        for i in range(n_ksp_frames):
            stack_epi3d = self._single_frame(sim_conf)  # of shape N_stack, N, 3
            for j, epi2d in enumerate(stack_epi3d):
                epi2d_r = epi2d.reshape(
                    sim_conf.shape[1], sim_conf.shape[2], 3
                )  # reorder to have
                for k, readout in enumerate(epi2d_r):
                    acq = mrd.Acquisition.from_array(data=zero_data, trajectory=readout)
                    acq.scan_counter = counter
                    acq.sample_time_us = self.obs_time_ms * 1000 / len(readout)
                    acq.idx.kspace_encode_step_1 = readout[0, 1]
                    acq.idx.slice = readout[0, 0]
                    acq.idx.repetition = i
                    val = dir_cos(readout[0], readout[1])
                    acq.read_dir = val
                    counter += 1
                    if k == 0:
                        acq.setFlag(mrd.ACQ_FIRST_IN_ENCODE_STEP1)
                        acq.setFlag(mrd.ACQ_FIRST_IN_SLICE)
                        if j == 0:
                            acq.setFlag(mrd.ACQ_FIRST_IN_REPETITION)
                    if k == len(epi2d_r) - 1:
                        acq.setFlag(mrd.ACQ_LAST_IN_ENCODE_STEP1)
                        acq.setFlag(mrd.ACQ_LAST_IN_SLICE)
                        if j == len(stack_epi3d) - 1:
                            acq.setFlag(mrd.ACQ_LAST_IN_REPETITION)
                            if i == n_ksp_frames - 1:
                                acq.setFlag(mrd.ACQ_LAST_IN_MEASUREMENT)
                    dataset.append_acquisition(acq)

        return dataset


def dir_cos(start: NDArray, end: NDArray) -> tuple[np.float32]:
    """Compute the directional cosine of the vector from beg to end point."""
    diff = np.float32(end) - np.float32(start)
    cos = diff / np.sqrt(np.sum(diff**2))
    return tuple(cos)
