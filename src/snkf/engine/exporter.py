"""Export data to mrd format."""

import logging
import os
import time
import ismrmrd as mrd
import numpy as np
from mrinufft.trajectories.utils import Gammas
from hydra_callbacks import PerfLogger
from ..smaps import get_smaps
from ..phantom import Phantom, DynamicData
from ..sampling import BaseSampler
from ..simulation import SimConfig


log = logging.getLogger(__name__)


def get_mrd_header(sim_conf: SimConfig) -> mrd.xsd.ismrmrdHeader:
    """Create a MRD Header for snake-fmri data."""
    H = mrd.xsd.ismrmrdHeader()
    # Experimental conditions
    H.experimentalConditions = mrd.xsd.experimentalConditionsType(
        H1resonanceFrequency_Hz=int(Gammas.H * 1e3),
    )

    # Acquisition System Information
    H.acquisitionSystemInformation = mrd.xsd.acquisitionSystemInformationType(
        deviceID="SNAKE-fMRI",
        systemVendor="SNAKE-fMRI",
        systemModel="SNAKE-fMRI",
        deviceSerialNumber=42,
        systemFieldStrength_T=sim_conf.hardware.field,
        receiverChannels=sim_conf.hardware.n_coils,
    )

    # Encoding
    # FOV computation
    input_fov = mrd.xsd.fieldOfViewMm(*(np.array(sim_conf.fov_mm)))
    input_matrix = mrd.xsd.matrixSizeType(*sim_conf.shape)

    output_fov = mrd.xsd.fieldOfViewMm(*(np.array(sim_conf.fov_mm)))
    output_matrix = mrd.xsd.matrixSizeType(*sim_conf.shape)

    encoding = mrd.xsd.encodingType(
        encodedSpace=mrd.xsd.encodingSpaceType(input_matrix, input_fov),
        reconSpace=mrd.xsd.encodingSpaceType(output_matrix, output_fov),
        trajectory=mrd.xsd.trajectoryType.OTHER,
        encodingLimits=mrd.xsd.encodingLimitsType(
            kspace_encoding_step_0=-1,
            kspace_encoding_step_1=-1,
            kspace_encoding_step_2=-1,
            repetition=-1,
        ),
    )
    H.encoding.append(encoding)

    # Sequence Parameters
    H.sequenceParameters = mrd.xsd.sequenceParametersType(
        TR=sim_conf.seq.TR,
        TE=sim_conf.seq.TE,
        flipAngle_deg=sim_conf.seq.FA,
    )

    return H


def add_all_acq_mrd(
    dataset: mrd.Dataset,
    sampler: BaseSampler,
    phantom: Phantom,
    sim_conf: SimConfig,
) -> mrd.Dataset:
    """Generate all mrd_acquisitions."""
    single_frame = sampler._single_frame(phantom, sim_conf)
    n_shots_frame = single_frame.shape[0]
    n_samples = single_frame.shape[1]
    TR_vol_ms = sim_conf.seq.TR * single_frame.shape[0]
    n_ksp_frames_true = sim_conf.max_sim_time * 1000 / TR_vol_ms
    n_ksp_frames = int(n_ksp_frames_true)

    log.info("Generating %d frames", n_ksp_frames)
    log.info("Frame have %d shots", n_shots_frame)
    log.info("Shot have %d samples", n_samples)
    log.info("volume TR: %f ms", TR_vol_ms)

    if n_ksp_frames == 0:
        raise ValueError(
            "No frame can be generated with the current configuration"
            " (TR/shot too long or max_sim_time too short)"
        )
    if n_ksp_frames != n_ksp_frames_true:
        log.warning(
            "Volumic TR does not align with max simulation time, "
            "last incomplete frame will be discarded."
        )
    log.info("Start Sampling pattern generation")
    counter = 0
    kspace_data_vol = np.zeros(
        (n_shots_frame, sim_conf.hardware.n_coils, n_samples),
        dtype=np.complex64,
    )
    for i in range(n_ksp_frames):
        kspace_traj_vol = sampler._single_frame(sim_conf)

        for j in range(n_shots_frame):
            acq = mrd.Acquisition.from_array(
                data=kspace_data_vol[j, :, :], trajectory=kspace_traj_vol[j, :]
            )
            acq.scan_counter = counter
            acq.sample_time_us = sampler.obs_time_ms * 1000 / n_samples
            acq.center_sample = n_samples // 2 if sampler.in_out else 0
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


def add_phantom_mrd(
    dataset: mrd.Dataset, phantom: Phantom, sim_conf: SimConfig
) -> mrd.Dataset:
    """Add the phantom to the dataset."""
    return phantom.to_mrd_dataset(dataset, sim_conf)


def add_smaps_mrd(dataset: mrd.Dataset, sim_conf: SimConfig) -> mrd.Dataset:
    """Add the Smaps to the dataset."""
    smaps = get_smaps(sim_conf.shape, n_coils=sim_conf.hardware.n_coils)

    dataset.append_image(
        "smaps",
        mrd.image.Image(
            head=mrd.image.ImageHeader(
                matrixSize=mrd.xsd.matrixSizeType(*smaps.shape[1:]),
                fieldOfView_mm=mrd.xsd.fieldOfViewMm(*sim_conf.fov_mm),
                channels=len(smaps),
                acquisition_time_stamp=0,
            ),
            data=smaps,
        ),
    )
    return dataset


def add_one_wave_mrd(
    dataset: mrd.Dataset, sim_conf: SimConfig, wave_properties: DynamicData
) -> mrd.Dataset:
    """Add a single waveform to the dataset."""
    dataset.append_waveform(mrd.Waveform(head=mrd.WaveformHeader()))
    return dataset


def make_base_mrd(
    filename: os.PathLike,
    sampler: BaseSampler,
    phantom: Phantom,
    sim_conf: SimConfig,
) -> mrd.Dataset:
    """Generate a sampling pattern."""
    try:
        os.remove(filename)
        log.warning("Existing %s it will be overwritten", filename)
    except Exception as e:
        log.error(e)
        pass
    dataset = mrd.Dataset(filename, "dataset", create_if_needed=True)
    dataset.write_xml_header(mrd.xsd.ToXML(get_mrd_header(sim_conf)))
    with PerfLogger(logger=log, name="acq"):
        add_all_acq_mrd(dataset, sampler, phantom, sim_conf)
    with PerfLogger(logger=log, name="phantom"):
        add_phantom_mrd(dataset, phantom, sim_conf)
    with PerfLogger(logger=log, name="smaps"):
        if sim_conf.hardware.n_coils > 1:
            add_smaps_mrd(dataset, sim_conf)
    dataset.close()
    return dataset
