"""Export data to mrd format."""

import logging
import os

import ismrmrd as mrd
import numpy as np
from hydra_callbacks import PerfLogger
from mrinufft.trajectories.utils import Gammas

from snkf.phantom import DynamicData, Phantom
from snkf.sampling import BaseSampler
from snkf.simulation import SimConfig
from snkf.smaps import get_smaps

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

    # FIXME: update the encoding in acquisition writer.
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


def make_base_mrd(
    filename: os.PathLike,
    sampler: BaseSampler,
    phantom: Phantom,
    sim_conf: SimConfig,
    dynamic_data: list[DynamicData] = None,
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
        sampler.add_all_acq_mrd(dataset, phantom, sim_conf)
    with PerfLogger(logger=log, name="phantom"):
        add_phantom_mrd(dataset, phantom, sim_conf)

    with PerfLogger(logger=log, name="dynamic"):
        if dynamic_data:
            for dyn in dynamic_data:
                dataset = dyn.to_mrd_dataset(dataset, sim_conf)

    with PerfLogger(logger=log, name="smaps"):
        if sim_conf.hardware.n_coils > 1:
            add_smaps_mrd(dataset, sim_conf)
    dataset.close()
    return dataset
