"""Export data to mrd format."""

from __future__ import annotations
import logging
import os
from typing import TYPE_CHECKING
import ismrmrd as mrd
import numpy as np
from numpy.typing import NDArray
from hydra_callbacks import PerfLogger
from mrinufft.trajectories.utils import Gammas
from snake._version import __version__ as version

if TYPE_CHECKING:
    from snake.phantom import DynamicData, Phantom
    from snake.sampling import BaseSampler
    from snake.simulation import SimConfig

log = logging.getLogger(__name__)


def get_mrd_header(sim_conf: SimConfig, engine: str) -> mrd.xsd.ismrmrdHeader:
    """Create a MRD Header for snake-fmri data."""
    H = mrd.xsd.ismrmrdHeader()
    # Experimental conditions
    H.experimentalConditions = mrd.xsd.experimentalConditionsType(
        H1resonanceFrequency_Hz=int(Gammas.H * 1e3),
    )

    # Acquisition System Information
    H.acquisitionSystemInformation = mrd.xsd.acquisitionSystemInformationType(
        deviceID="SNAKE",
        systemVendor="SNAKE",
        systemModel=f"SNAKE-v{version}-{engine}",
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
    H.userParameters = mrd.xsd.userParametersType(
        userParameterDouble=[
            mrd.xsd.userParameterDoubleType(name=name, value=value)
            for name, value in [
                ("gmax", sim_conf.hardware.gmax),
                ("smax", sim_conf.hardware.smax),
                ("dwell_time_ms", sim_conf.hardware.dwell_time_ms),
                ("rng_seed", sim_conf.rng_seed),
            ]
        ]
    )

    return H


def add_phantom_mrd(
    dataset: mrd.Dataset, phantom: Phantom, sim_conf: SimConfig
) -> mrd.Dataset:
    """Add the phantom to the dataset."""
    return phantom.to_mrd_dataset(dataset, sim_conf)


def add_smaps_mrd(
    dataset: mrd.Dataset,
    sim_conf: SimConfig,
    smaps: NDArray | None = None,
) -> mrd.Dataset:
    """Add the Smaps to the dataset."""
    if smaps is None:
        return dataset
    elif smaps.shape != (sim_conf.hardware.n_coils, *sim_conf.shape):
        raise ValueError(
            "Imcompatible smaps shape"
            f"{smaps.shape} != {(sim_conf.hardware.n_coils, *sim_conf.shape)} "
        )
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


def add_coil_cov_mrd(
    dataset: mrd.Dataset,
    sim_conf: SimConfig,
    coil_cov: NDArray | None = None,
) -> mrd.Dataset:
    """Add the Smaps to the dataset."""
    n_coils = sim_conf.hardware.n_coils
    if coil_cov is None:
        return dataset
    elif coil_cov.shape != (n_coils, n_coils):
        raise ValueError(
            f"Imcompatible coil_cov shape {coil_cov.shape} != {(n_coils, n_coils)} "
        )
    dataset.append_image(
        "coil_cov",
        mrd.image.Image(
            head=mrd.image.ImageHeader(
                matrixSize=mrd.xsd.matrixSizeType(*coil_cov.shape),
                fieldOfView_mm=mrd.xsd.fieldOfViewMm(*coil_cov.shape),
                channels=1,
                acquisition_time_stamp=0,
            ),
            data=coil_cov,
        ),
    )
    return dataset


def make_base_mrd(
    filename: os.PathLike,
    sampler: BaseSampler,
    phantom: Phantom,
    sim_conf: SimConfig,
    dynamic_data: list[DynamicData] | None = None,
    smaps: NDArray | None = None,
    coil_cov: NDArray | None = None,
) -> mrd.Dataset:
    """
    Create a base `.mrd` file from the simulation configurations.

    Parameters
    ----------
    filename : os.PathLike
        The output filename.
    sampler : BaseSampler
        The sampling pattern generator.
    phantom : Phantom
        The phantom object.
    sim_conf : SimConfig
        The simulation configurations.
    dynamic_data : list[DynamicData], optional
        The dynamic data, by default None
    smaps : NDArray, optional
        The coil sensitivity maps, by default None
    coil_covar : NDArray, optional
        The coil covariance matrix, by default None
    """
    try:
        os.remove(filename)
        log.warning("Existing %s it will be overwritten", filename)
    except Exception as e:
        log.error(e)
        pass
    dataset = mrd.Dataset(filename, "dataset", create_if_needed=True)
    dataset.write_xml_header(mrd.xsd.ToXML(get_mrd_header(sim_conf, sampler.engine)))
    with PerfLogger(logger=log, name="acq"):
        sampler.add_all_acq_mrd(dataset, sim_conf)
    with PerfLogger(logger=log, name="phantom"):
        add_phantom_mrd(dataset, phantom, sim_conf)

    with PerfLogger(logger=log, name="dynamic"):
        if dynamic_data is not None:
            for dyn in dynamic_data:
                if dyn is not None:
                    dataset = dyn.to_mrd_dataset(dataset, sim_conf)

    with PerfLogger(logger=log, name="smaps"):
        if sim_conf.hardware.n_coils > 1 and smaps is not None:
            add_smaps_mrd(dataset, sim_conf, smaps)
    with PerfLogger(logger=log, name="coil_cov"):
        if sim_conf.hardware.n_coils > 1 and coil_cov is not None:
            add_coil_cov_mrd(dataset, sim_conf, coil_cov)

    dataset.close()
    return dataset
