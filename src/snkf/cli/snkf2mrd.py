"""Convert snake fmri simulation file to a ISMRMRD file."""

import argparse
import numpy as np
import ismrmrd as mrd
from snkf.simulation import SimData

from mrinufft.trajectories.utils import Gammas


def snake2mrd_header(sim: SimData) -> mrd.xsd.ismrmrdHeader:
    """Get the header of the ISMRMRD file from the snake simulation."""
    header = mrd.xsd.ismrmrdHeader()
    # Experimental conditions
    exp = mrd.xsd.experimentalConditionsType()
    exp.H1resonanceFrequency_Hz = int(Gammas.H * 1e3)
    header.experimentalConditions = exp

    # Acquisition System Information
    sys = mrd.xsd.acquisitionSystemInformationType()
    sys.deviceID = "SNAKE-fMRI"
    sys.systemVendor = "SNAKE"
    sys.systemModel = "SNAKE-fMRI"
    # Current version of software.
    sys.deviceSerialNumber = 42
    sys.systemFieldStrength_T = sim.hardware.field
    # Only thing important in system header !
    sys.receiverChannels = sim.n_coils

    header.acquisitionSystemInformation = sys

    # Encoding
    encoding = mrd.xsd.encodingType()
    # TODO: Detect the trajectory type from sim.extra_infos["traj_name"]
    encoding.trajectory = mrd.xsd.trajectoryType.OTHER
    # FOV
    input_fov = mrd.xsd.fieldOfViewMm(*(np.array(sim.fov) * 1e3))
    input_matrix = mrd.xsd.matrixSizeType(*sim.shape)

    output_fov = mrd.xsd.fieldOfViewMm(*(np.array(sim.fov) * 1e3))
    output_matrix = mrd.xsd.matrixSizeType(*sim.shape)

    encoding.encodeSpace = mrd.xsd.encodingSpaceType(input_matrix, input_fov)
    encoding.reconSpace = mrd.xsd.encodingSpaceType(output_matrix, output_fov)

    # Get the trajectory and count points.
    # each limits as a min, center, max attribute.
    limits = mrd.xsd.encodingLimitsType()
    # TODO: change me for shot length ?
    limits.kspace_encoding_step_1 = mrd.xsd.limitType(
        0, sim.extra_infos["n_shot_per_frame"] - 1
    )
    limits.repetition = mrd.xsd.limitType(0, len(sim.kspace_data) - 1)
    encoding.encodingLimits = limits

    header.encoding.append(encoding)
    return header


def snake2mrd_acquisition(sim: SimData, dset: mrd.Dataset) -> mrd.Dataset:
    """Get the acquisition header of the ISMRMRD file from the snake simulation."""
    counter = 0
    for i in range(len(sim.kspace_mask)):
        kspace_traj_vol = sim.kspace_mask[i]
        kspace_data_vol = sim.kspace_data[i]
        if len(kspace_traj_vol.shape) == 2:
            n_samples, dim = kspace_traj_vol.shape
            n_shots = sim.extra_infos["n_shot_per_frame"]
        else:
            n_shots, n_samples, dim = kspace_traj_vol.shape
        kspace_traj_vol = kspace_traj_vol.reshape(n_shots, -1, dim)
        kspace_data_vol = kspace_data_vol.reshape(sim.n_coils, n_shots, -1)

        for j in range(n_shots):
            acq = mrd.Acquisition.from_array(
                data=kspace_data_vol[:, j, :], trajectory=kspace_traj_vol[j, :]
            )
            acq.scan_counter = counter
            acq.sample_time_us = 50000 / n_samples
            acq.idx.repetition = i
            acq.idx.kspace_encode_step_1 = j
            acq.idx.kspace_encode_step_2 = 1

            # Set flags: # TODO: upstream this in the acquisition handler.
            if j == 0:
                acq.setFlag(mrd.ACQ_FIRST_IN_ENCODE_STEP1)
                acq.setFlag(mrd.ACQ_FIRST_IN_REPETITION)
            if j == n_shots - 1:
                acq.setFlag(mrd.ACQ_LAST_IN_ENCODE_STEP1)
                acq.setFlag(mrd.ACQ_LAST_IN_REPETITION)

            dset.append_acquisition(acq)
            counter += 1
    return dset


def snake2mrd_extras(sim: SimData, dset: mrd.Dataset) -> mrd.Dataset:
    """Write extra stuff to the dataset."""
    image_counter = 0

    if sim.static_vol is not None:
        static_vol = mrd.Image.from_array(sim.static_vol, transpose=False)
        static_vol.image_index = image_counter
        static_vol.meta = mrd.Meta(
            ImageNumber=image_counter, ImageComments="Reference Phantom"
        )
        image_counter += 1
        dset.append_image("static_vol", static_vol)
    if sim.smaps is not None:
        smaps_image = mrd.Image.from_array(
            sim.smaps,
            transpose=False,
        )
        smaps_image.image_index = image_counter

        smaps_image.meta = mrd.Meta(ImageComments="Reference Smaps")
        image_counter += 1
        dset.append_image("smaps", smaps_image)

    return dset


def snake2mrd(sim: SimData, output_file: str) -> None:
    """
    Convert snake fmri simulation file to a ISMRMRD file.

    Parameters
    ----------
    input_file : str
        Path to the input snake file.
    output_file : str
        Path to the output ISMRMRD file.

    Returns
    -------
    None
    """
    # Load the snake file

    # Create the ISMRMRD dataset
    dset = mrd.Dataset(output_file, "datase", create_if_needed=True)

    header = snake2mrd_header(sim)
    dset.write_xml_header(mrd.xsd.ToXML(header))
    dset = snake2mrd_acquisition(sim, dset)
    dset = snake2mrd_extras(sim, dset)

    dset.close()


def main() -> None:
    """Convert Snake-fMRI data to ISMRMRD."""
    parser = argparse.ArgumentParser(
        description="Convert snake fmri simulation file to a ISMRMRD file."
    )
    parser.add_argument("input_file", type=str, help="Path to the input snake file.")
    parser.add_argument(
        "output_file", type=str, help="Path to the output ISMRMRD file."
    )
    args = parser.parse_args()

    snake2mrd(np.load(args.input_file, allow_pickle=True), args.output_file)


if __name__ == "__main__":
    main()
