"""Utilities for MRD file format."""

import atexit
import base64
import os
import pickle
from enum import IntFlag
from typing import Any

import ismrmrd as mrd
import numpy as np

from snkf._meta import LogMixin


def obj2b64encode(f: Any) -> bytes:
    """Return the base64 encoded pickle of a python object."""
    return base64.b64encode(pickle.dumps(f))


def b64encode2obj(s: str) -> Any:
    """Load a base64 string as a python object."""
    return pickle.loads(base64.b64decode(s))


# fmt: off
class ACQ(IntFlag):
    """Acquisition flags of MRD as an IntFlags."""

    FIRST_IN_ENCODE_STEP1               = 1 << 0
    LAST_IN_ENCODE_STEP1                = 1 << 1
    FIRST_IN_ENCODE_STEP2               = 1 << 2
    LAST_IN_ENCODE_STEP2                = 1 << 3
    FIRST_IN_AVERAGE                    = 1 << 4
    LAST_IN_AVERAGE                     = 1 << 5
    FIRST_IN_SLICE                      = 1 << 6
    LAST_IN_SLICE                       = 1 << 7
    FIRST_IN_CONTRAST                   = 1 << 8
    LAST_IN_CONTRAST                    = 1 << 9
    FIRST_IN_PHASE                      = 1 << 10
    LAST_IN_PHASE                       = 1 << 11
    FIRST_IN_REPETITION                 = 1 << 12
    LAST_IN_REPETITION                  = 1 << 13
    FIRST_IN_SET                        = 1 << 14
    LAST_IN_SET                         = 1 << 15
    FIRST_IN_SEGMENT                    = 1 << 16
    LAST_IN_SEGMENT                     = 1 << 17
    IS_NOISE_MEASUREMENT                = 1 << 18
    IS_PARALLEL_CALIBRATION             = 1 << 19
    IS_PARALLEL_CALIBRATION_AND_IMAGING = 1 << 20
    IS_REVERSE                          = 1 << 21
    IS_NAVIGATION_DATA                  = 1 << 22
    IS_PHASECORR_DATA                   = 1 << 23
    LAST_IN_MEASUREMENT                 = 1 << 24
    IS_HPFEEDBACK_DATA                  = 1 << 25
    IS_DUMMYSCAN_DATA                   = 1 << 26
    IS_RTFEEDBACK_DATA                  = 1 << 27
    IS_SURFACECOILCORRECTIONSCAN_DATA   = 1 << 28
    IS_PHASE_STABILIZATION_REFERENCE    = 1 << 29
    IS_PHASE_STABILIZATION              = 1 << 30
    COMPRESSION1                        = 1 << 52
    COMPRESSION2                        = 1 << 53
    COMPRESSION3                        = 1 << 54
    COMPRESSION4                        = 1 << 55
    USER1                               = 1 << 56
    USER2                               = 1 << 57
    USER3                               = 1 << 58
    USER4                               = 1 << 59
    USER5                               = 1 << 60
    USER6                               = 1 << 61
    USER7                               = 1 << 62
    USER8                               = 1 << 63
# fmt: on


class MRDLoader(LogMixin):
    """Base class for MRD data loader."""

    def __init__(self, filename: os.PathLike):
        self.filename = filename
        self.dataset = mrd.Dataset(filename, create_if_needed=False)
        self.header = mrd.xsd.CreateFromDocument(self.dataset.read_xml_header())

        matrixSize = self.header.encoding[0].encodedSpace.matrixSize
        self.shape = matrixSize.x, matrixSize.y, matrixSize.z
        atexit.register(self._cleanup)

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return self.header.encoding[0].encodingLimits.repetition.maximum

    @property
    def n_coils(self) -> int:
        """Number of coils."""
        return self.header.acquisitionSystemInformation.receiverChannels

    def __len__(self):
        return self.n_frames

    @property
    def n_sample(self) -> int:
        """Number of samples in a single acquisition."""
        return self.header.encoding[0].limits.kspace_encoding_step_0.maximum

    @property
    def n_shots(self) -> int:
        """Number of samples in a single acquisition.

        Notes
        -----
        for EPI this is the number of phase encoding lines in the EPI zigzag.
        """
        return self.header.encoding[0].limits.kspace_encoding_step_1.maximum

    def _cleanup(self) -> None:
        try:
            self.dataset.close()
        except Exception as e:
            self.log.error(e)
            pass

    def __iter__(self):
        raise NotImplementedError


class CartesianFrameDataLoader(MRDLoader):
    """Load cartesian MRD files k-space frames iteratively.

    Parameters
    ----------
    filename: source for the MRD file.

    Examples
    --------
    >>> for kspace, mask in CartesianFrameDataLoader("test.mrd"):
            image = ifft(kspace)
    """

    def __iter__(self):
        counter = 0
        yielded = False
        kspace = np.zeros((self.n_coils, *self.shape), dtype=np.complex64)
        mask = np.zeros(self.shape, dtype=bool)
        acq = self.dataset.read_acquisition(counter)
        n_acq = self.dataset.number_of_acquisitions()
        while counter < n_acq:
            traj_locs = tuple(np.int32(acq.traj.T))
            for c in range(self.n_coils):  # FIXME what is the good way of doing this ?
                kspace[c][traj_locs] = acq.data[c]

            mask[traj_locs] = True
            if (
                acq.flags & ACQ.LAST_IN_REPETITION
                or acq.flags & ACQ.LAST_IN_MEASUREMENT
            ):
                yield mask, kspace
                kspace[:] = 0
                mask[:] = False
                yielded = True
            counter += 1
            if counter < self.dataset.number_of_acquisitions():
                acq = self.dataset.read_acquisition(counter)
                if yielded:
                    yielded = False
                    if not (acq.flags & ACQ.FIRST_IN_REPETITION):
                        raise ValueError(
                            f"Flags error at {counter} {ACQ(acq.flags).__repr__()}"
                        )


class NonCartesianFrameDataLoader(MRDLoader):
    """Non Cartesian Dataloader."""

    def __iter__(self):
        counter = 0
        shot_counter = 0
        yielded = False
        kspace = np.zeros(
            (self.n_coils, self.n_shots, self.n_sample), dtype=np.complex64
        )
        acq = self.dataset.read_acquisition(counter)
        n_acq = self.dataset.number_of_acquisitions()
        samples_locs = np.zeros(
            (self.n_shots, self.n_sample, len(self.shape)), dtype=bool
        )
        while counter < n_acq:
            for c in range(self.n_coils):  # FIXME what is the good way of doing this ?
                kspace[c, shot_counter] = acq.data[c]
                samples_locs[shot_counter] = acq.traj.reshape(-1, len(self.shape))
            if (
                acq.flags & ACQ.LAST_IN_REPETITION
                or acq.flags & ACQ.LAST_IN_MEASUREMENT
            ):
                yield samples_locs, kspace
                kspace[:] = 0
                samples_locs[:] = 0
                shot_counter = 0
                yielded = True
            counter += 1
            shot_counter += 1
            if counter < self.dataset.number_of_acquisitions():
                acq = self.dataset.read_acquisition(counter)
                if yielded:
                    yielded = False
                    if not (acq.flags & ACQ.FIRST_IN_REPETITION):
                        raise ValueError(
                            f"Flags error at {counter} {ACQ(acq.flags).__repr__()}"
                        )


def parse_waveform_information(dataset: mrd.Dataset) -> dict[int, dict]:
    """Parse the waveform information from the MRD file.

    Returns a dictionary with id as key and waveform information
    (name, parameters, etc.. ) as value.

    Base64 encoded parameters are decoded.
    """
    hdr = mrd.xsd.CreateFromDocument(dataset.read_xml_header())
    waveform_info = dict()
    for wi in hdr.waveformInformation:
        infos = {"name": wi.waveformName}
        for ptype, p in wi.userParameters.__dict__.items():
            for pp in p:
                if ptype == "userParameterBase64":
                    infos[pp.name] = b64encode2obj(pp.value)
                elif ptype == "userParameterString":
                    infos[pp.name] = pp.value
                elif ptype == "userParameterLong":
                    infos[pp.name] = pp.value
                elif ptype == "userParameterDouble":
                    infos[pp.name] = pp.value
                else:
                    raise ValueError(f"Unknown parameter type {ptype}")

        waveform_info[int(wi.waveformType)] = infos

    return waveform_info
