"""Dynamic data object."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import ismrmrd as mrd
from numpy.typing import NDArray

from snkf.mrd_utils import b64encode2obj, obj2b64encode

from ..simulation import SimConfig
from .static import Phantom


@dataclass
class DynamicData:
    """Dynamic data object."""

    name: str
    data: NDArray
    func: Callable[[NDArray, Phantom], Phantom]

    def apply(self, phantom: Phantom, sim_conf: SimConfig) -> Phantom:
        """Apply the dynamic data to the phantom."""
        return self.func(self.data, phantom)

    def to_mrd_dataset(self, dataset: mrd.Dataset, sim_conf: SimConfig) -> mrd.Dataset:
        """Add the dynamic data to the mrd dataset.

        The data is added as a waveform with a unique id.
        The id is computed from the name of the dynamic data. and registerd in the header.

        In a lack of a better place, the function is added in the waveform type as a base64 encoded string.

        Parameters
        ----------
        dataset : mrd.Dataset
            The dataset to which to add the dynamic data.
        sim_conf : SimConfig
            The simulation configuration.

        Returns
        -------
        mrd.Dataset
            The dataset with the dynamic data added.

        """

        waveform_id = get_waveform_id(self.name)

        # add the type to the header.
        hdr = mrd.xsd.CreateFromDocument(dataset.read_xml_header())
        hdr.waveformInformation.append(
            mrd.xsd.waveformInformationType(
                waveformName=self.name,
                waveformType=waveform_id,
                userParameters=mrd.xsd.userParametersType(
                    userParameterBase64=[
                        mrd.xsd.userParameterBase64Type(
                            self.name, obj2b64encode(self.func)
                        )
                    ]
                ),
            )
        )
        dataset.write_xml_header(mrd.xsd.ToXML(hdr))

        if self.data.ndim == 1:
            channels = 1
            nsamples = self.data.shape[0]
        elif self.data.ndim == 2:
            channels, nsamples = self.data.shape
        else:
            raise ValueError(f"Invalid data shape: {self.data.shape}")
        dataset.append_waveform(
            mrd.Waveform(
                mrd.WaveformHeader(
                    waveform_id=waveform_id,
                    number_of_samples=nsamples,
                    channels=channels,
                    sample_time_us=sim_conf.sim_tr_ms * 1000,
                ),
                data=self.data,
            )
        )

        return dataset

    @classmethod
    def from_mrd_dataset(cls, dataset: mrd.Dataset, waveform_num: int) -> DynamicData:
        """Create a DynamicData object by reading the waveform from the dataset."""
        hdr = mrd.CreateFromDocument(dataset.read_xml_header())
        waveform = dataset.read_waveform(waveform_num)

        waveform_id = waveform.waveform_id


def get_waveform_id(input_string: str) -> int:
    """
    Generate a unique id from an input string.

    The generated id is guaranteed to be bigger than 1024 and smaller than 2^16.

    Parameters
    ----------
    input_string : str
        The input string from which to generate the unique id.

    Returns
    -------
    int
       Unique id generated from the input string.

    Examples
    --------
    >>> generate_unique_id('hello_world')
    32767
    >>> generate_unique_id('this_is_a_test')
    20481
    """

    # Convert the input string to a hash digest
    hash_object = hashlib.md5(input_string.encode())
    hash_digest = int(hash_object.hexdigest(), 16)

    # Ensure the id is bigger than 1024 and smaller than 2^16
    unique_id = hash_digest % (2**16 - 1024) + 1024

    return unique_id
