"""Dynamic data object."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import ismrmrd as mrd
import numpy as np
from numpy.typing import NDArray

from snkf.mrd_utils import obj2b64encode, parse_waveform_information

from ..simulation import SimConfig
from .static import Phantom


@dataclass
class DynamicData:
    """Dynamic data object."""

    name: str
    data: NDArray
    func: Callable[[NDArray, Phantom], Phantom]
    in_kspace: bool = False

    def apply(self, phantom: Phantom, sim_conf: SimConfig) -> Phantom:
        """Apply the dynamic data to the phantom."""
        return self.func(self.data, phantom)

    def to_mrd_dataset(self, dataset: mrd.Dataset, sim_conf: SimConfig) -> mrd.Dataset:
        """Add the dynamic data to the mrd dataset.

        The data is added as a waveform with a unique id.
        The id is computed from the name of dynamic data and registerd in the header.

        In a lack of a better place, the function is added in the waveform type as
        a base64 encoded string.

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
                    ],
                    userParameterString=[
                        mrd.xsd.userParameterStringType(
                            "domain", "kspace" if self.in_kspace else "image"
                        )
                    ],
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
        print(self.name, self.data.shape)
        dataset.append_waveform(
            mrd.Waveform(
                mrd.WaveformHeader(
                    waveform_id=waveform_id,
                    number_of_samples=nsamples,
                    channels=channels,
                    sample_time_us=sim_conf.sim_tr_ms * 1000,
                ),
                data=np.float32(self.data).view(np.uint32),
            )
        )

        return dataset

    @classmethod
    def from_mrd_dataset(cls, dataset: mrd.Dataset, waveform_num: int) -> DynamicData:
        """Create a DynamicData object by reading the waveform from the dataset."""
        all_waveform_infos = parse_waveform_information(dataset)
        waveform = dataset.read_waveform(waveform_num)
        wave_info = all_waveform_infos[waveform.waveform_id]
        return cls._from_waveform(waveform, wave_info)

    @classmethod
    def _from_waveform(cls, waveform: mrd.Waveform, wave_info: dict) -> DynamicData:
        return DynamicData(
            name=wave_info["name"],
            data=waveform.data.view(np.float32).reshape(
                waveform.channels, waveform.number_of_samples
            ),
            func=wave_info[wave_info["name"]],
            in_kspace=wave_info["domain"] == "kspace",
        )

    @classmethod
    def all_from_mrd_dataset(cls, dataset: mrd.Dataset) -> list[DynamicData]:
        """Read the dataset once , and get all dynamic datas."""
        all_waveform_infos = parse_waveform_information(dataset)
        all_dyn_data = []
        for i in range(dataset.number_of_waveforms()):
            waveform = dataset.read_waveform(i)
            wave_info = all_waveform_infos[waveform.waveform_id]
            all_dyn_data.append(cls._from_waveform(waveform, wave_info))
        return all_dyn_data


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
