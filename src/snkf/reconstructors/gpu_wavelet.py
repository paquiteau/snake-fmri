"""Wavelet Transform using pytorch compatible with Modopt API."""

import torch
import ptwt
import cupy as cp
import numpy as np


class TorchWaveletTransform:
    """Wavelet transform using pytorch."""

    wavedec3_keys = ["aad", "ada", "add", "daa", "dad", "dda", "ddd"]

    def __init__(
        self,
        shape: tuple[int, ...],
        wavelet: str,
        level: int,
        mode: str,
    ):
        self.wavelet = wavelet
        self.level = level
        self.shape = shape
        self.mode = mode

    def op(self, data: torch.Tensor) -> list[torch.Tensor]:
        """Apply the wavelet decomposition on.

        Parameters
        ----------
        data: torch.Tensor
            2D or 3D, real or complex data with last axes matching shape of
            the operator.

        Returns
        -------
        list[torch.Tensor]
            list of tensor each containing the data of a subband.
        """
        if data.shape == self.shape:
            data = data[None, ...]  # add a batch dimension

        if len(self.shape) == 2:
            if torch.is_complex(data):
                # 2D Complex
                data_ = torch.view_as_real(data)
                coeffs_ = ptwt.wavedec2(
                    data_, self.wavelet, level=self.level, mode=self.mode, axes=(-3, -2)
                )
                self.coeffs_shape = [coeffs_[0].shape]
                self.coeffs_shape += [tuple(cc.shape for cc in c) for c in coeffs_]
                # flatten list of tuple of tensors to a list of tensors
                coeffs = [torch.view_as_complex(coeffs_[0].contiguous())] + [
                    torch.view_as_complex(cc.contiguous())
                    for c in coeffs_[1:]
                    for cc in c
                ]

                return coeffs
            # 2D Real
            coeffs_ = ptwt.wavedec2(
                data, self.wavelet, level=self.level, mode=self.mode, axes=(-2, -1)
            )
            return [coeffs_[0]] + [cc for c in coeffs_[1:] for cc in c]

        if torch.is_complex(data):
            # 3D Complex
            data_ = torch.view_as_real(data)
            coeffs_ = ptwt.wavedec3(
                data_,
                self.wavelet,
                level=self.level,
                mode=self.mode,
                axes=(-4, -3, -2),
            )
            # flatten list of tuple of tensors to a list of tensors
            coeffs = [torch.view_as_complex(coeffs_[0].contiguous())] + [
                torch.view_as_complex(cc.contiguous())
                for c in coeffs_[1:]
                for cc in c.values()
            ]

            return coeffs
        # 3D Real
        coeffs_ = ptwt.wavedec3(
            data, self.wavelet, level=self.level, mode=self.mode, axes=(-3, -2, -1)
        )
        return [coeffs_[0]] + [cc for c in coeffs_[1:] for cc in c.values()]

    def adj_op(self, coeffs: list[torch.Tensor]) -> torch.Tensor:
        """Apply the wavelet recomposition.

        Parameters
        ----------
        list[torch.Tensor]
            list of tensor each containing the data of a subband.

        Returns
        -------
        data: torch.Tensor
            2D or 3D, real or complex data with last axes matching shape of the
            operator.

        """
        if len(self.shape) == 2:
            if torch.is_complex(coeffs[0]):
                ## 2D Complex ##
                # list of tensor to list of tuple of tensor
                coeffs = [torch.view_as_real(coeffs[0])] + [
                    tuple(torch.view_as_real(coeffs[i + k]) for k in range(3))
                    for i in range(1, len(coeffs) - 2, 3)
                ]
                data = ptwt.waverec2(coeffs, wavelet=self.wavelet, axes=(-3, -2))
                return torch.view_as_complex(data.contiguous())
            ## 2D Real ##
            coeffs_ = [coeffs[0]] + [
                tuple(coeffs[i + k] for k in range(3))
                for i in range(1, len(coeffs) - 2, 3)
            ]
            data = ptwt.waverec2(coeffs_, wavelet=self.wavelet, axes=(-2, -1))
            return data

        if torch.is_complex(coeffs[0]):
            ## 3D Complex ##
            # list of tensor to list of tuple of tensor
            coeffs = [torch.view_as_real(coeffs[0])] + [
                {
                    v: torch.view_as_real(coeffs[i + k])
                    for k, v in enumerate(self.wavedec3_keys)
                }
                for i in range(1, len(coeffs) - 6, 7)
            ]
            data = ptwt.waverec3(coeffs, wavelet=self.wavelet, axes=(-4, -3, -2))
            return torch.view_as_complex(data.contiguous())
        ## 3D Real ##
        coeffs_ = [coeffs[0]] + [
            {v: coeffs[i + k] for k, v in enumerate(self.wavedec3_keys)}
            for i in range(1, len(coeffs) - 6, 7)
        ]
        data = ptwt.waverec3(coeffs_, wavelet=self.wavelet, axes=(-3, -2, -1))
        return data


class CupyWaveletTransform:
    """Wrapper around torch wavelet transform."""

    def __init__(
        self,
        shape: tuple[int, ...],
        wavelet: str,
        level: int,
        mode: str,
    ):
        self.wavelet = wavelet
        self.level = level
        self.shape = shape
        self.mode = mode

        self.operator = TorchWaveletTransform(shape, wavelet, level, mode)

    def op(self, data: cp.array) -> cp.ndarray:
        """Apply Forward Wavelet transform on cupy array."""
        data_ = torch.as_tensor(data)
        tensor_list = self.operator.op(data_)
        # flatten the list of tensor to a cupy array
        # this requires an on device copy...
        self.coeffs_shape = [c.shape for c in tensor_list]
        n_tot_coeffs = np.sum([np.prod(s) for s in self.coeffs_shape])
        ret = cp.zeros(n_tot_coeffs, dtype=np.complex64)  # FIXME get dtype from torch
        start = 0
        for t in tensor_list:
            stop = start + np.prod(t.shape)
            ret[start:stop] = cp.asarray(t.flatten())
            start = stop

        return ret

    def adj_op(self, data: cp.ndarray) -> cp.ndarray:
        """Apply Adjoint Wavelet transform on cupy array."""
        start = 0
        tensor_list = [None] * len(self.coeffs_shape)
        for i, s in enumerate(self.coeffs_shape):
            stop = start + np.prod(s)
            tensor_list[i] = torch.as_tensor(data[start:stop].reshape(s), device="cuda")
            start = stop
        ret_tensor = self.operator.adj_op(tensor_list)
        return cp.from_dlpack(ret_tensor)
