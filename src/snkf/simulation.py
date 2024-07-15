"""SImulation base objects."""

from __future__ import annotations

from dataclasses import dataclass, field, InitVar

import numpy as np


def _repr_html_(obj, vertical=True) -> str:
    """
    Recursive HTML representation for dataclasses.

    This function generates an HTML table representation of a dataclass,
    including nested dataclasses.

    Parameters
    ----------
    obj: The dataclass instance.

    Returns
    -------
        str: An HTML table string representing the dataclass.
    """

    class_name = obj.__class__.__name__
    table_rows = [
        '<table style="border:1px solid lightgray;">'
        '<caption style="border:1px solid lightgray;">'
        f"<strong>{class_name}</strong></caption>"
    ]
    if vertical:
        for field_name, field_value in obj.__dict__.items():
            # Recursively call _repr_html_ for nested dataclasses
            try:
                field_value_str = field_value._repr_html_(vertical=not vertical)
            except AttributeError:
                field_value_str = repr(field_value)

            table_rows.append(
                f"<tr><td>{field_name}</td><td>{field_value_str}</td></tr>"
            )
    else:
        table_rows.append(
            "<tr>"
            + "".join([f"<td>{field_name}</td>" for field_name in obj.__dict__.keys()])
            + "</tr>"
        )
        values = []
        for field_value in obj.__dict__.values():
            # Recursively call _repr_html_ for nested dataclasses
            try:
                field_value_str = field_value._repr_html_(vertical=not vertical)
            except AttributeError:
                field_value_str = repr(field_value)
            values.append(f"<td>{field_value_str}</td>")
        table_rows.append("<tr>" + "".join(values) + "</tr>")
    table_rows.append("</table>")
    return "\n".join(table_rows)


@dataclass(frozen=True)
class GreConfig:
    """Gradient Recall Echo Sequence parameters."""

    TR: float
    TE: float
    FA: float

    _repr_html_ = _repr_html_


@dataclass(frozen=True)
class HardwareConfig:
    """Scanner Hardware parameters."""

    gmax: float
    smax: float
    dwell_time_ms: float
    n_coils: int
    field: float = 3.0

    _repr_html_ = _repr_html_


default_hardware = HardwareConfig(gmax=40, smax=200, dwell_time_ms=1e-3, n_coils=8)


@dataclass(frozen=True)
class SimConfig:
    """All base configuration of a simulation."""

    max_sim_time: float
    sim_tr_ms: float
    seq: GreConfig
    hardware: HardwareConfig = default_hardware
    fov_mm: tuple[float, float, float] = (192.0, 192.0, 128.0)
    shape: tuple[int, int, int] = (192, 192, 128)  # Target reconstruction shape
    has_relaxation: bool = True
    rng_seed: InitVar[int | None] = None
    rng: np.random.Generator = field(init=False)
    tmp_dir: str = "/tmp"

    _repr_html_ = _repr_html_

    def __post_init__(self, rng_seed: int | None):
        # To be compatible with frozen dataclass
        super().__setattr__("rng", np.random.default_rng(rng_seed))

    @property
    def max_n_frames(self) -> int:
        return int(self.max_sim_time * 1000 / self.sim_tr_ms)

    @property
    def res_mm(self) -> tuple[float, float, float]:
        return tuple(f / s for f, s in zip(self.fov_mm, self.shape, strict=True))
