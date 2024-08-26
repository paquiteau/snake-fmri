"""SImulation base objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np


def _repr_html_(obj: Any, vertical: bool = True) -> str:
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
    from typing import get_type_hints
    from dataclasses import fields

    resolved_hints = get_type_hints(obj)

    field_names = [f.name for f in fields(obj)]
    field_values = {name: getattr(obj, name) for name in field_names}
    resolved_field_types = {name: resolved_hints[name] for name in field_names}

    if vertical:  # switch between vertical and horizontal mode
        for field_name in field_names:
            # Recursively call _repr_html_ for nested dataclasses
            field_value = field_values[field_name]
            field_type = resolved_field_types[field_name].__name__
            try:
                field_value_str = field_value._repr_html_(vertical=not vertical)
            except AttributeError:
                field_value_str = repr(field_value)

            table_rows.append(
                f"<tr><td>{field_name} (<i>{field_type}</i>)</td><td>{field_value_str}</td></tr>"
            )
    else:
        table_rows.append(
            "<tr>"
            + "".join(
                [
                    f"<td>{field_name} (<i>{field_type}</i>)</td>"
                    for field_name, field_type in resolved_field_types.items()
                ]
            )
            + "</tr>"
        )
        values = []
        for field_value in field_values.values():
            # Recursively call _repr_html_ for nested dataclasses
            try:
                field_value_str = field_value._repr_html_(
                    vertical=not vertical
                )  # alternates orientation
            except AttributeError:
                field_value_str = repr(field_value)
            values.append(f"<td>{field_value_str}</td>")
        table_rows.append("<tr>" + "".join(values) + "</tr>")
    table_rows.append("</table>")
    return "\n".join(table_rows)


@dataclass
class GreConfig:
    """Gradient Recall Echo Sequence parameters."""

    TR: float
    TE: float
    FA: float

    _repr_html_ = _repr_html_


@dataclass
class HardwareConfig:
    """Scanner Hardware parameters."""

    gmax: float
    smax: float
    dwell_time_ms: float
    n_coils: int
    field: float = 3.0

    _repr_html_ = _repr_html_


default_hardware = HardwareConfig(gmax=40, smax=200, dwell_time_ms=1e-3, n_coils=8)

default_gre = GreConfig(TR=50, TE=30, FA=15)


@dataclass
class SimConfig:
    """All base configuration of a simulation."""

    max_sim_time: float = 300
    seq: GreConfig = default_gre
    hardware: HardwareConfig = default_hardware
    fov_mm: tuple[float, float, float] = (192.0, 192.0, 128.0)
    shape: tuple[int, int, int] = (192, 192, 128)  # Target reconstruction shape
    rng_seed: int | None = 19290506

    _repr_html_ = _repr_html_

    def __post_init__(self) -> None:
        # To be compatible with frozen dataclass
        self.rng: np.random.Generator = np.random.default_rng(self.rng_seed)

    @property
    def max_n_shots(self) -> int:
        """Maximum number of frames."""
        return int(self.max_sim_time * 1000 / self.sim_tr_ms)

    @property
    def res_mm(self) -> tuple[float, ...]:
        """Voxel resolution in mm."""
        return tuple(self.fov_mm[i] / self.shape[i] for i in range(3))

    @property
    def sim_tr_ms(self) -> float:
        """Simulation resolution in ms."""
        return self.seq.TR
