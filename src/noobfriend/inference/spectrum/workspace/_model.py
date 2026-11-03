"""The mean model, written once for NumPy and JAX.

``physical`` turns values of the free (and pooled) parameters into every
physical quantity; ``frame_model`` evaluates continuum and lines on one
frame's wavelengths with that frame's LSF. Values may carry a leading draw
axis (shape ``(S, 1)``) to evaluate many draws at once.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from noobfriend.inference.spectrum.profiles import line_profile
from noobfriend.inference.spectrum.workspace._handles import ContinuumHandle, LineHandle
from noobfriend.inference.spectrum.workspace._table import ParameterTable

type ModelView = Literal["observed", "emergent", "intrinsic"]
_VIEWS = ("observed", "emergent", "intrinsic")


@dataclass(frozen=True, slots=True)
class FrameModel:
    """One frame's mean model.

    Attributes
    ----------
    continuum
        Continuum flux density.
    components
        Each line's signed contribution, in the requested view.
    total
        Continuum plus every line as observed (kernels and LSF applied),
        whatever the view.
    """

    continuum: Any
    components: Mapping[str, Any]
    total: Any


def physical(table: ParameterTable, values: Mapping[str, Any]) -> dict[str, Any]:
    """Return every parameter and line quantity by flat name.

    Parameters
    ----------
    table
        The workspace's parameter table.
    values
        Values of all free and pooled rows by flat name.

    Returns
    -------
    dict
        Fixed rows as floats, free/pooled rows as given, and every line
        quantity evaluated from them.
    """
    out: dict[str, Any] = {}
    for row in table.values():
        out[row.name] = row.value if row.kind == "fixed" else values[row.name]
    for quantity in table.quantities.values():
        out[quantity.name] = quantity.evaluator.evaluate(out)
    return out


def frame_model(
    xp: Any,
    lines: tuple[LineHandle, ...],
    continuum: ContinuumHandle,
    values: Mapping[str, Any],
    frame_id: str,
    wavelength: Any,
    *,
    view: ModelView = "observed",
) -> FrameModel:
    """Evaluate the mean model of one frame.

    Parameters
    ----------
    xp
        Array namespace.
    lines, continuum
        The workspace's handles.
    values
        Output of :func:`physical`.
    frame_id
        The frame (selects LSF widths and continuum parameters).
    wavelength
        Observed wavelengths.
    view
        Which line components to return: ``"observed"`` (kernels and LSF),
        ``"emergent"`` (kernels only) or ``"intrinsic"`` (bare profile).

    Returns
    -------
    FrameModel
        Continuum, components and the observed total.
    """
    if view not in _VIEWS:
        raise ValueError(f"view must be one of {_VIEWS}; got {view!r}.")
    coefficients = [values[name] for name in continuum.frame_parameters[frame_id]]
    base = continuum.declaration.evaluate(xp, wavelength, coefficients)
    total = base
    components = {}
    for line in lines:
        observed = _line(xp, line, values, frame_id, wavelength, kernels=True, lsf=True)
        total = total + observed
        if view == "observed":
            components[line.name] = observed
        else:
            components[line.name] = _line(
                xp,
                line,
                values,
                frame_id,
                wavelength,
                kernels=view == "emergent",
                lsf=False,
            )
    return FrameModel(continuum=base, components=components, total=total)


def _line(
    xp: Any,
    line: LineHandle,
    values: Mapping[str, Any],
    frame_id: str,
    wavelength: Any,
    *,
    kernels: bool,
    lsf: bool,
) -> Any:
    name = line.name
    kernel_values = (
        tuple(
            (kind, values[f"{name}.{kind}.fwhm"], values[f"{name}.{kind}.fraction"])
            for kind in line.kernels
        )
        if kernels
        else ()
    )
    shape = line_profile(
        xp,
        wavelength,
        profile=line.profile,
        center=values[f"{name}.center"],
        fwhm_kms=values[f"{name}.fwhm"],
        kernels=kernel_values,
        lsf_fwhm_kms=line.lsf_fwhm_kms[frame_id] if lsf else None,
    )
    return line.sign * values[f"{name}.flux"] * shape
