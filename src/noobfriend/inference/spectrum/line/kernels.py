"""Built-in convolution kernels for composite line profiles.

A kernel is declared by name — ``"gaussian"`` or ``"laplace"`` — passed to
:meth:`~noobfriend.inference.spectrum.line.NoobLine.convolve`; the matching
function in this module is accepted too. Kernels are defined in zero-centred
velocity space (km/s), are flux-normalized, and are parameterized by their
FWHM in km/s so every width in a line declaration speaks the same language.
The function bodies are reference numpy implementations; fit backends resolve
kernels to closed forms where they exist.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import log, pi, sqrt
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from noobfriend.inference.spectrum.line.rules import _ParameterRule

_GAUSSIAN_FWHM_TO_SIGMA = 2.0 * sqrt(2.0 * log(2.0))


def gaussian(x: np.ndarray, fwhm: float) -> np.ndarray:
    """Evaluate the normalized Gaussian kernel in velocity space.

    Parameters
    ----------
    x
        Velocity offsets in km/s.
    fwhm
        Full width at half maximum in km/s.

    Returns
    -------
    numpy.ndarray
        Kernel values integrating to one over ``x``.
    """
    sigma = fwhm / _GAUSSIAN_FWHM_TO_SIGMA
    return np.exp(-0.5 * (np.asarray(x, dtype=float) / sigma) ** 2) / (
        sigma * sqrt(2.0 * pi)
    )


def laplace(x: np.ndarray, fwhm: float) -> np.ndarray:
    """Evaluate the normalized two-sided exponential (Laplace) kernel.

    Parameters
    ----------
    x
        Velocity offsets in km/s.
    fwhm
        Full width at half maximum in km/s (scale ``b = fwhm / (2 ln 2)``).

    Returns
    -------
    numpy.ndarray
        Kernel values integrating to one over ``x``.
    """
    scale = fwhm / (2.0 * log(2.0))
    return np.exp(-np.abs(np.asarray(x, dtype=float)) / scale) / (2.0 * scale)


type KernelName = Literal["gaussian", "laplace"]
type KernelFunction = Callable[..., np.ndarray]

#: Built-in convolution kernels, keyed by the name that ``NoobLine.convolve``
#: accepts. Each value is the kernel's reference numpy implementation.
BUILTIN_KERNELS: dict[KernelName, KernelFunction] = {
    "gaussian": gaussian,
    "laplace": laplace,
}


def _resolve_kernel(
    kernel: KernelName | KernelFunction,
) -> tuple[KernelName, KernelFunction]:
    """Normalize a kernel spec to its ``(name, function)`` pair.

    Accepts a built-in kernel name (``"gaussian"`` or ``"laplace"``) or the
    matching built-in kernel function; any other value is reserved for the
    not-yet-implemented numerical convolution backend.

    Raises
    ------
    ValueError
        If ``kernel`` is a string that is not a built-in kernel name.
    TypeError
        If ``kernel`` is neither a built-in name nor a built-in function.
    """
    if isinstance(kernel, str):
        function = BUILTIN_KERNELS.get(kernel)
        if function is None:
            valid = ", ".join(repr(name) for name in BUILTIN_KERNELS)
            raise ValueError(
                f"unknown built-in kernel {kernel!r}; choose from {valid}."
            )
        return kernel, function
    for name, function in BUILTIN_KERNELS.items():
        if kernel is function:
            return name, function
    raise TypeError(
        'convolve expects a built-in kernel name ("gaussian" or "laplace") or a '
        "built-in kernel function from noobfriend.inference.spectrum.line.kernels; "
        "arbitrary functions require the numerical convolution backend, which is "
        "not implemented yet."
    )


@dataclass(frozen=True, slots=True)
class LineKernel:
    """One convolution kernel attached to a line declaration.

    Attributes
    ----------
    kind
        Backend dispatch token (``"gaussian"`` or ``"laplace"``).
    name
        Namespace prefix for this kernel's shape parameters; a parameter
        ``p`` appears as the line shape parameter ``"{name}__{p}"``.
    rules
        Ordered ``(parameter, rule)`` pairs, one per kernel shape parameter.
    """

    kind: str
    name: str
    rules: tuple[tuple[str, _ParameterRule], ...]

    def shape_names(self) -> tuple[str, ...]:
        """Return the fully-qualified shape parameter names of this kernel."""
        return tuple(f"{self.name}__{parameter}" for parameter, _ in self.rules)

    def shape_name(self, parameter: str) -> str:
        """Return the fully-qualified name of one shape parameter."""
        return f"{self.name}__{parameter}"


__all__ = [
    "BUILTIN_KERNELS",
    "KernelFunction",
    "KernelName",
    "LineKernel",
    "gaussian",
    "laplace",
]
