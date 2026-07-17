"""Built-in convolution kernels for composite line profiles.

A kernel is declared by passing one of this module's functions to
:meth:`~noobfriend.inference.spectrum.line.NoobLine.convolve`. Kernels are
defined in zero-centred velocity space (km/s), are flux-normalized, and are
parameterized by their FWHM in km/s so every width in a line declaration
speaks the same language. The function bodies are reference numpy
implementations; fit backends recognize the functions by identity and use
closed forms where they exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, pi, sqrt
from typing import TYPE_CHECKING

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


#: Registered built-in kernels, mapped to their backend dispatch kind.
KERNEL_KINDS: dict[object, str] = {gaussian: "gaussian", laplace: "laplace"}


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


__all__ = ["KERNEL_KINDS", "LineKernel", "gaussian", "laplace"]
