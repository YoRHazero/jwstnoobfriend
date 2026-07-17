"""Continuum specification for one prepared spectrum model."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from collections.abc import Sequence

    from noobfriend.inference.spectrum.workspace.handles import LineHandle


type ContinuumSharing = Literal["pooled", "independent", "shared"]

_VALID_SHARING: frozenset[str] = frozenset({"pooled", "independent", "shared"})


@dataclass(frozen=True, slots=True)
class ContinuumSpec:
    """Polynomial continuum anchored at an observed-frame wavelength.

    ``sharing`` selects how continuum coefficients are combined across the
    frames of a joint fit: ``"pooled"`` (non-centered partial pooling),
    ``"independent"`` (a free continuum per frame), or ``"shared"`` (one
    continuum for every frame). It has no effect on a single-frame fit, whose
    continuum is always a single set of coefficients.
    """

    lambda_0: float
    order: int = 1
    sharing: ContinuumSharing = "pooled"

    def __post_init__(self) -> None:
        """Normalize and validate continuum inputs."""
        if (
            isinstance(self.order, bool)
            or not isinstance(self.order, int)
            or self.order < 0
        ):
            raise ValueError("continuum_order must be a nonnegative integer.")
        if self.sharing not in _VALID_SHARING:
            raise ValueError(
                f"continuum_sharing {self.sharing!r} must be one of "
                f"{tuple(sorted(_VALID_SHARING))}."
            )
        lambda_0 = float(self.lambda_0)
        if not isfinite(lambda_0):
            raise ValueError("continuum_lambda_0 must be finite.")
        object.__setattr__(self, "lambda_0", lambda_0)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Continuum parameter names in design-matrix column order."""
        return ("c", *(f"k{degree}" for degree in range(1, self.order + 1)))

    def design(self, wavelength: ArrayLike) -> np.ndarray:
        """Return columns for ``c + sum(k_i * (lambda - lambda_0)**i)``."""
        wl = np.asarray(wavelength, dtype=float)
        if wl.ndim != 1:
            raise ValueError("continuum wavelength must be a 1-D array.")
        x = wl - self.lambda_0
        columns = [np.ones_like(x), *(x**degree for degree in range(1, self.order + 1))]
        design = np.column_stack(columns)
        design.setflags(write=False)
        return design


def build_continuum(
    handles: Sequence[LineHandle],
    *,
    continuum_order: int = 1,
    continuum_lambda_0: float | None = None,
    continuum_sharing: ContinuumSharing = "pooled",
) -> ContinuumSpec:
    """Resolve prepare-time continuum options into a concrete spec."""
    if continuum_lambda_0 is None:
        if not handles:
            raise ValueError(
                "continuum_lambda_0 can only be omitted when at least one line is prepared."
            )
        continuum_lambda_0 = handles[0].observed_wavelength
    return ContinuumSpec(
        lambda_0=continuum_lambda_0,
        order=continuum_order,
        sharing=continuum_sharing,
    )
