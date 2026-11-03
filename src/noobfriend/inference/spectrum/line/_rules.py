"""Parameter rules recorded on a line declaration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from noobfriend.inference.spectrum.priors import ParameterValue

type CenterKind = Literal[
    "default", "delta_v_kms", "delta_wavelength", "delta_v_from_base_kms"
]
type ScaleKind = Literal["default", "independent", "ratio_to_base"]


@dataclass(frozen=True, slots=True)
class LineRule:
    """How one line parameter is set.

    Attributes
    ----------
    kind
        For the centre: ``"default"``, ``"delta_v_kms"``,
        ``"delta_wavelength"`` (both relative to the line's own nominal
        wavelength) or ``"delta_v_from_base_kms"`` (the velocity difference
        from the base). For width and flux: ``"default"``, ``"independent"``
        or ``"ratio_to_base"``. ``"default"`` is resolved to an explicit prior
        by ``prepare``.
    value
        The fixed value or prior; ``None`` for ``"default"``.
    """

    kind: str
    value: ParameterValue | None = None

    @property
    def is_relative(self) -> bool:
        """Whether the rule is stated relative to the base line."""
        return self.kind in ("delta_v_from_base_kms", "ratio_to_base")

    def __str__(self) -> str:
        """Compact form, e.g. ``ratio_to_base=1``."""
        return "default" if self.kind == "default" else f"{self.kind}={self.value}"


@dataclass(frozen=True, slots=True)
class LineKernel:
    """One convolution kernel on a line.

    Attributes
    ----------
    kind
        ``"gaussian"`` or ``"laplace"``.
    fwhm
        Kernel FWHM in km/s (fixed or prior), or ``None`` when locked to the
        base line's kernel of the same kind.
    fraction
        Convolved fraction in ``(0, 1]`` (fixed or prior), or ``None`` when
        locked to the base.
    """

    kind: str
    fwhm: ParameterValue | None
    fraction: ParameterValue | None

    def __str__(self) -> str:
        """Compact form."""
        fwhm = "locked" if self.fwhm is None else self.fwhm
        fraction = "locked" if self.fraction is None else self.fraction
        return f"{self.kind}(fwhm={fwhm}, fraction={fraction})"
