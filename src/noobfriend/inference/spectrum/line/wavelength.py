"""Observed-frame wavelength normalization."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose

from noobfriend.inference.spectrum.line.rules import _required_float
from noobfriend.inference.spectrum.types import VALID_WAVE_UNITS, WaveUnit


@dataclass(frozen=True, slots=True)
class Wavelength:
    """A line wavelength normalized around the observed frame."""

    z: float | None
    rest: float | None
    obs: float | None
    unit: WaveUnit

    @classmethod
    def normalize(
        cls,
        *,
        z: float | None,
        rest: float | None,
        obs: float | None,
        unit: WaveUnit,
    ) -> Wavelength:
        """Validate user wavelength inputs and preserve observed-frame access."""
        if unit not in VALID_WAVE_UNITS:
            raise ValueError(f"Unsupported wavelength unit: {unit!r}.")

        parsed_z = _optional_float(z, name="z")
        if parsed_z is not None and parsed_z <= 0:
            raise ValueError("z must be strictly positive when provided.")

        parsed_rest = _optional_positive_float(rest, name="rest")
        parsed_obs = _optional_positive_float(obs, name="obs")

        if parsed_rest is None and parsed_obs is None:
            raise ValueError("NoobLine requires rest or obs wavelength.")
        if parsed_rest is not None and parsed_obs is None and parsed_z is None:
            raise ValueError("z is required when rest is provided without obs.")
        if parsed_rest is not None and parsed_obs is not None and parsed_z is not None:
            expected_obs = parsed_rest * (1.0 + parsed_z)
            if not isclose(parsed_obs, expected_obs, rel_tol=1e-8, abs_tol=1e-8):
                raise ValueError("rest, obs, and z are inconsistent.")

        return cls(z=parsed_z, rest=parsed_rest, obs=parsed_obs, unit=unit)

    @property
    def observed(self) -> float:
        """Observed-frame wavelength in ``unit``."""
        if self.obs is not None:
            return self.obs
        if self.rest is None or self.z is None:
            raise RuntimeError(
                "Wavelength invariant violated: missing observed wavelength."
            )
        return self.rest * (1.0 + self.z)


def _optional_float(value: float | None, *, name: str) -> float | None:
    if value is None:
        return None
    return _required_float(value, name=name)


def _optional_positive_float(value: float | None, *, name: str) -> float | None:
    if value is None:
        return None
    parsed = _required_float(value, name=name)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive.")
    return parsed
