"""Noise declarations attached to spectrum frames.

Each frame's fitted pixels have covariance

    Sigma_ij = s**2 * sigma_i * sigma_j * rho(|k_i - k_j|),  rho(0) = 1,

with ``sigma`` the frame's error array, ``s`` the :class:`Noise` scale and
``k`` the native pixel index. White noise is ``rho(d >= 1) = 0``. A masked gap
simply drops rows and columns, so the remaining pixels keep their true
separations.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

import numpy as np

from noobfriend.inference.spectrum.priors import (
    CORRELATION,
    POSITIVE,
    ParameterValue,
    Prior,
    check_value,
)

#: Smallest spectral density accepted for an :class:`Autocorrelation`.
_SPECTRAL_TOLERANCE = -1e-10


@dataclass(frozen=True, slots=True)
class AR1:
    """First-order autoregressive correlation ``rho(d) = rho**d``.

    Parameters
    ----------
    rho
        Lag-one correlation in ``(-1, 1)``: a fixed ``float`` or a prior.
        It is the only correlation family whose parameter may be free, since
        its likelihood has an exact O(n) factorization.
    """

    rho: ParameterValue

    def __post_init__(self) -> None:
        """Validate ``rho``."""
        object.__setattr__(self, "rho", check_value(self.rho, CORRELATION, "AR1.rho"))

    def correlations(self, lags: np.ndarray, rho: float | None = None) -> np.ndarray:
        """Evaluate the correlation at integer lags.

        Parameters
        ----------
        lags
            Non-negative integer lags.
        rho
            Lag-one correlation; defaults to the fixed declaration.

        Returns
        -------
        numpy.ndarray
            ``rho**lags``.
        """
        value = self.rho if rho is None else rho
        if isinstance(value, Prior):
            raise ValueError("AR1.rho is free; pass a value to evaluate it.")
        return np.power(float(value), np.asarray(lags, dtype=float))

    def __str__(self) -> str:
        """Compact form."""
        return f"AR1(rho={self.rho})"


@dataclass(frozen=True, slots=True)
class Autocorrelation:
    """Fixed correlation at lags ``1 ... K``, zero beyond.

    Parameters
    ----------
    values
        ``rho(1), ..., rho(K)``. The implied spectral density
        ``1 + 2 sum_k rho(k) cos(k w)`` must be non-negative for every ``w``,
        which makes every covariance matrix built from it positive
        semi-definite.

    Raises
    ------
    ValueError
        If ``values`` is empty, non-finite, or has a negative spectral
        density somewhere.
    """

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate the lags and the spectral density."""
        values = tuple(float(value) for value in _sequence(self.values))
        if not values:
            raise ValueError(
                "Autocorrelation needs at least one lag; use Noise() for white noise."
            )
        if not all(isfinite(value) for value in values):
            raise ValueError("Autocorrelation values must be finite.")
        minimum = _minimum_spectral_density(np.asarray(values))
        if minimum < _SPECTRAL_TOLERANCE:
            raise ValueError(
                "Autocorrelation is not a valid correlation function: its "
                f"spectral density reaches {minimum:.3g} < 0."
            )
        object.__setattr__(self, "values", values)

    @property
    def max_lag(self) -> int:
        """The largest lag with a declared correlation."""
        return len(self.values)

    def correlations(self, lags: np.ndarray) -> np.ndarray:
        """Evaluate the correlation at integer lags (``rho(0) = 1``)."""
        lags = np.asarray(lags, dtype=int)
        table = np.concatenate([[1.0], self.values, [0.0]])
        return table[np.clip(lags, 0, self.max_lag + 1)]

    def __str__(self) -> str:
        """Compact form listing the lags."""
        shown = ", ".join(f"{value:.3g}" for value in self.values)
        return f"Autocorrelation({shown})"


@dataclass(frozen=True, slots=True)
class Noise:
    """A frame's noise model on top of its error array.

    Parameters
    ----------
    scale
        Multiplier ``s`` on the error array: a fixed ``float`` (``1.0`` keeps
        the errors as given) or a prior to fit it.
    correlation
        ``None`` for white noise, :class:`AR1` or :class:`Autocorrelation`.
    """

    scale: ParameterValue = 1.0
    correlation: AR1 | Autocorrelation | None = None

    def __post_init__(self) -> None:
        """Validate the scale and correlation."""
        object.__setattr__(
            self, "scale", check_value(self.scale, POSITIVE, "Noise.scale")
        )
        if self.correlation is not None and not isinstance(
            self.correlation, (AR1, Autocorrelation)
        ):
            raise TypeError(
                "Noise.correlation must be None, AR1(...) or Autocorrelation(...); "
                f"got {type(self.correlation).__name__}."
            )

    @property
    def is_white(self) -> bool:
        """Whether the noise has no pixel-to-pixel correlation."""
        return self.correlation is None

    def __str__(self) -> str:
        """Compact form."""
        correlation = "white" if self.correlation is None else str(self.correlation)
        return f"Noise(scale={self.scale}, {correlation})"


def estimated_noise(scale: float, correlations: Sequence[float]) -> Noise:
    """Wrap a noise estimate as a fixed :class:`Noise`.

    Parameters
    ----------
    scale
        Estimated error scale.
    correlations
        Estimated ``rho(1), ..., rho(K)``; empty for a scale-only estimate.

    Returns
    -------
    Noise
        Fixed scale, white or :class:`Autocorrelation` correlation.
    """
    correlation = Autocorrelation(tuple(correlations)) if len(correlations) else None
    return Noise(scale=float(scale), correlation=correlation)


def _sequence(values: object) -> Sequence[float]:
    if isinstance(values, (str, bytes)) or not isinstance(
        values, (Sequence, np.ndarray)
    ):
        raise TypeError("Autocorrelation values must be a sequence of floats.")
    return values  # type: ignore[return-value]


def _minimum_spectral_density(values: np.ndarray) -> float:
    """Minimum over frequency of ``1 + 2 sum_k rho(k) cos(k w)``."""
    frequencies = np.linspace(0.0, np.pi, max(4096, 64 * values.size))
    lags = np.arange(1, values.size + 1)
    density = 1.0 + 2.0 * np.cos(np.outer(frequencies, lags)) @ values
    return float(density.min())
