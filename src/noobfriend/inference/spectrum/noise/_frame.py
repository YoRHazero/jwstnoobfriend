"""Per-frame noise structure: likelihood, whitening and leave-one-out terms.

:class:`FrameNoise` freezes what one frame's noise model needs on its fitted
pixels (errors, native-index gaps, the unit-scale Cholesky factor of a fixed
correlation) so the formulas below run with NumPy or JAX through ``xp``.

Leave-one-out uses the exact Gaussian conditional ``y_i | y_-i``: with
precision ``Q = Sigma^-1`` and residual ``r``, its log density is
``log N(Qr_i / Q_ii; 0, 1 / Q_ii)`` (a non-factorized model needs no refit).
For AR(1) ``Q`` is tridiagonal, so every term is O(n).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from math import log, pi
from typing import Any, Literal

import numpy as np

from noobfriend.inference.spectrum._xp import readonly
from noobfriend.inference.spectrum.noise._models import AR1, Autocorrelation, Noise
from noobfriend.inference.spectrum.priors import Prior

_HALF_LOG_TWO_PI = 0.5 * log(2.0 * pi)

type NoiseKind = Literal["white", "fixed", "ar1"]


@dataclass(frozen=True, eq=False)
class FrameNoise:
    """Noise structure of one frame's fitted pixels.

    Attributes
    ----------
    noise
        The frame's declaration.
    sigma
        Error array on the fitted pixels.
    gaps
        Native-index separation between consecutive fitted pixels, length
        ``n - 1``.
    cholesky
        Lower Cholesky factor of ``diag(sigma) R diag(sigma)`` (unit scale)
        for an :class:`Autocorrelation`; ``None`` otherwise.
    """

    noise: Noise
    sigma: np.ndarray
    gaps: np.ndarray
    cholesky: np.ndarray | None

    @classmethod
    def build(
        cls, noise: Noise, sigma: np.ndarray, pixel_index: np.ndarray
    ) -> FrameNoise:
        """Precompute the structure for one frame.

        Parameters
        ----------
        noise
            The frame's declaration.
        sigma
            Errors on the fitted pixels.
        pixel_index
            Native pixel indices of the fitted pixels, increasing.

        Returns
        -------
        FrameNoise
            The frozen structure.

        Raises
        ------
        ValueError
            If a fixed correlation gives a covariance that is not positive
            definite on these pixels.
        """
        sigma = readonly(sigma)
        index = np.asarray(pixel_index, dtype=int)
        gaps = readonly(np.diff(index), dtype=int)
        cholesky = None
        if isinstance(noise.correlation, Autocorrelation):
            lags = np.abs(index[:, None] - index[None, :])
            covariance = noise.correlation.correlations(lags) * np.outer(sigma, sigma)
            try:
                cholesky = readonly(np.linalg.cholesky(covariance))
            except np.linalg.LinAlgError:
                raise ValueError(
                    f"{noise.correlation} gives a singular covariance on this "
                    "frame's fitted pixels; its spectral density touches zero."
                ) from None
        return cls(noise=noise, sigma=sigma, gaps=gaps, cholesky=cholesky)

    @property
    def kind(self) -> NoiseKind:
        """``"white"``, ``"fixed"`` (autocorrelation) or ``"ar1"``."""
        correlation = self.noise.correlation
        if correlation is None:
            return "white"
        if isinstance(correlation, AR1):
            return "ar1"
        return "fixed"

    @property
    def size(self) -> int:
        """Number of fitted pixels."""
        return int(self.sigma.size)

    @cached_property
    def precision(self) -> np.ndarray:
        """Unit-scale precision matrix of a fixed correlation."""
        if self.cholesky is None:
            raise AttributeError("only a fixed correlation has a dense precision.")
        inverse = np.linalg.inv(self.cholesky)
        return readonly(inverse.T @ inverse)

    def log_likelihood(self, xp: Any, resid: Any, scale: Any, rho: Any = None) -> Any:
        """Joint log density of residuals ``r = y - mean``.

        Parameters
        ----------
        xp
            Array namespace.
        resid
            Residuals, shape ``(..., n)``.
        scale
            Noise scale, shape ``(...)`` or scalar.
        rho
            AR(1) correlation when free; ignored otherwise.

        Returns
        -------
        array
            Log likelihood, shape ``(...)``.
        """
        scale = _batch(xp, scale)
        kind = self.kind
        if kind == "white":
            return xp.sum(self.pointwise(xp, resid, scale[..., 0]), axis=-1)
        whitened = self.whiten(xp, resid, scale[..., 0], rho)
        n = self.size
        quadratic = -0.5 * xp.sum(whitened**2, axis=-1) - n * _HALF_LOG_TWO_PI
        if kind == "fixed":
            log_det = float(np.sum(np.log(np.diag(self.cholesky))))
            return quadratic - log_det - n * xp.log(scale[..., 0])
        phi = self._couplings(xp, rho)
        log_det = 0.5 * xp.sum(xp.log1p(-(phi**2)), axis=-1)
        return quadratic - log_det - xp.sum(xp.log(scale * self.sigma), axis=-1)

    def pointwise(self, xp: Any, resid: Any, scale: Any, rho: Any = None) -> Any:
        """Per-pixel log densities for leave-one-out.

        White noise gives the factorized normal densities; correlated noise
        gives the exact conditional densities ``log p(y_i | y_-i)``.

        Returns
        -------
        array
            Shape ``(..., n)``.
        """
        scale = _batch(xp, scale)
        kind = self.kind
        if kind == "white":
            x = resid / (scale * self.sigma)
            return -0.5 * x**2 - xp.log(scale * self.sigma) - _HALF_LOG_TWO_PI
        if kind == "fixed":
            precision = self.precision
            q_resid = xp.matmul(resid, precision) / scale**2
            q_diag = np.diag(precision) / scale**2
            return 0.5 * xp.log(q_diag) - _HALF_LOG_TWO_PI - 0.5 * q_resid**2 / q_diag
        x = resid / (scale * self.sigma)
        phi = self._couplings(xp, rho)
        variance = 1.0 - phi**2
        left_phi = _pad(xp, phi, before=True)
        right_phi = _pad(xp, phi, before=False)
        diag = 1.0 / (1.0 - left_phi**2) + right_phi**2 / (1.0 - right_phi**2)
        off = -phi / variance
        q_x = diag * x
        q_x = q_x + _pad(xp, off * x[..., :-1], before=True)
        q_x = q_x + _pad(xp, off * x[..., 1:], before=False)
        conditional = 0.5 * xp.log(diag) - _HALF_LOG_TWO_PI - 0.5 * q_x**2 / diag
        return conditional - xp.log(scale * self.sigma)

    def whiten(self, xp: Any, resid: Any, scale: Any, rho: Any = None) -> Any:
        """Map residuals to independent unit normals under the noise model.

        Returns
        -------
        array
            Whitened residuals, shape ``(..., n)``; their sum of squares is
            the frame's chi-square.
        """
        scale = _batch(xp, scale)
        kind = self.kind
        if kind == "white":
            return resid / (scale * self.sigma)
        if kind == "fixed":
            shape = resid.shape
            flat = xp.reshape(resid, (-1, shape[-1]))
            solved = xp.linalg.solve_triangular(self.cholesky, flat.T, lower=True).T
            return xp.reshape(solved, shape) / scale
        x = resid / (scale * self.sigma)
        phi = self._couplings(xp, rho)
        innovations = (x[..., 1:] - phi * x[..., :-1]) / xp.sqrt(1.0 - phi**2)
        return xp.concatenate([x[..., :1], innovations], axis=-1)

    def distribution(self, mean: Any, scale: Any, rho: Any = None) -> Any:
        """Return the NumPyro observation distribution for this frame.

        White noise is an independent ``Normal`` (one log-likelihood value
        per pixel); a fixed correlation is a ``MultivariateNormal`` with
        ``scale_tril``; AR(1) is :func:`ar1_distribution`'s Markov normal.
        The last two give one value per frame.
        """
        import numpyro.distributions as dist

        kind = self.kind
        if kind == "white":
            return dist.Normal(mean, scale * self.sigma)
        if kind == "fixed":
            return dist.MultivariateNormal(mean, scale_tril=scale * self.cholesky)
        from noobfriend.inference.spectrum.noise._numpyro import ar1_distribution

        return ar1_distribution()(mean, scale * self.sigma, self._rho(rho), self.gaps)

    def _rho(self, rho: Any) -> Any:
        declared = self.noise.correlation.rho  # type: ignore[union-attr]
        if rho is None:
            if isinstance(declared, Prior):
                raise ValueError("this frame's AR(1) rho is free; pass its value.")
            return declared
        return rho

    def _couplings(self, xp: Any, rho: Any) -> Any:
        """``rho ** gap`` between consecutive fitted pixels, shape ``(..., n - 1)``."""
        return ar1_couplings(xp, _batch(xp, self._rho(rho)), self.gaps)


def ar1_couplings(xp: Any, rho: Any, gaps: Any) -> Any:
    """Return ``rho ** gaps`` exactly for negative ``rho`` too.

    Parameters
    ----------
    xp
        Array namespace.
    rho
        Lag-one correlation, broadcastable against ``gaps``.
    gaps
        Positive integer pixel separations.

    Returns
    -------
    array
        Correlation between pixels ``gaps`` apart.
    """
    odd = (gaps % 2) == 1
    magnitude = xp.abs(rho) ** gaps
    return xp.where(odd, xp.sign(rho) * magnitude, magnitude)


def _batch(xp: Any, value: Any) -> Any:
    """Append a trailing axis so a per-draw scalar broadcasts over pixels."""
    return xp.asarray(value)[..., None]


def _pad(xp: Any, values: Any, *, before: bool) -> Any:
    """Pad the last axis with one zero before or after."""
    zeros = xp.zeros_like(values[..., :1])
    parts = [zeros, values] if before else [values, zeros]
    return xp.concatenate(parts, axis=-1)
