"""Default priors that ``prepare`` writes out for undeclared parameters.

Every default becomes an explicit prior in the parameter table (and the
summary), so MLE and MCMC read the same object. Data-derived defaults use two
scales of the fitted pixels: the flux scale ``S`` and the half wavelength span
``W``.
"""

from __future__ import annotations

from noobfriend.inference.spectrum.priors import (
    LogUniform,
    Normal,
    Prior,
    TruncatedNormal,
    Uniform,
)

#: Line-centre velocity offset (km/s).
CENTER_PRIOR = Uniform(-300.0, 300.0)
#: Intrinsic FWHM (km/s) by component.
FWHM_PRIORS: dict[str, Prior] = {
    "narrow": LogUniform(10.0, 700.0),
    "broad": LogUniform(800.0, 5000.0),
}
#: Typical velocity offset (km/s): the sampling scale of centre parameters.
VELOCITY_SCALE_KMS = 100.0
#: Representative FWHM (km/s) by component, for data-derived flux scales and
#: optimizer starts.
NOMINAL_FWHM_KMS: dict[str, float] = {"narrow": 200.0, "broad": 1200.0}
#: Flux prior ``TruncatedNormal(0, FLUX_MULTIPLIER * ref, low=0)``.
FLUX_MULTIPLIER = 5.0
#: Continuum coefficient prior ``Normal(0, CONTINUUM_SIGMA * S / W**n)``.
CONTINUUM_SIGMA = 5.0
#: Pooled spread prior ``TruncatedNormal(0, SPREAD_SIGMA * S / W**n, low=0)``.
SPREAD_SIGMA = 0.5
#: Power-law index prior.
INDEX_PRIOR = Normal(0.0, 5.0)


def flux_prior(reference: float) -> TruncatedNormal:
    """Half-normal flux prior from a data-derived reference flux."""
    return TruncatedNormal(0.0, FLUX_MULTIPLIER * reference, low=0.0)


def coefficient_prior(
    flux_scale: float, wavelength_scale: float, degree: int
) -> Normal:
    """Polynomial coefficient prior ``Normal(0, 5 S / W**n)``."""
    return Normal(0.0, CONTINUUM_SIGMA * flux_scale / wavelength_scale**degree)


def level_prior(flux_scale: float) -> TruncatedNormal:
    """Power-law level prior ``TruncatedNormal(0, 5 S, low=0)``."""
    return TruncatedNormal(0.0, CONTINUUM_SIGMA * flux_scale, low=0.0)


def spread_prior(
    name: str, flux_scale: float, wavelength_scale: float
) -> TruncatedNormal:
    """Pooled spread prior of one continuum parameter.

    Coefficients spread in their own units; the power-law level pools in log
    space and the index additively, both with a dimensionless spread.
    """
    if name.startswith("c"):
        degree = int(name[1:])
        return TruncatedNormal(
            0.0, SPREAD_SIGMA * flux_scale / wavelength_scale**degree, low=0.0
        )
    return TruncatedNormal(0.0, SPREAD_SIGMA, low=0.0)
