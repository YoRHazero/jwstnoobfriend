"""Prior configuration for the decomposition model family.

Every sampled coordinate has a standard-normal prior; the numbers here are the
locations/scales of the deterministic transforms that give those coordinates
physical meaning (see ``_transforms``). Consequently the prior predictive
distribution is fully determined by this module plus the transform layer, and
prior sampling never needs an inference backend.

Linear amplitudes (component fluxes and per-band backgrounds) are never
sampled: they carry zero-mean Gaussian priors with the scales configured here
and are marginalized analytically inside the likelihood.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from noobfriend.inference.morphology.data import NoobImageSet

__all__ = [
    "AmplitudePriors",
    "CenterPrior",
    "HostPriors",
    "PointPriors",
    "Priors",
]


@dataclass(frozen=True)
class CenterPrior:
    """Normal prior on a centre in the shared arcsec frame.

    Parameters
    ----------
    x_loc, y_loc : float
        Prior mean of the centre in arcsec.
    scale : float
        Prior standard deviation in arcsec, shared by both axes.
    """

    x_loc: float = 0.0
    y_loc: float = 0.0
    scale: float = 0.1

    def __post_init__(self) -> None:
        """Validate the prior scale."""
        if self.scale <= 0 or not np.isfinite(self.scale):
            raise ValueError("CenterPrior.scale must be positive and finite.")


@dataclass(frozen=True)
class PointPriors:
    """Priors for point-source centres.

    Parameters
    ----------
    center : CenterPrior
        Prior for the primary point source (index 0), which is also the
        anchor of the host-containment constraint.
    extra_center : CenterPrior
        Prior for every additional point source (indices >= 1); typically
        much wider than ``center``.
    """

    center: CenterPrior = CenterPrior(scale=0.1)
    extra_center: CenterPrior = CenterPrior(scale=0.5)


@dataclass(frozen=True)
class HostPriors:
    """Priors for the Sersic host component.

    Parameters
    ----------
    log_r_eff_loc, log_r_eff_scale : float
        Normal prior on ``log(r_eff / arcsec)``.
    log_n_loc, log_n_scale : float
        Normal prior on ``log(n)`` (Sersic index).
    e_max : float
        Open upper bound on the ellipticity modulus ``|e|`` where
        ``e = (1 - q) / (1 + q)``; the ellipticity vector lives on the disk
        of this radius via a singularity-free bijection.
    e_scale : float
        Scale applied to the latent ellipticity coordinates before the disk
        map; larger values put more prior mass at high ellipticity.
    offset_scale : float
        Scale applied to the latent host-offset coordinates before the disk
        map. The offset disk is in normalized elliptical-radius units, so the
        primary point source is inside the one-``r_eff`` host ellipse by
        construction.
    center : CenterPrior
        Prior on the host centre, used only for host-only models
        (``n_points == 0``) where no containment anchor exists.
    """

    log_r_eff_loc: float = math.log(0.15)
    log_r_eff_scale: float = 0.7
    log_n_loc: float = math.log(1.5)
    log_n_scale: float = 0.5
    e_max: float = 0.9
    e_scale: float = 1.0
    offset_scale: float = 1.0
    center: CenterPrior = CenterPrior(scale=0.1)

    def __post_init__(self) -> None:
        """Validate scale-like fields."""
        for name in ("log_r_eff_scale", "log_n_scale", "e_scale", "offset_scale"):
            value = getattr(self, name)
            if value <= 0 or not np.isfinite(value):
                raise ValueError(f"HostPriors.{name} must be positive and finite.")
        if not 0 < self.e_max < 1:
            raise ValueError("HostPriors.e_max must be in (0, 1).")


@dataclass(frozen=True)
class AmplitudePriors:
    """Zero-mean Gaussian prior scales for the marginalized amplitudes.

    Parameters
    ----------
    flux_scale : dict of str to float
        Per-band prior standard deviation for every source amplitude
        (point fluxes and host flux), in image flux units.
    background_scale : dict of str to float
        Per-band prior standard deviation for the constant background level.

    Notes
    -----
    Amplitudes are Gaussian, not truncated-positive, so marginalization stays
    closed-form; a source amplitude posterior overlapping zero reads as
    "consistent with absent", and presence is decided by model comparison.
    The scales enter the marginal likelihood's Occam factor, so they should
    be generous but not absurd; :meth:`Priors.from_images` derives them from
    the data.
    """

    flux_scale: dict[str, float] = field(default_factory=dict)
    background_scale: dict[str, float] = field(default_factory=dict)

    def for_band(self, band: str) -> tuple[float, float]:
        """Return ``(flux_scale, background_scale)`` for one band.

        Raises
        ------
        KeyError
            If the band has no configured scales.
        """
        return self.flux_scale[band], self.background_scale[band]


@dataclass(frozen=True)
class Priors:
    """Complete prior configuration for one decomposition run."""

    point: PointPriors = PointPriors()
    host: HostPriors = HostPriors()
    amplitudes: AmplitudePriors = AmplitudePriors()

    @classmethod
    def from_images(cls, images: "NoobImageSet", **overrides: object) -> "Priors":
        """Build data-driven default priors for an image set.

        Centre scales come from the pixel scale and stamp field of view;
        amplitude scales come from robust per-band flux statistics.

        Parameters
        ----------
        images : NoobImageSet
            The prepared images to be fit.
        **overrides
            Replacement values for the ``point``, ``host`` or ``amplitudes``
            blocks (each replacing the whole block).

        Returns
        -------
        Priors
            Data-driven defaults with any provided blocks replaced.
        """
        min_pix = min(image.pixel_scale for image in images)
        field_arcsec = min(min(image.shape) * image.pixel_scale for image in images)
        center_scale = max(4.0 * min_pix, 0.05)
        extra_scale = max(field_arcsec / 4.0, center_scale)
        flux_scale: dict[str, float] = {}
        background_scale: dict[str, float] = {}
        for image in images:
            values = image.data[image.mask]
            median = float(np.nanmedian(values)) if values.size else 0.0
            mad_std = (
                float(1.4826 * np.nanmedian(np.abs(values - median)))
                if values.size
                else 1.0
            )
            positive = float(np.nansum(np.clip(values - median, 0.0, None)))
            flux_scale[image.name] = max(3.0 * positive, 10.0 * mad_std, 1e-12)
            background_scale[image.name] = max(abs(median) + 5.0 * mad_std, 1e-12)
        r_eff_loc = math.log(max(3.0 * min_pix, 1e-3))
        defaults = cls(
            point=PointPriors(
                center=CenterPrior(scale=center_scale),
                extra_center=CenterPrior(scale=extra_scale),
            ),
            host=HostPriors(
                log_r_eff_loc=r_eff_loc,
                center=CenterPrior(scale=center_scale),
            ),
            amplitudes=AmplitudePriors(
                flux_scale=flux_scale, background_scale=background_scale
            ),
        )
        return replace(defaults, **overrides) if overrides else defaults
