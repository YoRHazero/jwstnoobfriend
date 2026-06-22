"""Data-driven default priors -- the module's core value-add over raw numpyro.

A user who omits a parameter should still get a sensible, informative prior; they
write priors only for the one or two axes they care about. The omitted ones are
filled here, at model-build time, from seeds measured off the image: the centroid
and flux scale (which matter most for NUTS initialisation and scaling) and a loose
size from the light's second moment.

:class:`PriorPolicy` is the extensible seam; :class:`HighzPriorPolicy` is the only
implementation and the module default -- its assumptions (small sizes, low Sersic
index) are tuned for high-redshift galaxies. This module is internal; the policy
classes are re-exported from :mod:`noobfriend.inference.morphology`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from noobfriend.inference.morphology._param import (
    LogUniform,
    Normal,
    Prior,
    TruncNormal,
    Uniform,
)

if TYPE_CHECKING:
    from noobfriend.inference.morphology._band import Band
    from noobfriend.inference.morphology._geometry import BandGeometry


@dataclass(frozen=True)
class ImageSeeds:
    """Seeds measured off the image, feeding the default priors.

    Attributes
    ----------
    e0, n0 : float
        Light centroid offset east/north of the target centre, in arcsec.
    pixscale : float
        Representative native pixel scale, in arcsec.
    half_size : float
        Half the smaller stamp dimension, in arcsec (a size ceiling).
    r_eff : float
        A loose size seed (arcsec) from the second moment of the light.
    flux : dict[str, float]
        Positive flux scale per band, in the band's native data units.
    """

    e0: float
    n0: float
    pixscale: float
    half_size: float
    r_eff: float
    flux: dict[str, float]


class PriorPolicy(ABC):
    """Maps an omitted parameter axis to a concrete prior, given image seeds.

    Subclass and override :meth:`prior_for` to change the default-prior science.
    """

    @abstractmethod
    def prior_for(
        self, axis: str, seeds: ImageSeeds, *, band: str | None = None
    ) -> Prior:
        """Return the default prior for ``axis`` (``flux`` keyed by ``band``)."""


class HighzPriorPolicy(PriorPolicy):
    """High-redshift default priors: compact sizes, low Sersic index.

    Parameters
    ----------
    n_loc, n_scale : float
        Centre and width of the Sersic-index ``TruncNormal`` (bounded ``[0.3, 8]``).
    center_scale_pix : float
        Centroid prior width, in pixels.
    flux_decades : float
        Half-width, in decades, of the per-band flux ``LogUniform`` around the seed.
    """

    def __init__(
        self,
        *,
        n_loc: float = 1.5,
        n_scale: float = 1.0,
        center_scale_pix: float = 3.0,
        flux_decades: float = 3.0,
    ) -> None:
        self.n_loc = n_loc
        self.n_scale = n_scale
        self.center_scale_pix = center_scale_pix
        self.flux_decades = flux_decades

    def prior_for(
        self, axis: str, seeds: ImageSeeds, *, band: str | None = None
    ) -> Prior:
        """Return the high-z default prior for one axis."""
        scale = self.center_scale_pix * seeds.pixscale
        if axis == "x0":
            return Normal(seeds.e0, scale)
        if axis == "y0":
            return Normal(seeds.n0, scale)
        if axis == "r_eff":
            return TruncNormal(
                seeds.r_eff,
                seeds.r_eff,
                0.5 * seeds.pixscale,
                seeds.half_size,
            )
        if axis == "n":
            return TruncNormal(self.n_loc, self.n_scale, 0.3, 8.0)
        if axis == "q":
            return Uniform(0.1, 1.0)
        if axis == "theta":
            return Uniform(0.0, 180.0)
        if axis == "flux":
            if band is None:
                raise ValueError("flux prior needs a band.")
            f = seeds.flux[band]
            return LogUniform(f * 10.0**-self.flux_decades, f * 10.0**self.flux_decades)
        raise ValueError(f"no default prior for axis {axis!r}.")


def compute_seeds(
    bands: dict[str, Band],
    geometries: dict[str, BandGeometry],
    *,
    exclude: dict[str, np.ndarray] | None = None,
) -> ImageSeeds:
    """Measure image seeds from the unmasked light of each band.

    Centroid and size come from the band with the most positive flux; the flux
    scale is measured per band. The exclude masks must already fold in any
    neighbour auto-mask, so the centroid is not pulled onto a masked-out source.

    Parameters
    ----------
    bands : dict[str, Band]
        The per-band records.
    geometries : dict[str, BandGeometry]
        Matching per-band geometries.
    exclude : dict[str, numpy.ndarray], optional
        Per-band boolean exclude masks (``True`` = ignore). Defaults to each
        band's own ``mask``.

    Returns
    -------
    ImageSeeds
    """
    flux: dict[str, float] = {}
    best: tuple[float, float, float, float, float, float] | None = (
        None  # (sum,e0,n0,reff,px,half)
    )
    for name, band in bands.items():
        geom = geometries[name]
        mask = exclude.get(name) if exclude is not None else band.mask
        e0, n0, r_eff, total, pixscale, half = _band_seed(band, geom, mask)
        flux[name] = max(total, _tiny(total))
        if best is None or total > best[0]:
            best = (total, e0, n0, r_eff, pixscale, half)

    assert best is not None  # bands is non-empty by NoobImage invariant
    _, e0, n0, r_eff, pixscale, half = best
    return ImageSeeds(
        e0=e0, n0=n0, pixscale=pixscale, half_size=half, r_eff=r_eff, flux=flux
    )


def _band_seed(
    band: Band, geom: BandGeometry, exclude: np.ndarray | None
) -> tuple[float, float, float, float, float, float]:
    """Return ``(e0, n0, r_eff, total_flux, pixscale, half_size)`` for one band."""
    data = band.data
    err = band.error
    rows, cols = data.shape
    valid = np.isfinite(data) & np.isfinite(err) & (err > 0)
    if exclude is not None:
        valid &= ~exclude
    weight = np.where(valid, np.clip(data, 0.0, None), 0.0)

    pixscale = float(np.sqrt(geom.subpixel_area) * geom.oversample)
    half_size = 0.5 * min(rows, cols) * pixscale
    total = float(weight.sum())

    ys, xs = np.indices(data.shape)
    if total > 0:
        xcom = float((weight * xs).sum() / total)
        ycom = float((weight * ys).sum() / total)
        mxx = float((weight * (xs - xcom) ** 2).sum() / total)
        myy = float((weight * (ys - ycom) ** 2).sum() / total)
        r_pix = float(np.sqrt(max(0.5 * (mxx + myy), 0.0)))
    else:
        xcom, ycom = (cols - 1) / 2.0, (rows - 1) / 2.0
        r_pix = 1.0

    xc, yc = geom.pix_center
    affine_inv = np.linalg.inv(geom.affine)
    e0 = float(affine_inv[0, 0] * (xcom - xc) + affine_inv[0, 1] * (ycom - yc))
    n0 = float(affine_inv[1, 0] * (xcom - xc) + affine_inv[1, 1] * (ycom - yc))
    r_eff = max(r_pix * pixscale, pixscale)
    return e0, n0, r_eff, total, pixscale, half_size


def _tiny(scale: float) -> float:
    """Return a strictly-positive floor for a flux seed."""
    return abs(scale) * 1e-6 + 1e-30
