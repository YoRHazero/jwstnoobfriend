"""Detect compact sources and measure their shape (thin scikit-image wrapper).

Mechanism only, pixel-space (like :mod:`noobfriend.core.imgutils._segment`): a
matched-filter peak detector (:func:`skimage.feature.peak_local_max`) plus
**intensity-weighted** second moments (:func:`skimage.measure.moments` /
:func:`~skimage.measure.moments_central`) for shape. Everything is pixel space
and carries no astronomical interpretation -- there is no notion of "star".

Sky coordinates are attached only through an optional ``pixel_to_world`` callable
``(x, y) -> (ra, dec)``: this module never imports a WCS library, which keeps it
agnostic to the astropy-WCS vs gwcs interface difference (the caller adapts its
WCS to a callable). Turning the catalogue's columns into a stellar selection is
the caller's job; that is what keeps this in ``core``.

Shape is measured from the weighted central moments in a **fixed-radius
aperture** with negatives clipped to zero. Two deliberate choices: the binary
shape properties in :func:`skimage.measure.regionprops` (``eccentricity`` /
``axis_*_length``) ignore pixel intensity and are wrong for a PSF, so the raw
*weighted* moments are used instead; and a fixed aperture (rather than a
flux-threshold mask) gives a constant, predictable size bias so the stellar
locus stays tight across brightnesses.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from skimage.measure import moments, moments_central

if TYPE_CHECKING:
    from astropy.table import Table

_MAD_TO_SIGMA = 1.4826
#: FWHM of a Gaussian in units of its standard deviation, ``2 sqrt(2 ln 2)``.
_FWHM_PER_SIGMA = 2.354820045030949


@dataclass(frozen=True)
class SourceCatalog:
    """A columnar catalogue of detected sources (one entry per source).

    Every attribute is a 1-D array of the same length (one row per source). Index
    with a boolean or integer-index array to subset (``cat[cat.ellipticity <
    0.1]``); the result is a new :class:`SourceCatalog`. :func:`build_catalog`
    returns rows ordered by descending :attr:`peak`, but subsetting and
    :meth:`concat` do not preserve that order.

    Attributes
    ----------
    x, y : numpy.ndarray
        Intensity-weighted centroid, in pixel-index coordinates (integer index =
        pixel centre, astropy convention).
    ra, dec : numpy.ndarray
        Sky coordinates in degrees from the ``pixel_to_world`` callable, or
        ``NaN`` when none was supplied.
    fwhm : numpy.ndarray
        Round-source FWHM in pixels, ``2.3548 * sqrt(0.5 * (var_x + var_y))``
        from the weighted second moments.
    ellipticity : numpy.ndarray
        ``1 - b/a`` from the weighted-covariance eigenvalues; ``0`` is round.
    flux : numpy.ndarray
        Aperture-summed flux (the zeroth weighted moment, negatives clipped).
    peak : numpy.ndarray
        Value of the brightest pixel (data is already background-subtracted).
    snr : numpy.ndarray
        Aperture flux over its propagated noise (from ``err`` when given, else
        the robust per-pixel sky sigma).
    sharpness : numpy.ndarray
        Peak pixel over the mean of its eight neighbours -- high for hot pixels
        and cosmic rays, moderate for resolved sources.
    nn_dist : numpy.ndarray
        Distance (pixels) to the nearest other detection; ``inf`` if alone.
    """

    x: np.ndarray
    y: np.ndarray
    ra: np.ndarray
    dec: np.ndarray
    fwhm: np.ndarray
    ellipticity: np.ndarray
    flux: np.ndarray
    peak: np.ndarray
    snr: np.ndarray
    sharpness: np.ndarray
    nn_dist: np.ndarray

    def __len__(self) -> int:
        """Return the number of sources."""
        return int(self.x.size)

    def __getitem__(self, key: np.ndarray) -> "SourceCatalog":
        """Return the sub-catalogue selected by a boolean or integer-index array."""
        return SourceCatalog(
            x=self.x[key],
            y=self.y[key],
            ra=self.ra[key],
            dec=self.dec[key],
            fwhm=self.fwhm[key],
            ellipticity=self.ellipticity[key],
            flux=self.flux[key],
            peak=self.peak[key],
            snr=self.snr[key],
            sharpness=self.sharpness[key],
            nn_dist=self.nn_dist[key],
        )

    def to_table(self) -> "Table":
        """Return an :class:`astropy.table.Table` view of the columns."""
        from astropy.table import Table

        return Table(
            {
                "x": self.x,
                "y": self.y,
                "ra": self.ra,
                "dec": self.dec,
                "fwhm": self.fwhm,
                "ellipticity": self.ellipticity,
                "flux": self.flux,
                "peak": self.peak,
                "snr": self.snr,
                "sharpness": self.sharpness,
                "nn_dist": self.nn_dist,
            }
        )

    @classmethod
    def empty(cls) -> "SourceCatalog":
        """Return an empty catalogue."""
        z = np.empty(0, dtype=float)
        return cls(*(z.copy() for _ in range(11)))

    @classmethod
    def concat(cls, parts: "list[SourceCatalog]") -> "SourceCatalog":
        """Concatenate catalogues row-wise, or an empty one if all are empty."""
        nonempty = [part for part in parts if len(part) > 0]
        if not nonempty:
            return cls.empty()
        fields = (
            "x",
            "y",
            "ra",
            "dec",
            "fwhm",
            "ellipticity",
            "flux",
            "peak",
            "snr",
            "sharpness",
            "nn_dist",
        )
        return cls(
            **{
                name: np.concatenate([getattr(part, name) for part in nonempty])
                for name in fields
            }
        )


def build_catalog(
    data: np.ndarray,
    *,
    fwhm: float,
    nsigma: float = 5.0,
    err: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    pixel_to_world: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
    | None = None,
    aperture_radius: float | None = None,
    min_distance: float | None = None,
) -> SourceCatalog:
    """Detect compact sources in one frame and measure their shape.

    A Gaussian matched filter of the given ``fwhm`` is peak-found above
    ``nsigma`` robust sigmas; each peak is then characterised from the
    intensity-weighted moments in a fixed aperture (negatives clipped). See the
    module docstring for rationale and :class:`SourceCatalog` for the columns.

    Parameters
    ----------
    data : numpy.ndarray
        2-D image, assumed background-subtracted. Non-finite pixels are ignored.
    fwhm : float
        Expected point-source FWHM in pixels; sets the matched-filter width, the
        default aperture, and the default merge distance.
    nsigma : float, default 5.0
        Detection threshold in robust sigmas of the matched-filtered image.
    err : numpy.ndarray, optional
        Per-pixel 1-sigma error map (same shape as ``data``) for the aperture SNR.
        Pixels where it is non-finite or negative are excluded from detection.
    mask : numpy.ndarray, optional
        Boolean array where ``True`` marks pixels to exclude from detection and
        measurement, combined with the non-finite ``data`` / ``err`` exclusions.
        The pipeline DQ is deliberately *not* used here (it is too aggressive per
        single frame) -- pass only a mask you trust.
    pixel_to_world : callable, optional
        ``(x, y) -> (ra, dec)`` in degrees (arrays in, arrays out). When omitted,
        ``ra`` / ``dec`` are ``NaN``.
    aperture_radius : float, optional
        Radius (pixels) of the measurement aperture. Defaults to ``4 * sigma``
        (``~1.7 * fwhm``).
    min_distance : float, optional
        Minimum separation between detections; closer peaks keep only the
        brighter. Defaults to ``fwhm``.

    Returns
    -------
    SourceCatalog
        The detections, ordered by descending :attr:`~SourceCatalog.peak`.

    Raises
    ------
    ValueError
        If ``data`` is not 2-D, ``fwhm`` is not positive, or ``err`` / ``mask``
        shapes do not match ``data``.
    """
    image = np.asarray(data, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"data must be 2-D; got shape {image.shape}.")
    if not fwhm > 0:
        raise ValueError(f"fwhm must be positive; got {fwhm!r}.")
    if err is not None and err.shape != image.shape:
        raise ValueError(
            f"err shape {err.shape} does not match data shape {image.shape}."
        )
    if mask is not None and mask.shape != image.shape:
        raise ValueError(
            f"mask shape {mask.shape} does not match data shape {image.shape}."
        )

    sigma = fwhm / _FWHM_PER_SIGMA
    if aperture_radius is None:
        aperture_radius = 4.0 * sigma
    if min_distance is None:
        min_distance = fwhm

    bad = ~np.isfinite(image)
    if mask is not None:
        bad |= np.asarray(mask, dtype=bool)
    if err is not None:
        err_arr = np.asarray(err, dtype=float)
        bad |= ~np.isfinite(err_arr) | (err_arr < 0.0)
    good = ~bad
    if not good.any():
        return SourceCatalog.empty()

    clean = np.where(good, image, 0.0)
    sky_sigma = _robust_sigma(image[good])
    smoothed = ndi.gaussian_filter(clean, sigma)
    level = _detection_level(smoothed[good], nsigma)

    half = int(np.ceil(aperture_radius))
    coords = peak_local_max(
        smoothed,
        min_distance=max(1, int(np.ceil(min_distance))),
        threshold_abs=level,
        exclude_border=max(1, half),
    )
    if coords.shape[0] == 0:
        return SourceCatalog.empty()

    return _measure(
        clean, err, good, sky_sigma, aperture_radius, half, coords, pixel_to_world
    )


def _robust_sigma(values: np.ndarray) -> float:
    """Robust per-pixel sigma from the median absolute deviation."""
    median = float(np.median(values))
    return _MAD_TO_SIGMA * float(np.median(np.abs(values - median)))


def _detection_level(smoothed: np.ndarray, nsigma: float) -> float:
    """Absolute threshold ``nsigma`` robust sigmas above the smoothed median."""
    median = float(np.median(smoothed))
    mad = float(np.median(np.abs(smoothed - median)))
    return median + nsigma * _MAD_TO_SIGMA * mad


def _measure(
    clean: np.ndarray,
    err: np.ndarray | None,
    good: np.ndarray,
    sky_sigma: float,
    aperture_radius: float,
    half: int,
    coords: np.ndarray,
    pixel_to_world: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
    | None,
) -> SourceCatalog:
    """Measure weighted-moment shape and flux for every peak in ``coords``."""
    n_rows, n_cols = clean.shape
    ly, lx = np.mgrid[-half : half + 1, -half : half + 1]
    radius2 = ly**2 + lx**2
    within = radius2 <= aperture_radius**2

    xs, ys, fwhms, ells, fluxes, peaks, snrs, sharps = ([] for _ in range(8))
    for py, px in coords:
        r0, r1, c0, c1 = py - half, py + half + 1, px - half, px + half + 1
        if r0 < 0 or c0 < 0 or r1 > n_rows or c1 > n_cols:
            continue
        aperture = within & good[r0:r1, c0:c1]
        weight = np.where(aperture, np.clip(clean[r0:r1, c0:c1], 0.0, None), 0.0)

        m = moments(weight)
        flux = float(m[0, 0])
        if flux <= 0:
            continue
        yc = m[1, 0] / flux
        xc = m[0, 1] / flux
        mu = moments_central(weight, center=(yc, xc))
        var_y = mu[2, 0] / flux
        var_x = mu[0, 2] / flux
        covar = mu[1, 1] / flux
        if var_x <= 0 or var_y <= 0:
            continue

        fwhm_i = _FWHM_PER_SIGMA * np.sqrt(0.5 * (var_x + var_y))
        root = np.sqrt((var_x - var_y) ** 2 + 4.0 * covar**2)
        lam1 = 0.5 * (var_x + var_y + root)
        lam2 = 0.5 * (var_x + var_y - root)
        ellipticity = 1.0 - np.sqrt(lam2 / lam1) if lam1 > 0 else 0.0

        peak = float(clean[py, px])
        if err is not None:
            evar = float(np.sum(np.where(aperture, err[r0:r1, c0:c1], 0.0) ** 2))
            snr = flux / np.sqrt(evar) if evar > 0 else np.inf
        else:
            snr = peak / sky_sigma if sky_sigma > 0 else np.inf

        xs.append(c0 + xc)
        ys.append(r0 + yc)
        fwhms.append(float(fwhm_i))
        ells.append(float(ellipticity))
        fluxes.append(flux)
        peaks.append(peak)
        snrs.append(float(snr))
        sharps.append(_sharpness(clean, good, py, px, peak))

    if not xs:
        return SourceCatalog.empty()

    order = np.argsort(peaks)[::-1]
    x = np.asarray(xs)[order]
    y = np.asarray(ys)[order]
    if pixel_to_world is not None:
        ra, dec = pixel_to_world(x, y)
        ra = np.asarray(ra, dtype=float)
        dec = np.asarray(dec, dtype=float)
    else:
        ra = np.full(x.size, np.nan)
        dec = np.full(x.size, np.nan)
    return SourceCatalog(
        x=x,
        y=y,
        ra=ra,
        dec=dec,
        fwhm=np.asarray(fwhms)[order],
        ellipticity=np.asarray(ells)[order],
        flux=np.asarray(fluxes)[order],
        peak=np.asarray(peaks)[order],
        snr=np.asarray(snrs)[order],
        sharpness=np.asarray(sharps)[order],
        nn_dist=_nearest_neighbour(x, y),
    )


def _sharpness(
    clean: np.ndarray, good: np.ndarray, py: int, px: int, peak: float
) -> float:
    """Peak pixel divided by the mean of its eight valid neighbours."""
    block = clean[py - 1 : py + 2, px - 1 : px + 2]
    mask = good[py - 1 : py + 2, px - 1 : px + 2].copy()
    mask[1, 1] = False
    neighbours = block[mask]
    if neighbours.size == 0:
        return np.inf
    mean_neighbour = float(neighbours.mean())
    return peak / mean_neighbour if mean_neighbour > 0 else np.inf


def _nearest_neighbour(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Distance from each source to the nearest other (``inf`` if alone)."""
    if x.size < 2:
        return np.full(x.size, np.inf)
    points = np.column_stack([x, y])
    distances, _ = cKDTree(points).query(points, k=2)
    return distances[:, 1]
