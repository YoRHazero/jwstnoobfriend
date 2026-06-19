"""Astrometric inputs for stage-3 alignment: clean reference + clean image stars.

``jwst`` ``TweakRegStep`` aligns a mosaic in two phases -- relative (image-to-
image, from source catalogs detected in the frames) and absolute (the merged
catalog tied to an external reference, by default ``"GAIADR3"``). In small,
sparse fields the absolute tie is the weak link: only a handful of GAIA stars,
some of them blends or extended sources, and (since GAIA is at epoch J2016)
several mas of proper-motion drift by the observation epoch. This module
addresses both ends of that tie with pure, navigation-/jwst-agnostic helpers the
generated stage-3 script feeds to ``TweakRegStep``:

- **reference side** -- :func:`query_gaia` pulls GAIA DR3 over the field and
  :func:`clean_gaia` proper-motion-propagates it to the observation epoch and
  drops blends / poor-astrometry sources. :class:`GaiaReference` packages the
  two as a :class:`ReferenceProvider`; :func:`build_reference` runs one or more
  providers (the extension point for densifying with a deeper, GAIA-registered
  survey catalog in a crowded-but-shallow field) into one ``abs_refcat`` table.
- **image side** -- :func:`select_point_sources` turns the mechanism-only
  :class:`~noobfriend.core.imgutils.SourceCatalog` (no notion of "star") into a
  clean, isolated, PSF-like selection, and :func:`to_tweakreg_catalog` writes it
  in the ``x`` / ``y`` form a ``catfile`` entry expects -- so the relative phase
  matches on unblended centroids instead of star-on-galaxy blends.

The GAIA query is the only impure step; :mod:`astroquery` is imported lazily
inside :func:`query_gaia` so importing this module never requires it.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from astropy.table import Table

    from noobfriend.core.imgutils import SourceCatalog

#: GAIA DR3 columns pulled for quality filtering and PM propagation.
_GAIA_COLUMNS: tuple[str, ...] = (
    "ra",
    "dec",
    "pmra",
    "pmdec",
    "ref_epoch",
    "phot_g_mean_mag",
    "ruwe",
    "astrometric_excess_noise",
    "classprob_dsc_combmod_galaxy",
)


def query_gaia(ra: float, dec: float, radius: float, *, row_limit: int = -1) -> "Table":
    """Query GAIA DR3 over a cone (lazy :mod:`astroquery` import).

    Parameters
    ----------
    ra, dec : float
        Cone centre in degrees.
    radius : float
        Cone radius in degrees.
    row_limit : int, optional
        Maximum rows; ``-1`` (default) for no limit.

    Returns
    -------
    astropy.table.Table
        The columns in :data:`_GAIA_COLUMNS` for sources in the cone.
    """
    from astroquery.gaia import Gaia

    Gaia.ROW_LIMIT = row_limit
    adql = (
        f"SELECT {', '.join(_GAIA_COLUMNS)} FROM gaiadr3.gaia_source "
        f"WHERE 1=CONTAINS(POINT('ICRS', ra, dec), "
        f"CIRCLE('ICRS', {ra}, {dec}, {radius}))"
    )
    return Gaia.launch_job_async(adql).get_results()


def _col(table: "Table", name: str, fill: float) -> np.ndarray:
    """Return a masked-safe float column, missing entries replaced by ``fill``."""
    column = table[name]
    values = np.asarray(
        getattr(column, "filled", lambda v: column)(np.nan), dtype=float
    )
    return np.where(np.isfinite(values), values, fill)


def clean_gaia(
    table: "Table",
    obs_epoch: float,
    *,
    max_ruwe: float = 1.4,
    max_excess_noise: float = 1.0,
    max_pgal: float = 0.3,
) -> "Table":
    """Proper-motion-propagate GAIA to the epoch and keep clean point sources.

    Right ascension and declination are advanced from each source's
    ``ref_epoch`` to ``obs_epoch`` using its proper motion (missing PM treated as
    zero), then sources are filtered to ``ruwe < max_ruwe``,
    ``astrometric_excess_noise < max_excess_noise`` and
    ``classprob_dsc_combmod_galaxy < max_pgal`` -- removing the blends, binaries
    and galaxies that bias an absolute tie. Missing ``ruwe`` / excess-noise are
    treated as failing, missing galaxy-probability as passing.

    Parameters
    ----------
    table : astropy.table.Table
        A GAIA table with the :data:`_GAIA_COLUMNS` columns.
    obs_epoch : float
        Observation epoch as a decimal year (PM-propagation target).
    max_ruwe, max_excess_noise, max_pgal : float, optional
        Quality thresholds; see above.

    Returns
    -------
    astropy.table.Table
        A copy with ``ra`` / ``dec`` propagated, keeping only clean sources.
    """
    out = table.copy()
    dt = obs_epoch - _col(out, "ref_epoch", 2016.0)
    dec = _col(out, "dec", 0.0)
    pmra = _col(out, "pmra", 0.0)
    pmdec = _col(out, "pmdec", 0.0)
    cos_dec = np.cos(np.deg2rad(dec))
    out["ra"] = _col(out, "ra", 0.0) + (pmra / 3.6e6) * dt / cos_dec
    out["dec"] = dec + (pmdec / 3.6e6) * dt

    keep = (
        (_col(out, "ruwe", np.inf) < max_ruwe)
        & (_col(out, "astrometric_excess_noise", np.inf) < max_excess_noise)
        & (_col(out, "classprob_dsc_combmod_galaxy", 0.0) < max_pgal)
    )
    return out[keep]


def select_point_sources(
    catalog: "SourceCatalog",
    *,
    max_ellipticity: float = 0.15,
    fwhm_range: tuple[float, float] = (1.2, 3.5),
    min_snr: float = 20.0,
    min_isolation: float = 8.0,
) -> "SourceCatalog":
    """Select clean, isolated, PSF-like sources for astrometric matching.

    Applies cuts on the mechanism-only columns of a
    :class:`~noobfriend.core.imgutils.SourceCatalog`: round
    (``ellipticity < max_ellipticity``), PSF-sized (``fwhm`` in ``fwhm_range``),
    well-detected (``snr >= min_snr``) and isolated
    (``nn_dist >= min_isolation``). The isolation and roundness cuts are what
    keep star-on-galaxy blends -- the bias source -- out of the relative-alignment
    catalog. Saturated cores are best excluded upstream via the detection
    ``mask`` (e.g. the saturation DQ bit), not here.

    Parameters
    ----------
    catalog : SourceCatalog
        Detections from :func:`noobfriend.core.imgutils.build_catalog`.
    max_ellipticity : float, optional
        Maximum ``ellipticity`` (0 = round), by default 0.15.
    fwhm_range : tuple of float, optional
        Inclusive ``(min, max)`` FWHM in pixels, by default ``(1.2, 3.5)``.
    min_snr : float, optional
        Minimum aperture SNR, by default 20.
    min_isolation : float, optional
        Minimum nearest-neighbour distance in pixels, by default 8.

    Returns
    -------
    SourceCatalog
        The selected sub-catalogue.
    """
    keep = (
        (catalog.ellipticity < max_ellipticity)
        & (catalog.snr >= min_snr)
        & (catalog.nn_dist >= min_isolation)
        & (catalog.fwhm >= fwhm_range[0])
        & (catalog.fwhm <= fwhm_range[1])
    )
    return catalog[keep]


def to_tweakreg_catalog(catalog: "SourceCatalog") -> "Table":
    """Return an ``x`` / ``y`` table for a ``TweakRegStep`` ``catfile`` entry.

    Parameters
    ----------
    catalog : SourceCatalog
        The (already selected) image sources.

    Returns
    -------
    astropy.table.Table
        Columns ``x`` and ``y`` in the catalogue's pixel-index convention.
    """
    from astropy.table import Table

    return Table({"x": np.asarray(catalog.x), "y": np.asarray(catalog.y)})


class ReferenceProvider(Protocol):
    """A source of absolute-reference positions over a field.

    Implementations return an :class:`~astropy.table.Table` with at least ``RA``
    and ``DEC`` columns (degrees) for the given cone. :class:`GaiaReference` is
    the default; a densifying provider (a deeper survey catalog first registered
    to GAIA) is the intended extension for crowded-but-shallow fields.
    """

    def __call__(
        self, ra: float, dec: float, radius: float, obs_epoch: float
    ) -> "Table": ...


@dataclass(frozen=True)
class GaiaReference:
    """GAIA DR3 reference provider: query, PM-propagate, quality-filter.

    Attributes
    ----------
    max_ruwe, max_excess_noise, max_pgal : float
        Quality thresholds forwarded to :func:`clean_gaia`.
    """

    max_ruwe: float = 1.4
    max_excess_noise: float = 1.0
    max_pgal: float = 0.3

    def __call__(
        self, ra: float, dec: float, radius: float, obs_epoch: float
    ) -> "Table":
        """Return the cleaned GAIA reference as an ``RA`` / ``DEC`` table."""
        cleaned = clean_gaia(
            query_gaia(ra, dec, radius),
            obs_epoch,
            max_ruwe=self.max_ruwe,
            max_excess_noise=self.max_excess_noise,
            max_pgal=self.max_pgal,
        )
        return _as_refcat(cleaned)


def _as_refcat(table: "Table") -> "Table":
    """Normalise a provider table to just ``RA`` / ``DEC`` (degrees)."""
    from astropy.table import Table

    return Table({"RA": np.asarray(table["ra"]), "DEC": np.asarray(table["dec"])})


def within_footprints(
    ra: np.ndarray, dec: np.ndarray, footprints: "list[np.ndarray]"
) -> np.ndarray:
    """Boolean mask of ``(ra, dec)`` points inside the union of footprint quads.

    A TAP cone query returns sources over a circle roughly twice the area of the
    real (gappy, multi-detector) field, so most of them never land on a pixel.
    This keeps only the points inside at least one footprint polygon, via a
    half-plane (cross-product) test on the boundary-ordered corners -- valid for
    the convex detector quads, away from the RA 0/360 seam and the poles.

    Parameters
    ----------
    ra, dec : numpy.ndarray
        Point coordinates in degrees.
    footprints : list of numpy.ndarray
        One ``(K, 2)`` array of boundary-ordered ``(ra, dec)`` corners per frame.

    Returns
    -------
    numpy.ndarray
        Boolean mask, ``True`` where the point falls inside any footprint.
    """
    pts = np.column_stack([np.asarray(ra, float), np.asarray(dec, float)])
    inside = np.zeros(len(pts), dtype=bool)
    for footprint in footprints:
        verts = np.asarray(footprint, dtype=float)
        crosses = []
        for i in range(len(verts)):
            a, b = verts[i], verts[(i + 1) % len(verts)]
            edge = b - a
            crosses.append(edge[0] * (pts[:, 1] - a[1]) - edge[1] * (pts[:, 0] - a[0]))
        signs = np.array(crosses)
        inside |= np.all(signs >= -1e-12, axis=0) | np.all(signs <= 1e-12, axis=0)
    return inside


def build_reference(
    ra: float,
    dec: float,
    radius: float,
    obs_epoch: float,
    *,
    providers: tuple[ReferenceProvider, ...] = (GaiaReference(),),
    footprints: "list[np.ndarray] | None" = None,
) -> "Table":
    """Assemble the absolute-reference catalog from one or more providers.

    Each provider's ``RA`` / ``DEC`` rows are stacked into a single
    ``abs_refcat`` table. With the single default provider this is just cleaned
    GAIA; supplying additional providers (a deeper, GAIA-registered catalog) is
    how a sparse field is densified while staying on the GAIA frame. When
    ``footprints`` is given the result is trimmed to sources actually inside the
    coverage (the cone over-queries by ~2x), so the count reflects the real
    on-pixel anchors rather than the whole circle.

    Parameters
    ----------
    ra, dec, radius : float
        Field cone (degrees) used for the provider query.
    obs_epoch : float
        Observation epoch (decimal year).
    providers : tuple of ReferenceProvider, optional
        Providers to query, by default a single :class:`GaiaReference`.
    footprints : list of numpy.ndarray, optional
        Per-frame ``(ra, dec)`` corner arrays; when given, the reference is
        filtered to sources inside the union of footprints (see
        :func:`within_footprints`).

    Returns
    -------
    astropy.table.Table
        The merged (and optionally footprint-trimmed) reference with ``RA`` /
        ``DEC`` columns.
    """
    from astropy.table import vstack

    parts = [p(ra, dec, radius, obs_epoch) for p in providers]
    out = parts[0] if len(parts) == 1 else vstack(parts)
    if footprints is not None:
        out = out[
            within_footprints(np.asarray(out["RA"]), np.asarray(out["DEC"]), footprints)
        ]
    return out
