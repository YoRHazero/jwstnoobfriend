"""The :class:`BoxGrism`: a source's WFSS extraction across a box's exposures.

Obtained only through
:meth:`BoxExtract.grism <noobfriend.navigation.noobox.extract._core.BoxExtract.grism>`
(``box.extract.grism(ra, dec, ...)``); not constructed directly. ``BoxGrism``
drives :mod:`noobfriend.extraction.grism`: it finds the exposures whose order-1
trace covers a sky position, resolves the common ``(spatial, wavelength)`` grid,
and -- on demand -- rectifies and combines them into 2-D spectra. It reads each
exposure's pixels through the box's shared cache (the ``load`` seam the
extraction layer leaves open) and keys the per-exposure spectra by their
:class:`~noobfriend.navigation.NooBook` so provenance travels with the data.

The data products (:class:`~noobfriend.extraction.grism.GrismSpectrum`) are
navigation-free and live in :mod:`noobfriend.extraction`; ``BoxGrism`` is only
the navigation-side driver -- the grism counterpart of
``box.extract.sources`` returning a ``SourceExtractor``, not of ``BoxCutout``.

The coverage gate
-----------------
``find_coverage`` reads each candidate's grism WCS (a file fetch), so testing
every grism frame in a large box is expensive. With ``pre_gate=True`` a cheap
resident-footprint filter runs first: per ``(module, dispersion)`` group the
footprint-nearest exposure is the *seed*, whose WCS is read once to get the
trace's angular length and sign; every other exposure's trace is then drawn as a
short sky segment using its *own* footprint corners for the dispersion direction
(so different position angles are handled) and intersected with its footprint.
``find_coverage`` then confirms only the survivors.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from noobase.axis import Grid

    from noobfriend.extraction.grism import (
        FrameCoverage,
        GrismExtractor,
        GrismSpectrum,
    )
    from noobfriend.navigation.noobook._core import NooBook
    from noobfriend.navigation.noobox._core import NooBox

_Point = tuple[float, float]


# -- grism identity / grouping ------------------------------------------------


def _is_grism(book: NooBook) -> bool:
    """Whether ``book`` is a NIRCam WFSS exposure, by its resident pupil."""
    return book.pupil is not None and book.pupil.upper().startswith("GRISM")


def _module_of(detector: str | None) -> str | None:
    """Return the NIRCam module (``"a"``/``"b"``) from a detector name."""
    if detector is None:
        return None
    name = detector.lower()
    if name.startswith("nrca"):
        return "a"
    if name.startswith("nrcb"):
        return "b"
    return None


def _dispersion_of(pupil: str | None) -> str | None:
    """Return the dispersion axis (``"row"``/``"column"``) from the pupil."""
    if pupil is None:
        return None
    name = pupil.upper()
    if "GRISMR" in name:
        return "row"
    if "GRISMC" in name:
        return "column"
    return None


def _default_group(book: NooBook) -> tuple[str | None, str | None]:
    """Default :meth:`BoxGrism.combine` key: ``(module, dispersion)``."""
    return (_module_of(book.detector), _dispersion_of(book.pupil))


# -- flat-sky geometry for the footprint gate ---------------------------------


def _projector(ra0: float, dec0: float) -> Callable[[float, float], _Point]:
    """Return a local flat-sky projection about ``(ra0, dec0)`` (degrees).

    Maps ``(ra, dec)`` to ``((ra - ra0) cos dec0, dec - dec0)`` so small-field
    segment/polygon geometry is Euclidean and the source sits at the origin.
    """
    cos_dec = math.cos(math.radians(dec0))

    def project(ra: float, dec: float) -> _Point:
        return ((float(ra) - ra0) * cos_dec, float(dec) - dec0)

    return project


def _corners_xy(
    footprint: Any, project: Callable[[float, float], _Point]
) -> list[_Point]:
    """Project a footprint's world corners to local flat-sky points."""
    return [project(ra, dec) for ra, dec in footprint.corners]


def _dispersion_unit(corners_xy: list[_Point], dispersion: str) -> _Point:
    """Return the dispersion-axis unit vector from projected footprint corners.

    Corner order is ``(0,0), (nx,0), (nx,ny), (0,ny)`` in detector pixels, so the
    detector-x (row) direction is ``corner1 - corner0`` and detector-y (column)
    is ``corner3 - corner0``.
    """
    c0, c1, _c2, c3 = corners_xy
    if dispersion == "row":
        vx, vy = c1[0] - c0[0], c1[1] - c0[1]
    else:
        vx, vy = c3[0] - c0[0], c3[1] - c0[1]
    norm = math.hypot(vx, vy)
    return (vx / norm, vy / norm) if norm > 0 else (1.0, 0.0)


def _point_in_polygon(point: _Point, polygon: list[_Point]) -> bool:
    """Even-odd point-in-polygon test in the flat-sky plane."""
    x, y = point
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            x_cross = xi + (y - yi) / (yj - yi) * (xj - xi)
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def _ccw(a: _Point, b: _Point, c: _Point) -> float:
    """Twice the signed area of triangle ``abc`` (orientation test)."""
    return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_cross(p1: _Point, p2: _Point, p3: _Point, p4: _Point) -> bool:
    """Whether open segments ``p1p2`` and ``p3p4`` straddle each other."""
    d1, d2 = _ccw(p3, p4, p1), _ccw(p3, p4, p2)
    d3, d4 = _ccw(p1, p2, p3), _ccw(p1, p2, p4)
    return (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0)


def _segment_hits_polygon(p0: _Point, p1: _Point, polygon: list[_Point]) -> bool:
    """Whether segment ``p0p1`` touches ``polygon`` (endpoint inside or crossing)."""
    if _point_in_polygon(p0, polygon) or _point_in_polygon(p1, polygon):
        return True
    n = len(polygon)
    return any(
        _segments_cross(p0, p1, polygon[i], polygon[(i + 1) % n]) for i in range(n)
    )


def _footprint_distance(
    footprint: Any, project: Callable[[float, float], _Point]
) -> float:
    """Distance from the source (origin) to a footprint's projected centroid."""
    pts = _corners_xy(footprint, project)
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return math.hypot(cx, cy)


def _seed_extent(
    seed: NooBook,
    ra: float,
    dec: float,
    dispersion: str,
    project: Callable[[float, float], _Point],
) -> tuple[float, float]:
    """Along-dispersion sky extent of the trace, from the seed exposure.

    Reads the seed's grism WCS once, traces order 1 over its wavelength domain,
    maps the two endpoints back to sky through the direct astrometry, and projects
    them onto the seed's dispersion axis -- giving the signed angular offsets
    ``(t_lo, t_hi)`` of the trace ends from the source, reused (with each frame's
    own direction) across the group.
    """
    from noobfriend.extraction._wcs import (
        grism_trace_transform,
        world_detector_transforms,
    )
    from noobfriend.extraction.grism import read_wavelength_domain

    wcs = seed.wcs
    lo, hi = read_wavelength_domain(wcs, ra, dec, order=1)
    world_to_detector, detector_to_world = world_detector_transforms(wcs)
    trace = grism_trace_transform(wcs)
    x0, y0 = world_to_detector(ra, dec)

    unit = _dispersion_unit(_corners_xy(seed.footprint, project), dispersion)
    extents: list[float] = []
    for wavelength in (lo, hi):
        xg, yg = trace(float(x0), float(y0), float(wavelength))
        sky = detector_to_world(float(xg), float(yg))
        px, py = project(float(sky[0]), float(sky[1]))
        extents.append(px * unit[0] + py * unit[1])
    return extents[0], extents[1]


def _pre_gate(
    grism_books: list[NooBook], ra: float, dec: float, *, margin: float = 0.1
) -> list[NooBook]:
    """Resident-footprint candidate filter for grism coverage (see module doc).

    Returns the subset of ``grism_books`` whose trace plausibly lands on the
    detector, reading only one seed WCS per ``(module, dispersion)`` group. The
    segment is grown by ``margin`` (fraction of its length) so the gate is
    generous -- ``find_coverage`` confirms the survivors exactly.
    """
    project = _projector(ra, dec)
    groups: dict[tuple[str | None, str | None], list[NooBook]] = {}
    for book in grism_books:
        groups.setdefault(_default_group(book), []).append(book)

    candidates: list[NooBook] = []
    for (_module, dispersion), members in groups.items():
        if dispersion is None:
            candidates.extend(members)  # cannot orient the trace; keep them all
            continue
        seed = min(members, key=lambda b: _footprint_distance(b.footprint, project))
        t_lo, t_hi = _seed_extent(seed, ra, dec, dispersion, project)
        pad = margin * abs(t_hi - t_lo)
        low, high = min(t_lo, t_hi) - pad, max(t_lo, t_hi) + pad
        for book in members:
            unit = _dispersion_unit(_corners_xy(book.footprint, project), dispersion)
            p0 = (low * unit[0], low * unit[1])
            p1 = (high * unit[0], high * unit[1])
            if _segment_hits_polygon(p0, p1, _corners_xy(book.footprint, project)):
                candidates.append(book)
    return candidates


def _safe_coverage(
    ra: float, dec: float, meta: Any, book: NooBook, min_fraction: float
) -> Any:
    """Coverage for one frame, reporting an unevaluable frame as not covered.

    ``find_coverage`` resolves the wavelength domain at the source position and
    traces it; for a source that falls well off this detector the dispersion
    model can raise (e.g. a non-positive wavelength). Such a frame does not cover
    the source, so the failure is reported as ``covered=False`` rather than
    propagated -- which also lets ``pre_gate=False`` survive the off-array frames
    the footprint gate would otherwise have removed.
    """
    from noobfriend.extraction._wcs import world_detector_transforms
    from noobfriend.extraction.grism import FrameCoverage, find_coverage

    try:
        return find_coverage(ra, dec, [meta], min_fraction=min_fraction)[0]
    except Exception:
        world_to_detector, _ = world_detector_transforms(meta.wcs)
        x0, y0 = world_to_detector(ra, dec)
        return FrameCoverage(
            meta=meta,
            covered=False,
            wavelength_range=None,
            dispersion=_dispersion_of(book.pupil) or "row",
            source_xy=(float(x0), float(y0)),
        )


# -- the handle ---------------------------------------------------------------


def _build(
    box: NooBox,
    ra: float,
    dec: float,
    *,
    spatial_half: int,
    wavelength: Grid | None = None,
    oversample: int = 1,
    min_fraction: float = 0.0,
    pre_gate: bool = True,
    skip_missing_wcs: bool = True,
    probe: bool = True,
    progress: bool = True,
) -> BoxGrism:
    """Build a :class:`BoxGrism` (see :meth:`BoxExtract.grism`)."""
    from noobfriend.extraction.grism import FrameMeta, GrismExtractor

    grism_books: list[NooBook] = []
    for original in box:
        book = original.probe() if probe and not original.is_probed else original
        if not _is_grism(book):
            continue
        if book.footprint is None:
            if skip_missing_wcs:
                continue
            raise ValueError(
                f"{book.id} is a grism frame with no footprint "
                "(assign a WCS or probe the book)."
            )
        grism_books.append(book)

    if not grism_books:
        raise ValueError("no grism (pupil=GRISM*) frames in the box.")

    candidates = _pre_gate(grism_books, ra, dec) if pre_gate else list(grism_books)
    if not candidates:
        raise ValueError(f"no grism frame's trace reaches (ra, dec) = ({ra}, {dec}).")

    if progress:
        from noobfriend.core.display import track

        iterable = track(candidates, "Grism coverage")
    else:
        iterable = candidates
    coverage = [
        _safe_coverage(
            ra,
            dec,
            FrameMeta(id=book.id, wcs=book.wcs, shape=tuple(book.shape[-2:])),
            book,
            min_fraction,
        )
        for book in iterable
    ]
    covered = [cov for cov in coverage if cov.covered]
    if not covered:
        raise ValueError(
            f"no grism frame covers (ra, dec) = ({ra}, {dec}) "
            f"(checked {len(coverage)} candidate frame(s))."
        )

    extractor = GrismExtractor.from_world(
        ra,
        dec,
        covered,
        spatial_half=spatial_half,
        oversample=oversample,
        wavelength=wavelength,
    )
    books_by_id = {book.id: book for book in candidates}
    return BoxGrism(
        ra=ra,
        dec=dec,
        extractor=extractor,
        coverage=coverage,
        candidates=candidates,
        books_by_id=books_by_id,
    )


class BoxGrism:
    """A source's grism extraction, driven across a box's WFSS exposures.

    Construct it with ``box.extract.grism(...)`` rather than directly. On
    construction the coverage is resolved and the common output grid is set, all
    from WCS + resident footprints -- no pixels are read. The per-exposure 2-D
    spectra (:attr:`spectra`) and their combination (:meth:`combine`) read pixels
    lazily, through the box's shared cache, on first access.

    Parameters
    ----------
    ra, dec : float
        Source world coordinates in degrees.
    extractor : GrismExtractor
        The resolved extraction (common ``(spatial, wavelength)`` grid + covered
        exposures).
    coverage : list of FrameCoverage
        The full :func:`~noobfriend.extraction.grism.find_coverage` result over
        the candidate frames (covered and not).
    candidates : list of NooBook
        The footprint-gate survivors that coverage was tested on.
    books_by_id : dict of {str: NooBook}
        Lookup from book id to NooBook, for rectifying covered exposures.
    """

    def __init__(
        self,
        *,
        ra: float,
        dec: float,
        extractor: GrismExtractor,
        coverage: list[FrameCoverage],
        candidates: list[NooBook],
        books_by_id: dict[str, NooBook],
    ) -> None:
        """See the class docstring; built by ``box.extract.grism(...)``."""
        self._ra = ra
        self._dec = dec
        self._extractor = extractor
        self._coverage = coverage
        self._candidates = candidates
        self._books_by_id = books_by_id
        self._spectra: dict[NooBook, GrismSpectrum] | None = None

    # -- pixel-free inspection ------------------------------------------------

    @property
    def ra(self) -> float:
        """Right ascension of the source, in degrees."""
        return self._ra

    @property
    def dec(self) -> float:
        """Declination of the source, in degrees."""
        return self._dec

    @property
    def candidates(self) -> list[NooBook]:
        """The footprint-gate survivor exposures (pre-``find_coverage``)."""
        return list(self._candidates)

    @property
    def coverage(self) -> list[FrameCoverage]:
        """The full coverage result over the candidates (covered and not)."""
        return list(self._coverage)

    @property
    def books(self) -> list[NooBook]:
        """The covering exposures, in coverage order."""
        return [self._books_by_id[cov.meta.id] for cov in self._extractor.coverage]

    @property
    def wavelength(self) -> Grid:
        """The resolved common wavelength axis (microns); set at construction."""
        return self._extractor.wavelength

    # -- spectra (read pixels lazily) -----------------------------------------

    def _load(self, book_id: str) -> tuple[Any, Any]:
        """Fetch one exposure's ``(data, error)`` through the shared cache."""
        book = self._books_by_id[book_id]
        error = book.err
        if error is None:
            raise ValueError(
                f"grism book {book.id} has no ERR array to rectify; "
                "WFSS rectification needs an error plane."
            )
        return book.data, error

    @property
    def spectra(self) -> dict[NooBook, GrismSpectrum]:
        """Per-exposure rectified 2-D spectra, keyed by their NooBook.

        Each value is a single-exposure
        :class:`~noobfriend.extraction.grism.GrismSpectrum` (``n_frames == 1``) on
        the common grid -- the per-exposure spectra to inspect (contamination, bad
        pixels) before combining. Accessing this **reads every covering
        exposure's pixels** through the shared cache on first call; the result is
        then cached on the handle.

        Returns
        -------
        dict of {NooBook: GrismSpectrum}
        """
        if self._spectra is None:
            by_id = self._extractor.rectify(self._load)
            self._spectra = {self._books_by_id[k]: v for k, v in by_id.items()}
        return self._spectra

    def combine(self, group_by: Any = _default_group) -> Any:
        """Co-add the per-exposure spectra, grouped by ``group_by``.

        Reuses the cached :attr:`spectra` (no re-read) and inverse-variance stacks
        them on the common grid. ``group_by`` is a key *selector* (not values):

        - a callable ``book -> key`` (default: ``(module, dispersion)``), or a
          :class:`~noobfriend.navigation.NooBook` attribute name -- returns one
          :class:`~noobfriend.extraction.grism.GrismSpectrum` per distinct key, as
          a list;
        - ``None`` -- ignore grouping and co-add **all** covering exposures into a
          single :class:`~noobfriend.extraction.grism.GrismSpectrum`.

        Parameters
        ----------
        group_by : str or callable or None, optional
            The grouping-key selector; see above.

        Returns
        -------
        list of GrismSpectrum or GrismSpectrum
            A list (one per group) for a grouping selector, or a single combined
            spectrum when ``group_by`` is ``None``.
        """
        spectra = self.spectra
        if group_by is None:
            relabelled = {
                book.id: replace(s, group=None) for book, s in spectra.items()
            }
            return self._extractor.combine(relabelled)[0]
        keyfn = (
            (lambda book: getattr(book, group_by))
            if isinstance(group_by, str)
            else group_by
        )
        relabelled = {
            book.id: replace(s, group=keyfn(book)) for book, s in spectra.items()
        }
        return self._extractor.combine(relabelled)

    def plot(self, **kwargs: Any) -> Any:
        """Plot the all-combined 2-D spectrum (and its 1-D collapse).

        Co-adds every covering exposure (``combine(group_by=None)``) and hands it
        to :func:`noobfriend.core.display.plot.plot_spectrum2d`; styling keywords
        pass straight through. Reads pixels (via :attr:`spectra`).

        Returns
        -------
        matplotlib.figure.Figure
        """
        from noobfriend.core.display.plot import plot_spectrum2d

        spectrum = self.combine(group_by=None)
        wavelength = getattr(spectrum.wavelength, "values", spectrum.wavelength)
        spatial = getattr(spectrum.spatial_offset, "values", spectrum.spatial_offset)
        return plot_spectrum2d(
            wavelength,
            spectrum.flux_2d,
            spatial=spatial,
            error_2d=spectrum.error_2d,
            **kwargs,
        )

    def __len__(self) -> int:
        """Return the number of covering exposures."""
        return len(self._extractor.coverage)

    def __repr__(self) -> str:
        """Return a one-line summary: covering count and wavelength range."""
        values = getattr(self.wavelength, "values", self.wavelength)
        lo, hi = float(values[0]), float(values[-1])
        return f"BoxGrism({len(self)} covering, lambda {lo:.2f}-{hi:.2f} um)"
