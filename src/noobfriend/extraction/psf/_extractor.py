"""Accumulate matched point-source detections across frames for PSF building.

:class:`SourceExtractor` is the stateful wrapper of the PSF pipeline. Each
calibrated frame is handed to :meth:`~SourceExtractor.add_from_frame`, which
detects sources (:func:`noobfriend.core.imgutils.build_catalog`), **matches them
by sky position** to the running catalogue, drops sources that are neither
point-like here nor already known, and caches a fixed-size cutout per kept
detection. The accumulated catalogue is one flat table: one row per detection,
with an ``index`` column that ties the same physical source together across
frames and bands (a star is detected once per exposure and per filter, so each
``index`` gathers many rows). Selection, the interactive panel, and the build all
consume that table; this class only finds, matches, and cuts.

It takes plain arrays plus a WCS object (which it normalises to a
``pixel_to_world`` callable for ``build_catalog``), never a
:class:`~noobfriend.navigation.NooBox`, so navigation drives it but it never
depends on navigation. Cutouts are stored at one ``cutout_size`` as ``float32``
in memory; core / wing stamps are sliced from them at build time.
"""

from collections.abc import Callable
from fnmatch import fnmatchcase
from pathlib import Path
from re import Pattern

import numpy as np

from noobfriend.core.imgutils import SourceCatalog, build_catalog
from noobfriend.extraction.psf._build import (
    CorePsf,
    EffectivePsf,
    build_core,
    build_wings,
)
from noobfriend.extraction.psf._detections import _Detections
from noobfriend.extraction.psf._store import Cutout, CutoutStore

#: On-disk layout version for :meth:`SourceExtractor.save` / ``load``.
_SAVE_FORMAT_VERSION: int = 1


def _group_aggregate(
    index: np.ndarray, values: np.ndarray, func: Callable[[np.ndarray], float]
) -> dict[int, float]:
    """Return ``{source_index: func(that source's detection values)}``."""
    if index.size == 0:
        return {}
    order = np.argsort(index, kind="stable")
    sorted_index = index[order]
    bounds = np.flatnonzero(np.diff(sorted_index)) + 1
    groups = np.split(values[order], bounds)
    ids = sorted_index[np.concatenate(([0], bounds))]
    return {int(gid): float(func(group)) for gid, group in zip(ids, groups)}


def _match_labels(labels: list[object] | np.ndarray, matcher: object) -> np.ndarray:
    """Return which labels match an exact value, glob string, or compiled regex."""
    values = np.asarray(labels, dtype=object)
    if isinstance(matcher, Pattern):
        return np.array(
            [
                label is not None and matcher.fullmatch(str(label)) is not None
                for label in values
            ],
            dtype=bool,
        )
    if isinstance(matcher, str):
        return np.array(
            [
                label is not None and fnmatchcase(str(label), matcher)
                for label in values
            ],
            dtype=bool,
        )
    return values == matcher


def _pixel_to_world(
    wcs: object,
) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Adapt a WCS to a ``(x, y) -> (ra_deg, dec_deg)`` callable.

    Prefers the package-wide :class:`~noobfriend.core.wcs.TransformWCS` contract
    -- ``get_transform("detector", "world")`` -- which is how every other WCS
    consumer in noobfriend evaluates pixel -> world (see
    :mod:`noobfriend.extraction.cutout`, which documents this as the canonical
    path over the slower APE-14 ``pixel_to_world_values`` wrapper). That one
    branch covers everything the package treats as a WCS: a compiled
    :class:`~noobfriend.core.wcs.NoobWCS` (what ``NooBook.wcs`` now returns), a
    :class:`gwcs.WCS`, and the astropy / grizli adapters that wrap a plain FITS
    WCS as a ``TransformWCS``. The transform returns ``(ra, dec)`` in degrees
    (first frame -> last); any extra WFSS world outputs (wavelength / order) are
    ignored.

    A bare :mod:`astropy.wcs` / ``gwcs`` object passed directly (no
    ``get_transform``) falls back to its APE-14 ``pixel_to_world``.
    """
    get_transform = getattr(wcs, "get_transform", None)
    if callable(get_transform) and getattr(wcs, "available_frames", None) is not None:
        transform = get_transform("detector", "world")

        def to_world(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            world = transform(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
            return np.asarray(world[0], dtype=float), np.asarray(world[1], dtype=float)

        return to_world

    if hasattr(wcs, "pixel_to_world"):

        def to_world(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            sky = wcs.pixel_to_world(np.asarray(x), np.asarray(y))  # type: ignore[attr-defined]
            return (
                np.asarray(sky.ra.deg, dtype=float),
                np.asarray(sky.dec.deg, dtype=float),
            )

        return to_world

    raise TypeError(
        "wcs must satisfy the TransformWCS protocol (available_frames + "
        "get_transform, e.g. NoobWCS / gwcs) or expose an APE-14 pixel_to_world "
        f"(astropy.wcs); got {type(wcs).__name__}."
    )


class SourceExtractor:
    """Find, sky-match, and cache point-source detections across frames.

    Parameters
    ----------
    fwhm : float
        Expected point-source FWHM in pixels (passed to ``build_catalog``).
    cutout_size : int
        Edge length of the cached cutouts; must exceed the largest stamp to be
        sliced later (the wing size) plus a couple of pixels for re-centring.
    nsigma : float, default 5.0
        Detection threshold forwarded to ``build_catalog``.
    match_radius : float, default 0.1
        Sky-match tolerance in **arcseconds**: a new detection within this of a
        known source's mean position is the same physical source.
    aperture_radius, min_distance : float, optional
        Forwarded to ``build_catalog`` (default to ``4*sigma`` / ``fwhm``).
    dq_bad_bits : int, default 1
        DQ bit mask marking bad pixels (default ``DO_NOT_USE``).
    max_in_memory : int, optional
        Cutouts kept resident before they spill to disk; ``None`` (with no
        ``spill_dir``) keeps them all in memory. Use a context manager
        (``with SourceExtractor(...) as ext:``) or :meth:`close` to clean up the
        spill directory.
    spill_dir : str or Path, optional
        Base directory for spilled cutouts (a private sub-directory is created
        inside it); enables spilling even when ``max_in_memory`` is ``None``.
    """

    def __init__(
        self,
        *,
        fwhm: float,
        cutout_size: int,
        nsigma: float = 5.0,
        match_radius: float = 0.1,
        aperture_radius: float | None = None,
        min_distance: float | None = None,
        dq_bad_bits: int = 1,
        max_in_memory: int | None = None,
        spill_dir: str | Path | None = None,
    ) -> None:
        """See the class docstring for parameters."""
        if cutout_size <= 0:
            raise ValueError(f"cutout_size must be positive; got {cutout_size!r}.")
        self.fwhm = fwhm
        self.cutout_size = int(cutout_size)
        self.nsigma = nsigma
        self.match_radius = float(match_radius)
        self.aperture_radius = aperture_radius
        self.min_distance = min_distance
        self.dq_bad_bits = dq_bad_bits
        self.max_in_memory = max_in_memory
        self.spill_dir = spill_dir

        self._det = _Detections()
        self._store: CutoutStore = CutoutStore(
            self.cutout_size, max_in_memory=max_in_memory, spill_dir=spill_dir
        )

    def add_from_frame(
        self,
        data: np.ndarray,
        err: np.ndarray | None = None,
        dq: np.ndarray | None = None,
        *,
        wcs: object,
        mask: np.ndarray | None = None,
        filename: object = None,
        filter: object = None,  # noqa: A002 -- the JWST band label, an astro convention
        detector: object = None,
        max_ellipticity: float = 0.1,
    ) -> SourceCatalog:
        """Detect, match, and cache the sources of one frame.

        A detection is kept when it is **point-like here**
        (``ellipticity <= max_ellipticity``) **or matches** an already-known
        source within :attr:`match_radius`; everything else is dropped. Kept
        detections are assigned a physical-source ``index`` (an existing one if
        matched, else a new one) and a cached cutout; detections whose cutout
        would fall off the frame are dropped.

        Parameters
        ----------
        data : numpy.ndarray
            2-D calibrated, background-subtracted detector frame.
        err : numpy.ndarray, optional
            Per-pixel 1-sigma error map (for SNR and stamp weights); non-finite
            or negative pixels are excluded from detection.
        dq : numpy.ndarray, optional
            Integer data-quality array; flagged pixels (``dq_bad_bits``) are
            recorded in each cutout's ``bad`` mask **only** -- they are *not*
            excluded from detection (the pipeline DQ is too aggressive per frame).
        wcs : object
            Any WCS the package accepts: a
            :class:`~noobfriend.core.wcs.TransformWCS` (``available_frames`` +
            ``get_transform`` -- a compiled
            :class:`~noobfriend.core.wcs.NoobWCS`, as ``NooBook.wcs`` returns, a
            :class:`gwcs.WCS`, or the astropy/grizli adapters), or a bare
            :mod:`astropy.wcs` with an APE-14 ``pixel_to_world``. Required, since
            matching is by sky position.
        mask : numpy.ndarray, optional
            Boolean array marking pixels to exclude from detection (forwarded to
            ``build_catalog``) -- the trusted bad-pixel mask, separate from ``dq``.
        filename : object, optional
            Provenance label stored on every detection from this frame.
        filter, detector : object, optional
            The frame's band and detector, stored per detection so a later
            :meth:`select` can restrict to one ``(filter, module)`` group.
        max_ellipticity : float, default 0.1
            The point-like threshold for the keep rule above.

        Returns
        -------
        SourceCatalog
            The detections kept (and cached) from this frame.
        """
        catalogue = build_catalog(
            data,
            fwhm=self.fwhm,
            nsigma=self.nsigma,
            err=err,
            mask=mask,
            pixel_to_world=_pixel_to_world(wcs),
            aperture_radius=self.aperture_radius,
            min_distance=self.min_distance,
        )
        if len(catalogue) == 0:
            return catalogue

        matched, match_index = self._match(catalogue)
        keep = (catalogue.ellipticity <= max_ellipticity) | matched

        kept_rows: list[int] = []
        kept_index: list[int] = []
        for i in np.nonzero(keep)[0]:
            cutout = self._slice_cutout(data, err, dq, catalogue.x[i], catalogue.y[i])
            if cutout is None:
                continue  # window falls off the frame
            index = int(match_index[i]) if matched[i] else self._det.new_index()
            kept_index.append(index)
            self._store.append(cutout)
            kept_rows.append(int(i))

        if not kept_rows:
            return SourceCatalog.empty()
        rows = catalogue[np.asarray(kept_rows, dtype=int)]
        self._det.append_frame(
            rows, kept_index, filename=filename, filter=filter, detector=detector
        )
        return rows

    def _match(self, catalogue: SourceCatalog) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(matched, match_index)`` of each detection to a known source."""
        n = len(catalogue)
        if self._det.n_sources == 0:
            return np.zeros(n, dtype=bool), np.full(n, -1, dtype=int)

        from astropy.coordinates import SkyCoord

        mean_ra, mean_dec = self._det.source_means()
        known = SkyCoord(mean_ra, mean_dec, unit="deg")
        new = SkyCoord(catalogue.ra, catalogue.dec, unit="deg")
        index, separation, _ = new.match_to_catalog_sky(known)
        matched = separation.arcsec < self.match_radius
        return matched, np.where(matched, index, -1)

    def _slice_cutout(
        self,
        data: np.ndarray,
        err: np.ndarray | None,
        dq: np.ndarray | None,
        x: float,
        y: float,
    ) -> Cutout | None:
        """Cut a ``cutout_size`` window around ``(x, y)``, or ``None`` if off-frame."""
        size = self.cutout_size
        r0 = int(round(y)) - size // 2
        c0 = int(round(x)) - size // 2
        r1, c1 = r0 + size, c0 + size
        if r0 < 0 or c0 < 0 or r1 > data.shape[0] or c1 > data.shape[1]:
            return None
        data_cut = np.asarray(data[r0:r1, c0:c1], dtype=np.float32)
        err_cut = (
            None if err is None else np.asarray(err[r0:r1, c0:c1], dtype=np.float32)
        )
        bad_cut = (
            None
            if dq is None
            else (np.asarray(dq[r0:r1, c0:c1]).astype(np.int64) & self.dq_bad_bits) != 0
        )
        return Cutout(data=data_cut, err=err_cut, bad=bad_cut, origin=(r0, c0))

    # -- selection and build --------------------------------------------------

    def select(
        self,
        *,
        include: "list[int] | np.ndarray | None" = None,
        exclude: "list[int] | np.ndarray | None" = None,
        filter: object = None,  # noqa: A002 -- the JWST band label, an astro convention
        module: object = None,
        detector: object = None,
        snr_min: float | None = None,
        max_ellipticity: float | None = None,
        fwhm: tuple[float, float] | None = None,
        flux: tuple[float, float] | None = None,
        isolation_min: float | None = None,
    ) -> "SourceExtractor":
        """Return a sub-extractor of the detections passing every criterion.

        Three kinds of cut combine with logical AND:

        - **per-detection labels** -- ``filter`` / ``module`` (the NIRCam module
          letter, ``detector[3:4]``) / ``detector`` restrict to one group. String
          matchers use shell wildcards (plain strings are exact), compiled regex
          matchers use ``fullmatch``, and non-string values use equality;
        - **per-source aggregates** -- ``snr_min`` / ``fwhm`` / ``flux`` /
          ``isolation_min`` test a source's *median* over its (label-restricted)
          detections, while ``max_ellipticity`` tests the *maximum* ellipticity
          over **all** of a source's detections (so the cross-band point-likeness
          survives a later ``filter`` restriction);
        - **explicit ids** -- ``include`` keeps only those source indices,
          ``exclude`` drops them.

        Parameters
        ----------
        include, exclude : list of int, optional
            Source indices to keep / drop.
        filter, detector : object, optional
            Restrict to detections of this band / detector. Strings may include
            shell wildcards; compiled regular expressions match the whole label.
        module : object, optional
            Restrict to this NIRCam module letter (``"a"`` / ``"b"``), with the
            same string / regex matching rules as ``detector``.
        snr_min, isolation_min : float, optional
            Minimum per-source median SNR / nearest-neighbour distance.
        max_ellipticity : float, optional
            Maximum per-source ellipticity across all its bands.
        fwhm, flux : tuple of float, optional
            ``(low, high)`` band for the per-source median FWHM / flux.

        Returns
        -------
        SourceExtractor
            A sub-extractor (its own cutouts and re-compacted source indices).
        """
        n = len(self)
        if n == 0:
            return self._subset(np.zeros(0, dtype=bool))
        catalogue = self.catalog
        index = self.index
        keep = np.ones(n, dtype=bool)

        if max_ellipticity is not None:
            source_max = _group_aggregate(index, catalogue.ellipticity, np.max)
            keep &= np.array([source_max[i] <= max_ellipticity for i in index])
        if filter is not None:
            keep &= _match_labels(self._det.filters, filter)
        if detector is not None:
            keep &= _match_labels(self._det.detectors, detector)
        if module is not None:
            letters = np.array(
                [(d or "")[3:4] for d in self._det.detectors], dtype=object
            )
            keep &= _match_labels(letters, module)

        if any(c is not None for c in (snr_min, fwhm, flux, isolation_min)):
            kept_index = index[keep]
            passing = set(np.unique(kept_index).tolist())
            checks = [
                (snr_min, catalogue.snr, lambda a: a >= snr_min),
                (fwhm, catalogue.fwhm, lambda a: fwhm[0] <= a <= fwhm[1]),
                (flux, catalogue.flux, lambda a: flux[0] <= a <= flux[1]),
                (isolation_min, catalogue.nn_dist, lambda a: a >= isolation_min),
            ]
            for criterion, values, ok in checks:
                if criterion is None:
                    continue
                median = _group_aggregate(kept_index, values[keep], np.median)
                passing &= {i for i in passing if ok(median[i])}
            keep &= np.isin(index, list(passing))

        if include is not None:
            keep &= np.isin(index, np.asarray(include))
        if exclude is not None:
            keep &= ~np.isin(index, np.asarray(exclude))
        return self._subset(keep)

    def _subset(self, keep: np.ndarray) -> "SourceExtractor":
        """Return a new extractor over the kept detections (indices re-compacted)."""
        sub = SourceExtractor(
            fwhm=self.fwhm,
            cutout_size=self.cutout_size,
            nsigma=self.nsigma,
            match_radius=self.match_radius,
            aperture_radius=self.aperture_radius,
            min_distance=self.min_distance,
            dq_bad_bits=self.dq_bad_bits,
            max_in_memory=self.max_in_memory,
            spill_dir=self.spill_dir,
        )
        rows = np.flatnonzero(keep)
        if rows.size == 0:
            return sub
        sub._det = self._det.subset(rows)
        # A lazy view over the parent store: select stays cheap (no cutout IO),
        # and only a later build reads the selected stamps. The sub-extractor is
        # thus bound to this extractor's cutouts -- do not close the parent while
        # the sub-extractor is still in use.
        sub._store = self._store.view(rows)
        return sub

    def build_psf_core(
        self,
        *,
        oversample: int = 3,
        core_size: int = 21,
        max_iter: int = 50,
        tol: float = 1e-4,
        residual_reweight: str = "huber",
    ) -> CorePsf:
        """Build the oversampled core from this (selected) extractor's cutouts.

        Slices a ``core_size`` stamp from each cached cutout and drives
        :func:`noobase.image.psf.build_epsf`. Select first (e.g.
        ``ext.select(filter="F210M", module="a", ...)``) so only one group's
        point-like sources are used.

        Returns
        -------
        CorePsf
        """
        if self.cutout_size < core_size:
            raise ValueError(
                f"cutout_size {self.cutout_size} is smaller than core_size {core_size}."
            )
        return build_core(
            self._store,
            self.n_sources,
            oversample=oversample,
            core_size=core_size,
            max_iter=max_iter,
            tol=tol,
            residual_reweight=residual_reweight,
        )

    def build_psf_wings(
        self, core: CorePsf, *, wing_size: int = 51, **extended_kwargs: object
    ) -> EffectivePsf:
        """Stitch native wings onto ``core`` from this (selected) extractor.

        Slices a larger ``wing_size`` stamp from each cached cutout and drives
        :func:`noobase.image.psf.build_extended_psf`, which needs the ``core`` to
        stitch onto. Typically called on a brighter, more isolated selection than
        the core (``ext.select(snr_min=..., isolation_min=...)``).

        Returns
        -------
        EffectivePsf
        """
        if self.cutout_size < wing_size:
            raise ValueError(
                f"cutout_size {self.cutout_size} is smaller than wing_size {wing_size}."
            )
        return build_wings(self._store, core, wing_size=wing_size, **extended_kwargs)

    # -- persistence ----------------------------------------------------------

    def save(
        self, path: str | Path, *, overwrite: bool = False, chunk_size: int = 4096
    ) -> None:
        """Persist this extractor to ``path`` for a later :meth:`load`.

        Writes a self-contained directory: ``meta.json`` (config and counts),
        ``catalog.fits`` (the detection table plus per-detection ``index`` /
        ``filename`` / ``filter`` / ``detector`` labels as a FITS binary table)
        and ``cutouts/`` (the cached cutouts as ``.npz`` chunks). ``meta.json`` is
        written **last** as a commit marker, so an interrupted save is detectable
        (:meth:`load` requires it). Cutouts are streamed out chunk by chunk, so
        saving never needs the whole cache resident.

        Pre-filter with :meth:`select` before saving to persist only the cutouts
        you intend to build from -- e.g.
        ``ext.select(snr_min=8, max_ellipticity=0.25).save(path)``.

        Parameters
        ----------
        path : str or Path
            Destination directory.
        overwrite : bool, default False
            Replace ``path`` if it already exists.
        chunk_size : int, default 4096
            Cutouts per ``.npz`` chunk file.

        Raises
        ------
        FileExistsError
            ``path`` already exists and ``overwrite`` is ``False``.
        """
        import json
        import shutil

        path = Path(path)
        if path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"{path} already exists; pass overwrite=True to replace it."
                )
            shutil.rmtree(path) if path.is_dir() else path.unlink()
        path.mkdir(parents=True)

        self._det.to_table().write(path / "catalog.fits", format="fits", overwrite=True)
        chunk_lens = self._store.save(path / "cutouts", chunk_size=chunk_size)

        meta = {
            "format_version": _SAVE_FORMAT_VERSION,
            "fwhm": self.fwhm,
            "cutout_size": self.cutout_size,
            "nsigma": self.nsigma,
            "match_radius": self.match_radius,
            "aperture_radius": self.aperture_radius,
            "min_distance": self.min_distance,
            "dq_bad_bits": self.dq_bad_bits,
            "n_detections": len(self),
            "n_sources": self.n_sources,
            "chunk_lens": chunk_lens,
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        max_in_memory: int | None = None,
        spill_dir: str | Path | None = None,
    ) -> "SourceExtractor":
        """Reconstruct an extractor written by :meth:`save`.

        The cutouts stay on disk and are read lazily, so a large cache is not
        fully materialised; a later :meth:`select` streams through them. The
        per-source sky positions are recomputed from the catalogue rather than
        stored. ``max_in_memory`` / ``spill_dir`` configure the *new* stores that
        :meth:`select` derives -- the loaded cache directory itself is never
        written to or deleted.

        Parameters
        ----------
        path : str or Path
            Directory written by :meth:`save`.
        max_in_memory, spill_dir
            Spill configuration for stores derived via :meth:`select`.

        Returns
        -------
        SourceExtractor

        Raises
        ------
        FileNotFoundError
            ``path`` has no ``meta.json`` (absent, or an interrupted save).
        """
        import json

        path = Path(path)
        meta_path = path / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"{meta_path} not found; not a SourceExtractor save (or incomplete)."
            )
        meta = json.loads(meta_path.read_text())

        from astropy.table import Table

        ext = cls(
            fwhm=meta["fwhm"],
            cutout_size=meta["cutout_size"],
            nsigma=meta["nsigma"],
            match_radius=meta["match_radius"],
            aperture_radius=meta["aperture_radius"],
            min_distance=meta["min_distance"],
            dq_bad_bits=meta["dq_bad_bits"],
            max_in_memory=max_in_memory,
            spill_dir=spill_dir,
        )
        ext._det = _Detections.from_table(
            Table.read(path / "catalog.fits", format="fits")
        )
        ext._store = CutoutStore.load(
            path / "cutouts",
            meta["cutout_size"],
            max_in_memory=max_in_memory,
            spill_dir=spill_dir,
        )
        return ext

    def panel(
        self,
        *,
        band: object = None,
        show_thumbs: bool = True,
        max_thumbs: int = 3,
        height: int = 820,
    ) -> object:
        """Return the interactive Bokeh selection panel for these sources.

        Aggregates the detections to one point per physical source: ``flux`` /
        ``fwhm`` / ``ell`` / ``snr`` are medians over the ``band`` detections (all
        bands if ``band`` is ``None``), and the colour is the cross-band maximum
        ellipticity. Read the slider ranges and the copied include / exclude
        index lists back into :meth:`select`.

        Parameters
        ----------
        band : object, optional
            Plot only sources detected in this band (their ``filter`` label).
        show_thumbs : bool, default True
            Embed per-source thumbnails (one per band, up to ``max_thumbs``),
            shown in a hover-switching preview panel. Pass ``False`` for a large
            catalogue to skip rendering them.
        max_thumbs : int, default 3
            Maximum thumbnails per source.
        height : int, default 820
            Iframe height of the returned display handle.

        Returns
        -------
        A notebook-displayable Bokeh handle.
        """
        from noobfriend.core.display.plot import psf_select_panel

        index = self.index
        catalogue = self.catalog
        labels = np.asarray(self._det.filters, dtype=object)
        in_band = np.ones(len(index), dtype=bool) if band is None else (labels == band)
        source_ids = np.unique(index[in_band]) if len(index) else np.empty(0, dtype=int)

        flux, fwhm, ell, snr, max_ell = ([] for _ in range(5))
        thumbs: list[list[np.ndarray]] = []
        thumb_labels: list[list[str]] = []
        for source in source_ids:
            rows = np.flatnonzero(index == source)
            band_rows = rows if band is None else rows[in_band[rows]]
            flux.append(float(np.median(catalogue.flux[band_rows])))
            fwhm.append(float(np.median(catalogue.fwhm[band_rows])))
            ell.append(float(np.median(catalogue.ellipticity[band_rows])))
            snr.append(float(np.median(catalogue.snr[band_rows])))
            max_ell.append(float(np.max(catalogue.ellipticity[rows])))
            if show_thumbs:
                arrays, names = [], []
                for name in sorted({labels[r] for r in rows}, key=str):
                    band_subset = rows[labels[rows] == name]
                    best = int(band_subset[np.argmax(catalogue.snr[band_subset])])
                    arrays.append(self._store[best].data)
                    names.append(
                        f"<b>{name}</b><br>fwhm {np.median(catalogue.fwhm[band_subset]):.2f}"
                        f" · ell {np.median(catalogue.ellipticity[band_subset]):.3f}"
                        f"<br>snr {np.median(catalogue.snr[band_subset]):.0f}"
                        f" · n {band_subset.size}"
                    )
                    if len(arrays) >= max_thumbs:
                        break
                thumbs.append(arrays)
                thumb_labels.append(names)

        return psf_select_panel(
            np.asarray(source_ids),
            np.asarray(flux),
            np.asarray(fwhm),
            np.asarray(ell),
            np.asarray(max_ell),
            np.asarray(snr),
            thumbs=thumbs if show_thumbs else None,
            thumb_labels=thumb_labels if show_thumbs else None,
            show_thumbs=show_thumbs,
            height=height,
        )

    def close(self) -> None:
        """Remove any spilled cutout files (idempotent). Safe to call always."""
        self._store.close()

    def __enter__(self) -> "SourceExtractor":
        """Enter a context that :meth:`close`-s the extractor on exit."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Close the extractor, cleaning up any spilled cutouts."""
        self.close()

    # -- accessors ------------------------------------------------------------

    @property
    def catalog(self) -> SourceCatalog:
        """The accumulated detection catalogue (one row per kept detection)."""
        return self._det.catalog

    @property
    def index(self) -> np.ndarray:
        """Physical-source index of each detection (groups rows across frames)."""
        return self._det.index

    @property
    def filename(self) -> np.ndarray:
        """Provenance label of each detection's source frame."""
        return np.asarray(self._det.filenames, dtype=object)

    @property
    def cutouts(self) -> CutoutStore:
        """The cutout store (indexable / iterable), aligned with :attr:`catalog`."""
        return self._store

    @property
    def n_sources(self) -> int:
        """Number of distinct physical sources collected so far."""
        return self._det.n_sources

    def __len__(self) -> int:
        """Return the number of detections collected so far."""
        return len(self._det)
