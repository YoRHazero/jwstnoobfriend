"""The :class:`BoxExtract` accessor: collection-level extraction for a NooBox.

Reached as
:attr:`NooBox.extract <noobfriend.navigation.noobox._core.NooBox.extract>`. It
turns a curated collection of :class:`~noobfriend.navigation.NooBook` products
into extraction inputs; each method keeps the collection concerns here
(iteration order, shared byte-cache reads, per-product provenance) and delegates
the work to a sibling module:

- :meth:`~BoxExtract.cutout` -> :class:`~.._cutout.BoxCutout` (a navigation type,
  keyed on NooBook);
- :meth:`~BoxExtract.sources` -> :class:`~noobfriend.extraction.psf.SourceExtractor`
  (a navigation-free engine in :mod:`noobfriend.extraction`).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from noobfriend.navigation.noobox.extract._sources import (
    DEFAULT_PSF_CUTOUT_SIZE,
    DEFAULT_PSF_FWHM,
    MaskBuilder,
)

if TYPE_CHECKING:
    from noobfriend.extraction.psf import SourceExtractor
    from noobfriend.navigation.noobook._core import NooBook
    from noobfriend.navigation.noobox._core import NooBox
    from noobfriend.navigation.noobox.extract._cutout import BoxCutout


class BoxExtract:
    """Extraction adapters bound to one :class:`NooBox`.

    ``BoxExtract`` keeps collection concerns in navigation: iteration order,
    shared byte-cache reads, and per-product provenance labels. It does not hold
    extraction results; each method returns the object that owns the accumulated
    state.

    Parameters
    ----------
    box : NooBox
        The collection whose members are fed into extraction workflows.
    """

    def __init__(self, box: "NooBox") -> None:
        """See the class docstring for parameters."""
        self._box = box

    def cutout(
        self,
        ra: float,
        dec: float,
        *,
        half: int | None = None,
        half_width: int | None = None,
        half_height: int | None = None,
        left: int | None = None,
        right: int | None = None,
        top: int | None = None,
        bottom: int | None = None,
        half_world: float | None = None,
        on: "NooBook | str | None" = None,
        reproject: bool = True,
        with_err: bool = False,
        skip_missing_wcs: bool = True,
        probe: bool = True,
        progress: bool = True,
    ) -> "BoxCutout":
        """Cut one sky position out of every box member that covers it.

        Each member's resident :class:`~noobfriend.navigation._footprint.Footprint`
        is tested first, so only covering books are read; thin books are probed
        (``probe=True``) to obtain their footprint. The size is one of the
        :meth:`~noobfriend.extraction.cutout.Cutout.from_world` forms (give
        exactly one). Curate the box with :meth:`~NooBox.select` /
        :meth:`~NooBox.filter` first to pick the relevant stage, field or band.

        Two grid models:

        - ``reproject=True`` (default): every covering book is resampled onto a
          single shared grid (surface-brightness conserving), so the stamps are
          aligned and can be :meth:`~.._cutout.BoxCutout.coadd` /
          :meth:`~.._cutout.BoxCutout.blink`. The grid is defined by ``on`` (a
          book or its id/stage), else a covering stage-3 mosaic, else the first
          covering book.
        - ``reproject=False``: each book is an axis-aligned integer slice on its
          own grid (cheaper; stamps may differ in size and orientation).

        Parameters
        ----------
        ra, dec : float
            Cutout centre in degrees.
        half, half_width, half_height, left, right, top, bottom, half_world
            Size specification forwarded to
            :meth:`~noobfriend.extraction.cutout.Cutout.from_world`; provide
            exactly one form.
        on : NooBook or str, optional
            Only with ``reproject=True``: the book (or its id / stage) whose WCS
            defines the common grid. Defaults to a covering stage-3 mosaic, then
            the first covering book.
        reproject : bool, default True
            Resample onto a common grid (``True``) or slice each book natively
            (``False``).
        with_err : bool, default False
            Also extract each book's ``ERR`` array. Display-grade once
            reprojected (reprojection correlates noise); off by default.
        skip_missing_wcs : bool, default True
            Skip books with no footprint (no assigned WCS, or unprobed with
            ``probe=False``); when ``False``, raise instead.
        probe : bool, default True
            Probe thin books so their footprint is available for the coverage
            test.
        progress : bool, default True
            Show a progress bar while reading the covering books.

        Returns
        -------
        BoxCutout
            The stamps, keyed by the covering books.

        Raises
        ------
        ValueError
            If no member covers ``(ra, dec)``, or the size form is invalid, or a
            footprint is missing while ``skip_missing_wcs`` is ``False``.
        """
        from noobfriend.navigation.noobox.extract._cutout import _build

        return _build(
            self._box,
            ra,
            dec,
            half=half,
            half_width=half_width,
            half_height=half_height,
            left=left,
            right=right,
            top=top,
            bottom=bottom,
            half_world=half_world,
            on=on,
            reproject=reproject,
            with_err=with_err,
            skip_missing_wcs=skip_missing_wcs,
            probe=probe,
            progress=progress,
        )

    def sources(
        self,
        *,
        fwhm: float = DEFAULT_PSF_FWHM,
        cutout_size: int = DEFAULT_PSF_CUTOUT_SIZE,
        nsigma: float = 5.0,
        match_radius: float = 0.1,
        aperture_radius: float | None = None,
        min_distance: float | None = None,
        dq_bad_bits: int = 1,
        max_in_memory: int | None = None,
        spill_dir: str | Path | None = None,
        max_ellipticity: float = 0.1,
        mask_fn: MaskBuilder | None = None,
        skip_missing_wcs: bool = False,
        probe: bool = True,
        progress: bool = True,
    ) -> "SourceExtractor":
        """Build a PSF source extractor from this box's products.

        This is the hand-off from navigation to PSF extraction. The method
        iterates the current box members in order, optionally probes thin books
        for resident header labels, reads each member's ``SCI`` / ``ERR`` /
        ``DQ`` arrays and WCS, then calls
        :meth:`noobfriend.extraction.psf.SourceExtractor.add_from_frame`. (It
        builds a source catalogue; the PSF itself is built later via
        ``SourceExtractor.build_psf_core`` / ``build_psf_wings`` -- hence
        ``sources``, not ``psf``.)

        Curate the box first with :meth:`NooBox.select` or :meth:`NooBox.filter`
        to choose the relevant stage, field or detector set. The returned
        :class:`~noobfriend.extraction.psf.SourceExtractor` is where sky
        matching, candidate selection, the interactive panel, and core / wing
        PSF building happen.

        The defaults are a practical first pass for calibrated NIRCam imaging:
        ``fwhm=4.0`` pixels gives the matched-filter detector a broad
        point-source scale, and ``cutout_size=55`` supports the default
        :meth:`~noobfriend.extraction.psf.SourceExtractor.build_psf_wings`
        ``wing_size=51`` while staying compact.

        Typical workflow::

            ext = box.select(stage="2bi").extract.sources()
            ext.panel(band="F210M")
            stars = ext.select(filter="F210M", detector="nrca*")
            core = stars.build_psf_core()
            psf = stars.build_psf_wings(core)

        Parameters
        ----------
        fwhm : float, default 4.0
            Expected point-source FWHM in pixels, forwarded to the detection
            layer. A starting value, not a calibrated instrument constant.
        cutout_size : int, default 55
            Edge length of the cached per-detection cutouts. The default supports
            the default PSF wing build (``wing_size=51``).
        nsigma, match_radius, aperture_radius, min_distance, dq_bad_bits,
        max_in_memory, spill_dir
            Forwarded to :class:`noobfriend.extraction.psf.SourceExtractor`.
        max_ellipticity : float, default 0.1
            Per-frame point-like threshold forwarded to
            :meth:`~noobfriend.extraction.psf.SourceExtractor.add_from_frame`.
        mask_fn : callable, optional
            ``mask_fn(book, data, err, dq) -> mask`` for trusted pixels to
            exclude from detection. ``DQ`` is still forwarded separately to the
            PSF extractor as each cutout's bad-pixel mask.
        skip_missing_wcs : bool, default False
            When ``True``, silently skip products without assigned WCS. Otherwise
            raise a :class:`ValueError`.
        probe : bool, default True
            Probe thin books before extraction so header labels like ``filter``
            are available for later ``SourceExtractor.select(filter=...)`` calls.
        progress : bool, default True
            Show a progress bar while iterating over the box.

        Returns
        -------
        SourceExtractor
            The accumulated PSF source extractor.
        """
        from noobfriend.navigation.noobox.extract._sources import build_sources

        return build_sources(
            self._box,
            fwhm=fwhm,
            cutout_size=cutout_size,
            nsigma=nsigma,
            match_radius=match_radius,
            aperture_radius=aperture_radius,
            min_distance=min_distance,
            dq_bad_bits=dq_bad_bits,
            max_in_memory=max_in_memory,
            spill_dir=spill_dir,
            max_ellipticity=max_ellipticity,
            mask_fn=mask_fn,
            skip_missing_wcs=skip_missing_wcs,
            probe=probe,
            progress=progress,
        )
