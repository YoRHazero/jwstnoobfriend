"""Collection-level extraction adapters for :class:`NooBox`.

Reached as
:attr:`NooBox.extract <noobfriend.navigation.noobox._core.NooBox.extract>`.
This layer turns a curated collection of :class:`~noobfriend.navigation.NooBook`
products into inputs for :mod:`noobfriend.extraction`; the returned extraction
object owns the science-specific state and follow-up operations.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from noobfriend.extraction.psf import SourceExtractor
    from noobfriend.navigation.noobook._core import NooBook
    from noobfriend.navigation.noobox._core import NooBox

MaskBuilder = Callable[
    ["NooBook", np.ndarray, np.ndarray | None, np.ndarray | None], np.ndarray | None
]

_DEFAULT_PSF_FWHM = 4.0
_DEFAULT_PSF_CUTOUT_SIZE = 55


class BoxExtract:
    """Extraction adapters bound to one :class:`NooBox`.

    ``BoxExtract`` keeps collection concerns in navigation: iteration order,
    shared byte-cache reads, and per-product provenance labels. It does not hold
    extraction results; each method returns an object from
    :mod:`noobfriend.extraction` that owns the accumulated state.

    Parameters
    ----------
    box : NooBox
        The collection whose members are fed into extraction workflows.
    """

    def __init__(self, box: "NooBox") -> None:
        """See the class docstring for parameters."""
        self._box = box

    def psf(
        self,
        *,
        fwhm: float = _DEFAULT_PSF_FWHM,
        cutout_size: int = _DEFAULT_PSF_CUTOUT_SIZE,
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
        """Create a PSF source extractor from this box's products.

        This is the hand-off from navigation to PSF extraction. The method
        iterates the current box members in order, optionally probes thin books
        for resident header labels, reads each member's ``SCI`` / ``ERR`` /
        ``DQ`` arrays and WCS, then calls
        :meth:`noobfriend.extraction.psf.SourceExtractor.add_from_frame`.

        Curate the box first with :meth:`NooBox.select` or :meth:`NooBox.filter`
        to choose the relevant stage, field or detector set. The returned
        :class:`~noobfriend.extraction.psf.SourceExtractor` is where sky
        matching, candidate selection, the interactive panel, and core / wing
        PSF building happen.

        The defaults are intended as a practical first pass for calibrated
        NIRCam imaging products: ``fwhm=4.0`` pixels gives the matched-filter
        detector a broad point-source scale to start from, and
        ``cutout_size=55`` is large enough for the default
        :meth:`~noobfriend.extraction.psf.SourceExtractor.build_psf_wings`
        ``wing_size=51`` while still keeping per-detection cutouts compact. Tune
        ``fwhm`` when the stellar locus or detections show the PSF is
        substantially narrower or broader in the selected products. Increase
        ``cutout_size`` before asking for larger wings.

        Typical workflow::

            ext = box.select(stage="2a").extract.psf()
            ext.panel(band="F210M")
            stars = ext.select(filter="F210M", detector="nrca*")
            core = stars.build_psf_core()
            psf = stars.build_psf_wings(core)

        Parameters
        ----------
        fwhm : float, default 4.0
            Expected point-source FWHM in pixels, forwarded to the detection
            layer. Treat it as a starting value, not a calibrated instrument
            constant.
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
        from noobfriend.extraction.psf import SourceExtractor

        extractor = SourceExtractor(
            fwhm=fwhm,
            cutout_size=cutout_size,
            nsigma=nsigma,
            match_radius=match_radius,
            aperture_radius=aperture_radius,
            min_distance=min_distance,
            dq_bad_bits=dq_bad_bits,
            max_in_memory=max_in_memory,
            spill_dir=spill_dir,
        )

        books = list(self._box)
        if progress:
            from noobfriend.core.display import track

            iterable = track(books, "Extracting PSF candidates")
        else:
            iterable = books

        for original in iterable:
            book = original.probe() if probe and not original.is_probed else original
            wcs = book.wcs
            if wcs is None:
                if skip_missing_wcs:
                    continue
                raise ValueError(f"{book.id} has no assigned WCS for PSF extraction.")

            data = book.data
            err = book.err
            dq = book.dq
            mask = None if mask_fn is None else mask_fn(book, data, err, dq)
            extractor.add_from_frame(
                data,
                err,
                dq,
                wcs=wcs,
                mask=mask,
                filename=book.id,
                filter=book.filter,
                detector=book.detector,
                max_ellipticity=max_ellipticity,
            )

        return extractor
