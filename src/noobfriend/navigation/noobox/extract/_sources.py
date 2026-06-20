"""The :meth:`BoxExtract.sources` implementation: box -> PSF source extractor.

Turns a curated :class:`~noobfriend.navigation.NooBox` into the input for
:mod:`noobfriend.extraction.psf`: it iterates the box members, reads each one's
``SCI`` / ``ERR`` / ``DQ`` arrays and WCS through the shared byte cache, attaches
per-product provenance labels, and feeds them to
:meth:`~noobfriend.extraction.psf.SourceExtractor.add_from_frame`. The returned
:class:`~noobfriend.extraction.psf.SourceExtractor` owns the science state and
lives in :mod:`noobfriend.extraction` -- navigation drives it but it never
depends on navigation (the accepted asymmetry with :class:`BoxCutout`).
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

#: ``mask_fn(book, data, err, dq) -> mask`` for pixels to exclude from detection.
MaskBuilder = Callable[
    ["NooBook", np.ndarray, np.ndarray | None, np.ndarray | None], np.ndarray | None
]

DEFAULT_PSF_FWHM = 4.0
DEFAULT_PSF_CUTOUT_SIZE = 55


def build_sources(
    box: "NooBox",
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
    """Build a PSF source extractor from ``box`` (see :meth:`BoxExtract.sources`)."""
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

    books = list(box)
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
