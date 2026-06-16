"""The :class:`BoxExtract` accessor: extraction sugar bound to one NooBox.

Reached as
:attr:`NooBox.extract <noobfriend.navigation.noobox._core.NooBox.extract>`.
Methods delegate to :mod:`noobfriend.extraction`, using the box as the
collection driver and byte-cache owner.
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


class BoxExtract:
    """Extraction sugar bound to one :class:`NooBox`.

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
        fwhm: float,
        cutout_size: int,
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
        """Accumulate PSF-star candidates from every member of this box.

        The box drives I/O and provenance only: each member contributes its
        ``SCI`` / ``ERR`` / ``DQ`` arrays plus WCS, filter and detector labels to
        :meth:`noobfriend.extraction.psf.SourceExtractor.add_from_frame`. The
        returned extractor owns the accumulated detections, cutouts, selection
        panel and core / wing PSF build methods.

        Parameters
        ----------
        fwhm, cutout_size, nsigma, match_radius, aperture_radius, min_distance,
        dq_bad_bits, max_in_memory, spill_dir
            Forwarded to :class:`noobfriend.extraction.psf.SourceExtractor`.
        max_ellipticity : float, default 0.1
            Per-frame point-like threshold forwarded to
            :meth:`~noobfriend.extraction.psf.SourceExtractor.add_from_frame`.
        mask_fn : callable, optional
            ``mask_fn(book, data, err, dq) -> mask`` for a trusted bad-pixel mask
            passed to detection. ``DQ`` is still forwarded separately as the PSF
            cutout bad-pixel mask.
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
