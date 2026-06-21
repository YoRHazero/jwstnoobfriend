"""Grism (WFSS) spectral extraction from JWST products.

Given a source's world position, find the exposures whose order-1 trace covers
it, load only those, then rectify and combine them into 2-D spectra::

    from pathlib import Path
    from noobfriend.extraction.grism import (
        FrameMeta, find_coverage, read_wavelength_domain, GrismExtractor,
    )

    metas = [FrameMeta(id=Path(p).stem, wcs=w, shape=s) for p, w, s in frames]
    wrange = read_wavelength_domain(metas[0].wcs, ra, dec)   # read model once
    coverage = find_coverage(ra, dec, metas, wavelength_range=wrange)
    covered = [c for c in coverage if c.covered]
    extractor = GrismExtractor.from_world(ra, dec, covered, spatial_half=8)
    spectra = extractor.rectify(load)         # load(id) -> (data, error), per exposure
    products = extractor.combine(spectra)     # stack by group -> one per group

Going the other way -- from a feature seen at one dispersed pixel back to the
source positions that could have produced it -- use :func:`source_locus`::

    from noobfriend.extraction.grism import source_locus

    locus = source_locus(wcs, x_line, y_line)    # the (x0, y0, lambda) degeneracy
    ys, xs = locus.pixels(bounds=direct.shape)   # candidate pixels to cross-match

The public surface is ``read_wavelength_domain``, ``FrameMeta``,
``FrameCoverage``, ``find_coverage``, ``GrismExtractor``, ``GrismSpectrum``,
``SourceLocus``, ``source_locus``, and ``source_loci``; the modules they are
built from (``_wavelength``, ``_coverage``, ``_core``, ``_locus``) are internal.
"""

from noobfriend.extraction.grism._core import GrismExtractor, GrismSpectrum
from noobfriend.extraction.grism._coverage import (
    FrameCoverage,
    FrameMeta,
    find_coverage,
)
from noobfriend.extraction.grism._locus import (
    SourceLocus,
    source_loci,
    source_locus,
)
from noobfriend.extraction.grism._wavelength import read_wavelength_domain

__all__ = [
    "read_wavelength_domain",
    "FrameMeta",
    "FrameCoverage",
    "find_coverage",
    "GrismExtractor",
    "GrismSpectrum",
    "SourceLocus",
    "source_locus",
    "source_loci",
]
