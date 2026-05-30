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
    beams = extractor.rectify(load)           # load(id) -> (data, error), per exposure
    products = extractor.combine(beams)       # stack by group -> one per group

The public surface is ``read_wavelength_domain``, ``FrameMeta``,
``FrameCoverage``, ``find_coverage``, ``GrismExtractor``, and ``GrismSpectrum``;
the modules they are built from (``_wavelength``, ``_coverage``, ``_core``) are
internal.
"""

from noobfriend.extraction.grism._core import GrismExtractor, GrismSpectrum
from noobfriend.extraction.grism._coverage import (
    FrameCoverage,
    FrameMeta,
    find_coverage,
)
from noobfriend.extraction.grism._wavelength import read_wavelength_domain

__all__ = [
    "read_wavelength_domain",
    "FrameMeta",
    "FrameCoverage",
    "find_coverage",
    "GrismExtractor",
    "GrismSpectrum",
]
