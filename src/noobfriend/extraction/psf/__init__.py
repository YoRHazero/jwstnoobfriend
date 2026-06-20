"""Effective-PSF construction from calibrated per-exposure detector frames.

The input layer for the noobase effective-PSF engine. :class:`SourceExtractor`
finds clean point sources in calibrated frames, matches them across frames and
bands by sky position into one flat detection catalogue, and caches a fixed-size
cutout per detection; :meth:`SourceExtractor.select` carves out one
``(filter, module)`` group of point-like sources, and
:meth:`SourceExtractor.build_psf_core` / :meth:`SourceExtractor.build_psf_wings`
drive the noobase engine (the wings stitch onto a :class:`CorePsf`).

``SourceExtractor`` takes plain arrays plus a WCS, never a
:class:`~noobfriend.navigation.NooBox`, so navigation can drive it but it never
depends on navigation.
"""

from noobfriend.extraction.psf._build import (
    CorePsf,
    EffectivePsf,
    build_core,
    build_wings,
)
from noobfriend.extraction.psf._extractor import SourceExtractor
from noobfriend.extraction.psf._render import convolve, render_core, render_effective
from noobfriend.extraction.psf._store import Cutout, CutoutStore

__all__ = [
    "CorePsf",
    "Cutout",
    "CutoutStore",
    "EffectivePsf",
    "SourceExtractor",
    "build_core",
    "build_wings",
    "convolve",
    "render_core",
    "render_effective",
]
