"""Blind (catalog-free) emission-line detection in WFSS grism frames.

The discovery counterpart to the targeted ``GrismExtractor``: it sweeps single
dispersed frames for compact emission-line features without a source catalog or
direct image. The primary product is a band-pass SNR **heatmap** (hot = likely
line), per exposure and combined over same-field dithers; a peak ``catalog`` is
an auxiliary convenience. Source association and redshift are out of scope.

The public surface is :class:`GrismLineFinder` (the configured orchestrator)
plus the result types :class:`CombinedHeatmap`, :class:`UnionGrid`, and
:class:`Candidate`. The modules they are built from (``_filter``, ``_detect``,
``_combine``, ``_linefind``) are internal.
"""

from noobfriend.extraction.grism.linefind._combine import (
    CombinedHeatmap,
    UnionGrid,
)
from noobfriend.extraction.grism.linefind._detect import Candidate
from noobfriend.extraction.grism.linefind._linefind import GrismLineFinder

__all__ = ["GrismLineFinder", "CombinedHeatmap", "UnionGrid", "Candidate"]
