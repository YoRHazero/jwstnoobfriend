"""Cutout extraction from JWST products.

Build a region with one of the :class:`Cutout` factory methods, then extract it
from any data array::

    from noobfriend.extraction.cutout import Cutout

    cut = Cutout.from_world(wcs, ra, dec, half=64)
    stamp = cut.array(science_data)          # fast integer slice
    image, footprint, weight = cut.reproject(other_data, other_wcs)  # resample

``Cutout`` is the entire public surface; the modules it is built from
(``_core``, ``_geometry``, ``_slice``, ``_reproject``) are internal.
"""

from noobfriend.extraction.cutout._core import Cutout

__all__ = ["Cutout"]
