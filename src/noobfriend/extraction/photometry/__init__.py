"""Multi-band aperture photometry and SED extraction.

The public entry point is :class:`ApertureSED`: pass a mapping of band names to
plain dictionaries (image data, WCS, and optional error / wavelength /
calibration metadata), then call :meth:`ApertureSED.measure` at a source
position to obtain an :class:`ApertureSEDResult`.

Each band is measured on its *own* native pixel grid through one shared union
aperture; only binary aperture masks are reprojected between grids, never the
science data. The caller-facing API stays dictionary-based — internal types
normalize those inputs, so callers never construct them.
"""

from noobfriend.extraction.photometry._core import ApertureSED
from noobfriend.extraction.photometry._measure import BandPhotometry
from noobfriend.extraction.photometry._result import ApertureSEDResult

__all__ = ["ApertureSED", "ApertureSEDResult", "BandPhotometry"]
