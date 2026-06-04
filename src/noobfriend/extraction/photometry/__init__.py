"""Multi-band aperture photometry and SED plotting.

The public entry point is :class:`ApertureSED`: pass a mapping of band names to
plain dictionaries containing image data, WCS, and optional uncertainty /
wavelength metadata, then call :meth:`ApertureSED.measure` at a source position.

The package deliberately keeps the caller-facing API dictionary-based. Internal
dataclasses normalize those inputs, but callers do not have to construct them.
"""

from noobfriend.extraction.photometry._core import (
    ApertureSED,
    ApertureSEDResult,
    BandPhotometry,
)

__all__ = ["ApertureSED", "ApertureSEDResult", "BandPhotometry"]
