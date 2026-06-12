"""Mechanism-only spectrum-array utilities (the spectral sibling of imgutils).

Pixel-space helpers for 2-D and 1-D spectra, with no fitting or astronomical
interpretation:

- :func:`collapse` -- sum a 2-D spectrum over a cross-dispersion window into a
  1-D flux + naive (independent-pixel) error.
- :func:`cross_dispersion_boost` -- the correlated-noise inflation of that error,
  estimated from the 2-D source-free background's cross-dispersion correlation.
- :func:`continuum_boost` -- the same inflation estimated empirically from a 1-D
  spectrum's line-free scatter.
"""

from noobfriend.core.specutils._collapse import collapse
from noobfriend.core.specutils._noise import continuum_boost, cross_dispersion_boost

__all__ = [
    "collapse",
    "continuum_boost",
    "cross_dispersion_boost",
]
