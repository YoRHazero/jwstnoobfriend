"""Shared astropy TAN-WCS builders for the mosaic tests."""

import numpy as np
from astropy.wcs import WCS


def tan_wcs(ra0: float, dec0: float, shape: tuple[int, int], scale: float) -> WCS:
    """Return a plain TAN astropy WCS centred at (ra0, dec0), scale in arcsec/pix."""
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra0, dec0]
    w.wcs.crpix = [shape[1] / 2 + 0.5, shape[0] / 2 + 0.5]
    w.wcs.cd = (scale / 3600.0) * np.array([[-1.0, 0.0], [0.0, 1.0]])
    return w


def corners(w: WCS, shape: tuple[int, int]) -> np.ndarray:
    """World coordinates of the four pixel-grid corners of a ``shape`` image."""
    ny, nx = shape
    px = [(0, 0), (nx, 0), (nx, ny), (0, ny)]
    return np.array([w.wcs_pix2world([[x, y]], 0)[0] for x, y in px], dtype=float)
