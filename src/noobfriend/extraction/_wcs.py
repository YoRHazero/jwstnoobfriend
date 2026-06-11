"""Internal helpers for world<->detector transforms on a JWST GWCS.

The two non-obvious things this module isolates from the rest of the package:

1. The forward/inverse transform pair (``world -> detector`` and
   ``detector -> world``) for a :class:`gwcs.WCS`.
2. The JWST WFSS *grism* special case, where those frames are not reachable
   with a plain ``(ra, dec)`` / ``(x, y)`` call but require extra spectral
   inputs. Keeping the magic numbers here means no other module has to know
   the grism dispersion convention.
"""

from dataclasses import dataclass
from typing import Any, Callable

from astropy.coordinates import SkyCoord
from astropy.wcs import WCS as AstropyWCS
from gwcs import WCS

#: A 2-D coordinate transform: ``f(a, b) -> (c, d)``. Inputs and outputs may be
#: scalars or broadcastable arrays (e.g. when transforming a whole meshgrid).
Transform = Callable[[Any, Any], tuple[Any, Any]]


def world_detector_transforms(wcs: WCS) -> tuple[Transform, Transform]:
    """Return the ``(world_to_detector, detector_to_world)`` transform pair.

    Parameters
    ----------
    wcs : gwcs.WCS
        The world-coordinate system to build transforms from.

    Returns
    -------
    world_to_detector : Transform
        Maps ``(ra, dec)`` in degrees to ``(x, y)`` detector pixels.
    detector_to_world : Transform
        Maps ``(x, y)`` detector pixels to ``(ra, dec)`` in degrees.

    Notes
    -----
    When the WCS exposes a ``grism_detector`` frame (JWST WFSS products), the
    underlying transforms additionally consume a spectral order and a source
    identifier. The wrappers below supply the fixed values ``(4, 1)`` and keep
    only the first two outputs, reproducing the dispersion convention used by
    the upstream pipeline. Non-grism products use the transforms unchanged.
    """
    is_grism = "grism_detector" in wcs.available_frames
    if is_grism:
        world_to_detector_raw = wcs.get_transform("world", "detector")
        detector_to_world_raw = wcs.get_transform("detector", "world")

        def world_to_detector(ra: Any, dec: Any) -> tuple[Any, Any]:
            return world_to_detector_raw(ra, dec, 4, 1)[0:2]

        def detector_to_world(x: Any, y: Any) -> tuple[Any, Any]:
            return detector_to_world_raw(x, y, 4, 1)[0:2]

        return world_to_detector, detector_to_world

    return wcs.get_transform("world", "detector"), wcs.get_transform(
        "detector", "world"
    )


def grism_trace_transform(
    wcs: WCS, *, order: int = 1
) -> Callable[[Any, Any, Any], tuple[Any, Any]]:
    """Return the grism trace transform for a spectral order.

    Wraps the ``detector -> grism_detector`` transform of a JWST WFSS WCS so it
    can be called as ``f(x0, y0, wavelength) -> (x_grism, y_grism)``, hiding the
    extra spectral-order input and the trailing pass-through outputs of the
    underlying model.

    Parameters
    ----------
    wcs : gwcs.WCS
        A JWST WFSS grism WCS (one exposing a ``grism_detector`` frame).
    order : int, optional
        Spectral order to trace, by default ``1``. NIRCam grisms have no 0th
        order and place order 2 largely off the detector, so order 1 is the
        only one normally needed.

    Returns
    -------
    Callable[[Any, Any, Any], tuple[Any, Any]]
        ``f(x0, y0, wavelength) -> (x_grism, y_grism)`` mapping a source's
        undispersed detector position ``(x0, y0)`` and a wavelength in microns to
        the dispersed pixel position of the trace.

    Notes
    -----
    The raw ``detector -> grism_detector`` transform takes
    ``(x0, y0, wavelength, order)`` and returns
    ``(x_grism, y_grism, x0, y0, order)``; the wrapper fixes ``order`` and keeps
    only the first two outputs. Inputs may be scalars or broadcastable arrays.
    The underlying NIRCam model requires ``x0``, ``y0``, and ``wavelength`` to
    share the same shape when given arrays; ``order`` stays a scalar. When all
    three are passed as equal-length 1-D arrays the model broadcasts them
    outer-product style, returning a 2-D grid whose diagonal holds the
    element-wise result (see :func:`~noobfriend.extraction.grism.source_locus`).
    """
    raw = wcs.get_transform("detector", "grism_detector")

    def trace(x0: Any, y0: Any, wavelength: Any) -> tuple[Any, Any]:
        return raw(x0, y0, wavelength, order)[:2]

    return trace


def pixel_scale_per_deg(
    detector_to_world: Transform, x_index: int, y_index: int
) -> tuple[float, float]:
    """Estimate local pixels-per-degree along the x and y axes.

    Uses a one-pixel finite difference about ``(x_index, y_index)`` and the
    great-circle separation between neighbouring pixel centres, so the result
    is correct even where the plate scale varies or the axes are not aligned
    with RA/Dec.

    Parameters
    ----------
    detector_to_world : Transform
        Detector-to-world transform, e.g. from :func:`world_detector_transforms`.
    x_index, y_index : int
        Integer pixel about which to evaluate the local scale.

    Returns
    -------
    tuple[float, float]
        ``(pixels_per_deg_x, pixels_per_deg_y)``. A component is ``0.0`` if the
        corresponding one-pixel step maps to a zero on-sky separation.
    """
    center_ra, center_dec = detector_to_world(x_index, y_index)
    next_x_ra, next_x_dec = detector_to_world(x_index + 1, y_index)
    next_y_ra, next_y_dec = detector_to_world(x_index, y_index + 1)

    center = SkyCoord(center_ra, center_dec, unit="deg")
    deg_per_pixel_x = center.separation(SkyCoord(next_x_ra, next_x_dec, unit="deg")).deg
    deg_per_pixel_y = center.separation(SkyCoord(next_y_ra, next_y_dec, unit="deg")).deg

    pixel_per_deg_x = 1.0 / deg_per_pixel_x if deg_per_pixel_x > 0 else 0.0
    pixel_per_deg_y = 1.0 / deg_per_pixel_y if deg_per_pixel_y > 0 else 0.0
    return pixel_per_deg_x, pixel_per_deg_y


@dataclass(frozen=True)
class _AstropyWCSAdapter:
    """Expose an astropy FITS WCS through noobfriend's transform protocol."""

    wcs: AstropyWCS
    available_frames: tuple[str, str] = ("detector", "world")

    def get_transform(self, from_frame: str, to_frame: str) -> Transform:
        """Return the detector/world transform using astropy's 0-based API."""
        if (from_frame, to_frame) == ("world", "detector"):
            return self.wcs.world_to_pixel_values
        if (from_frame, to_frame) == ("detector", "world"):
            return self.wcs.pixel_to_world_values
        raise ValueError(f"Unsupported transform: {from_frame!r} -> {to_frame!r}.")


def tangent_plane_wcs(
    ra: float, dec: float, size_arcsec: float, shape: tuple[int, int]
) -> _AstropyWCSAdapter:
    """Synthesise an axis-aligned tangent-plane WCS for a WCS-less cutout.

    Builds a ``TAN`` WCS centred on ``(ra, dec)`` -- the source is taken to sit
    at the image centre -- whose pixel scale follows from the cutout spanning
    ``size_arcsec`` along each axis over ``shape`` pixels. It makes a band that
    carries no ``wcs`` behave like any real one for seeding, pixel-scale
    estimation, and mask reprojection; see
    :func:`noobfriend.extraction.photometry._band.normalize_band`.

    Parameters
    ----------
    ra, dec : float
        Source / cutout-centre world coordinate in degrees (ICRS).
    size_arcsec : float
        Full angular size of the cutout along each axis, in arcseconds.
    shape : tuple[int, int]
        Image ``(rows, cols)`` shape.

    Returns
    -------
    _AstropyWCSAdapter
        A WCS adapter exposing the detector/world transform protocol, so it is a
        drop-in for the WCS objects the rest of the package consumes.

    Notes
    -----
    The synthesised WCS is axis-aligned (no rotation) and centred on the source,
    so multiple WCS-less bands are taken to be co-centred cutouts of the *same*
    source -- correct for typical single-source stamps, but not for bands that
    are not co-centred (give those a real WCS).
    """
    n_rows, n_cols = shape
    size_deg = float(size_arcsec) / 3600.0
    wcs = AstropyWCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [float(ra), float(dec)]
    wcs.wcs.crpix = [(n_cols + 1) / 2.0, (n_rows + 1) / 2.0]  # 1-based image centre
    wcs.wcs.cdelt = [-size_deg / n_cols, size_deg / n_rows]
    return _AstropyWCSAdapter(wcs)
