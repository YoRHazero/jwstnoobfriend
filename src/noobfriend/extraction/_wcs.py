"""Internal helpers for world<->detector transforms on a JWST GWCS.

The two non-obvious things this module isolates from the rest of the package:

1. The forward/inverse transform pair (``world -> detector`` and
   ``detector -> world``) for a :class:`gwcs.WCS`.
2. The JWST WFSS *grism* special case, where those frames are not reachable
   with a plain ``(ra, dec)`` / ``(x, y)`` call but require extra spectral
   inputs. Keeping the magic numbers here means no other module has to know
   the grism dispersion convention.
"""

from typing import Any, Callable

from astropy.coordinates import SkyCoord
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
