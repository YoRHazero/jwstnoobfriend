"""Read a grism order's calibrated wavelength domain from its GWCS model.

A JWST WFSS dispersion model describes the spectral trace with a dimensionless
*trace parameter* ``t`` that runs from one end of the trace (``t = 0``) to the
other (``t = 1``). Wavelength, and the trace's pixel displacement, are all
polynomials in ``t`` (with coefficients that depend on the source's
direct-image position). The polynomials are only fit over ``t in [0, 1]``, so
the wavelength interval ``[lambda(t=0), lambda(t=1)]`` *is* the range over which
the dispersion solution is defined and trustworthy — its **valid domain**.

:func:`read_wavelength_domain` digs the dispersion model out of the compound
GWCS transform and evaluates that interval at the source position. It is the
authoritative replacement for sweeping wavelengths and watching where the
trace pixel "freezes" (the model clamps ``t`` to ``[0, 1]`` outside the domain).

This module is internal; :func:`read_wavelength_domain` is re-exported from
``noobfriend.extraction.grism``.
"""

from typing import Any

from astropy.modeling import Model
from gwcs import WCS

from noobfriend.extraction._wcs import world_detector_transforms


def _find_dispersion_model(model: Model, class_name: str) -> Model | None:
    """Recursively locate a submodel by its class name within a compound model.

    Walks the binary tree of an astropy ``CompoundModel`` via its ``.left`` and
    ``.right`` attributes, returning the first leaf whose
    ``type(m).__name__`` equals ``class_name``, or ``None`` if no such submodel
    exists.

    Parameters
    ----------
    model : astropy.modeling.Model
        The (possibly compound) model to search.
    class_name : str
        The target submodel's class name, e.g. ``"NIRCAMBackwardGrismDispersion"``.

    Returns
    -------
    astropy.modeling.Model or None
        The matching submodel, or ``None`` if not found.
    """
    if type(model).__name__ == class_name:
        return model
    left = getattr(model, "left", None)
    if left is not None:
        found = _find_dispersion_model(left, class_name)
        if found is not None:
            return found
    right = getattr(model, "right", None)
    if right is not None:
        found = _find_dispersion_model(right, class_name)
        if found is not None:
            return found
    return None


def read_wavelength_domain(
    wcs: WCS,
    ra: float,
    dec: float,
    *,
    order: int = 1,
    t_range: tuple[float, float] = (0.0, 1.0),
) -> tuple[float, float]:
    """Return the calibrated wavelength domain of a grism order.

    Reads the dispersion model's wavelength polynomial and evaluates it at the
    trace-parameter endpoints ``t_range`` (by default the full fit domain
    ``t = 0`` to ``t = 1``), at the direct-image position of ``(ra, dec)``. The
    default result is the wavelength interval over which the trace/dispersion
    solution was fit; sampling the trace outside it is meaningless (the model
    clamps to the endpoints).

    This does real work — it locates the grism dispersion submodel inside the
    compound GWCS transform and evaluates astropy polynomials — so it is meant
    to be called once and its result reused (e.g. passed as the shared
    ``wavelength_range`` to :func:`~noobfriend.extraction.grism.find_coverage`),
    not invoked per frame.

    Parameters
    ----------
    wcs : gwcs.WCS
        A JWST WFSS grism WCS (one exposing a ``grism_detector`` frame).
    ra, dec : float
        Source world coordinates in degrees. The valid domain depends weakly on
        the source's direct-image position, so it is evaluated there.
    order : int, optional
        Spectral order whose domain to read, by default ``1``.
    t_range : tuple[float, float], optional
        Trace-parameter endpoints to evaluate, by default ``(0.0, 1.0)`` (the
        full fit domain). Pass a sub-range (e.g. ``(0.05, 0.90)``) to get the
        wavelengths bounding a trimmed, well-behaved interior of the trace,
        away from the band edges where the polynomial inverse turns unreliable.

    Returns
    -------
    tuple[float, float]
        ``(lambda_min, lambda_max)`` in microns, sorted ascending.

    Raises
    ------
    ValueError
        If ``order`` is not present in the dispersion model.
    NotImplementedError
        If the dispersion model class is not recognised. Recognised models are
        tried NIRCam-first, then NIRISS; for any other instrument the caller
        must supply the wavelength range explicitly.

    Notes
    -----
    The trace parameter convention ``t in [0, 1]`` is shared by the JWST NIRCam
    and NIRISS grism dispersion models, but the concrete model classes differ,
    so the extraction of the wavelength polynomial is instrument-specific.
    """
    world_to_detector, _ = world_detector_transforms(wcs)
    x0, y0 = world_to_detector(ra, dec)

    transform = wcs.get_transform("detector", "grism_detector")
    model = _find_dispersion_model(transform, "NIRCAMBackwardGrismDispersion")
    if model is None:
        raise NotImplementedError(
            "Unrecognised grism dispersion model; supply the wavelength range "
            "explicitly."
        )

    orders: list[int] = list(model.orders)
    if order not in orders:
        raise ValueError(
            f"Spectral order {order} not present in dispersion model "
            f"(available orders: {orders})."
        )
    idx = orders.index(order)
    coeffs: list[Any] = model.lmodels[idx]

    t_lo, t_hi = t_range
    lam_lo = sum(float(coeff(x0, y0)) * (t_lo**k) for k, coeff in enumerate(coeffs))
    lam_hi = sum(float(coeff(x0, y0)) * (t_hi**k) for k, coeff in enumerate(coeffs))
    return tuple(sorted((lam_lo, lam_hi)))
