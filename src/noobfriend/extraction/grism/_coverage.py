"""Decide which exposures cover a source's spectral trace, cheaply.

Coverage is a *pre-load* step: it needs only each exposure's WCS and detector
shape (lightweight header metadata), never the pixel data. Given a sky
position, it reports — per exposure — whether the order-1 trace lands on the
detector, over which wavelength interval, and along which axis it disperses.
The caller then loads ``data``/``error`` for the covered exposures only.

This module is internal; :class:`FrameMeta`, :class:`FrameCoverage`,
:func:`find_coverage`, and the :data:`Dispersion` alias are re-exported from
``noobfriend.extraction.grism``.
"""

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from gwcs import WCS

from noobfriend.extraction._wcs import (
    grism_trace_transform,
    world_detector_transforms,
)
from noobfriend.extraction.grism._wavelength import read_wavelength_domain

#: The detector axis a grism order disperses along: ``"row"`` (dispersion runs
#: along x, NIRCam GRISMR / NIRISS GR150R) or ``"column"`` (along y, GRISMC /
#: GR150C). Determined from the local trace tangent, so it is always defined —
#: it is a property of the grism's orientation, independent of whether the
#: source's trace happens to fall on the detector.
Dispersion = Literal["row", "column"]


@dataclass(frozen=True)
class FrameMeta:
    """Lightweight description of one WFSS exposure (no pixel data).

    Attributes
    ----------
    id : str
        Caller-chosen identifier, used to key data and outputs. Typically the
        file stem (e.g. ``pathlib.Path(path).stem``).
    wcs : gwcs.WCS
        The exposure's grism WCS.
    shape : tuple[int, int]
        Detector array shape ``(H, W)``, from the FITS header. Needed for the
        on-array test, since the WCS carries no bounding box.
    wavelength_range : tuple[float, float] | None, optional
        Per-exposure wavelength override in microns. Highest priority in the
        range resolution (see :func:`find_coverage`). ``None`` defers to the
        shared range or to automatic extraction.
    group : collections.abc.Hashable | None, optional
        Optional grouping key. Spectra are combined only within the same ``group``,
        so set it to keep apart frames that should not be co-added — e.g. NIRCam
        modules A/B, or GRISMR vs GRISMC if you want them separate (see
        :attr:`FrameCoverage.dispersion`). ``None`` (default) combines everything
        into a single spectrum.
    """

    id: str
    wcs: WCS
    shape: tuple[int, int]
    wavelength_range: tuple[float, float] | None = None
    group: Hashable | None = None


@dataclass(frozen=True)
class FrameCoverage:
    """Per-exposure result of :func:`find_coverage`.

    Attributes
    ----------
    meta : FrameMeta
        The input exposure this result describes (carries ``id``/``wcs``/
        ``shape``).
    covered : bool
        Whether the order-1 trace falls on the detector by at least
        ``min_fraction`` of its usable length.
    wavelength_range : tuple[float, float] | None
        The resolved wavelength interval in microns, after intersecting the
        requested range with the on-array span. ``None`` when not covered.
    dispersion : Dispersion
        Dispersion axis, inferred from the local trace tangent. Always defined.
    source_xy : tuple[float, float]
        The source's undispersed direct-image position ``(x0, y0)``.
    """

    meta: FrameMeta
    covered: bool
    wavelength_range: tuple[float, float] | None
    dispersion: Dispersion
    source_xy: tuple[float, float]


def find_coverage(
    ra: float,
    dec: float,
    frames: Sequence[FrameMeta],
    *,
    wavelength_range: tuple[float, float] | None = None,
    min_fraction: float = 0.0,
) -> list[FrameCoverage]:
    """Report, per exposure, whether and where it covers a source's trace.

    For each exposure the wavelength interval is resolved by priority:

    1. ``frame.wavelength_range`` (per-exposure override), else
    2. the shared ``wavelength_range`` argument, else
    3. :func:`~noobfriend.extraction.grism.read_wavelength_domain` on that
       exposure's WCS (the only path that reads the dispersion model).

    The resolved interval is then intersected with the span over which the
    order-1 trace actually lands on the detector (using ``frame.shape``). A
    user-supplied interval is taken **at face value** — it is not clamped to the
    model's valid domain — so requesting wavelengths outside that domain is the
    caller's responsibility (the trace freezes there and yields meaningless,
    on-array pixels).

    Parameters
    ----------
    ra, dec : float
        Source world coordinates in degrees.
    frames : Sequence[FrameMeta]
        Exposures to test (WCS + shape only; no pixel data loaded).
    wavelength_range : tuple[float, float] | None, optional
        Shared wavelength interval in microns, used for any exposure without its
        own override. ``None`` triggers per-exposure automatic extraction.
    min_fraction : float, optional
        Minimum fraction of the trace's usable length that must fall on the
        detector for ``covered`` to be ``True``, by default ``0.0`` (any
        overlap counts).

    Returns
    -------
    list[FrameCoverage]
        One entry per input exposure, in input order. Filter on ``covered`` to
        select exposures to load and pass to
        :meth:`~noobfriend.extraction.grism.GrismSpectrum.from_world`.

    Raises
    ------
    NotImplementedError
        If automatic extraction is needed (no override and no shared range) but
        the dispersion model class is unrecognised. Pass an explicit
        ``wavelength_range`` in that case.
    """
    results: list[FrameCoverage] = []
    for frame in frames:
        world_to_detector, _ = world_detector_transforms(frame.wcs)
        x0, y0 = world_to_detector(ra, dec)
        source_xy = (float(x0), float(y0))

        trace = grism_trace_transform(frame.wcs)

        wr = (
            frame.wavelength_range
            or wavelength_range
            or read_wavelength_domain(frame.wcs, ra, dec, order=1)
        )
        lo, hi = wr

        xg_lo, yg_lo = trace(x0, y0, lo)
        xg_hi, yg_hi = trace(x0, y0, hi)
        dx = float(xg_hi) - float(xg_lo)
        dy = float(yg_hi) - float(yg_lo)
        dispersion: Dispersion = "row" if abs(dx) >= abs(dy) else "column"

        height, width = frame.shape
        lam = np.linspace(lo, hi, 200)
        xg, yg = trace(
            np.full_like(lam, x0),
            np.full_like(lam, y0),
            lam,
        )
        # The NIRCam dispersion model broadcasts its three equal-length inputs
        # outer-product style, returning a 2-D grid whose diagonal holds the
        # per-wavelength trace pixels; recover that 1-D sequence.
        xg = np.asarray(xg)
        yg = np.asarray(yg)
        if xg.ndim == 2:
            xg = np.diagonal(xg)
            yg = np.diagonal(yg)
        mask = (xg >= 0.0) & (xg < width) & (yg >= 0.0) & (yg < height)

        on_array_fraction = float(mask.mean())
        covered = bool(mask.any()) and on_array_fraction >= min_fraction

        wavelength_range_out: tuple[float, float] | None = None
        if covered:
            covered_lo = float(lam[mask].min())
            covered_hi = float(lam[mask].max())
            inter_lo = max(wr[0], covered_lo)
            inter_hi = min(wr[1], covered_hi)
            if inter_lo <= inter_hi:
                wavelength_range_out = (inter_lo, inter_hi)

        results.append(
            FrameCoverage(
                meta=frame,
                covered=covered,
                wavelength_range=wavelength_range_out,
                dispersion=dispersion,
                source_xy=source_xy,
            )
        )

    return results
