"""Normalized internal representation of one photometry band.

Callers pass each band as a plain mapping (``"data"`` / ``"wcs"`` plus optional
extras); :func:`normalize_band` validates that mapping once and freezes it into
a :class:`Band`. Keeping the public surface dictionary-based means producers
such as :func:`noobfriend.core.io.load_grizli_cutout` do not have to import or
construct any photometry type.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np

from noobfriend.core.imgutils import NoiseAutocovariance


class BandSpec(TypedDict, total=False):
    """One photometry band as a plain dict, with statically-checked keys.

    Because this is a :class:`~typing.TypedDict`, a dict literal passed to
    :class:`~noobfriend.extraction.photometry.ApertureSED` is checked by editors
    for key names and value types -- no import or construction needed. ``data``
    is the only required key (enforced at runtime by :func:`normalize_band`);
    ``wcs`` is optional -- omit it to have a tangent-plane WCS synthesised from
    the source position and ``ApertureSED``'s ``cutout_size_arcsec``.

    Keys
    ----
    data : numpy.ndarray
        2-D science image on the band's native grid (required).
    wcs : Any
        WCS exposing the noobfriend transform protocol; omit to synthesise one.
    error : numpy.ndarray or None
        1-sigma error image.
    wavelength : float or None
        Effective wavelength in microns.
    wavelength_error : tuple[float, ...] or None
        Left/right wavelength uncertainty (a tuple of length 1 or 2).
    flux_scale_mjy : float or None
        Multiplicative scale from one raw per-pixel value to mJy.
    flux_unit : str or None
        Original image unit or calibration source behind the scale.
    label_map : numpy.ndarray or None
        Segmentation labels restricting aperture growth.
    label_allowed : sequence of int or None
        Labels from ``label_map`` allowed for aperture growth.
    allow_background : bool
        Whether growth may extend off the seed's segment into the background.
    noise_kernel : NoiseAutocovariance or None
        The band's field-level noise autocovariance ``C(d)`` on **this band's own
        native grid** (e.g. ``NooBook.noise_kernel`` of the stage-3 product the
        band was cut from). Used as the correlation-aware aperture-error fallback
        when the cutout has too little blank sky to estimate ``C(d)`` locally.
    """

    data: np.ndarray
    wcs: Any
    error: np.ndarray | None
    wavelength: float | None
    wavelength_error: tuple[float, ...] | None
    flux_scale_mjy: float | None
    flux_unit: str | None
    label_map: np.ndarray | None
    label_allowed: Sequence[int] | None
    allow_background: bool
    noise_kernel: NoiseAutocovariance | None


#: Keys :func:`normalize_band` accepts; anything else is treated as a typo.
_BAND_KEYS: frozenset[str] = frozenset(BandSpec.__annotations__)


@dataclass(frozen=True)
class Band:
    """One photometry band on its own native pixel grid.

    Attributes
    ----------
    name : str
        Band name.
    data : numpy.ndarray
        2-D science image on the band's native grid.
    wcs : Any
        World-coordinate system exposing the noobfriend transform protocol.
    error : numpy.ndarray or None
        1-sigma error image, or ``None``.
    wavelength : float or None
        Effective wavelength in microns, or ``None``.
    wavelength_error : tuple[float, float] or None
        Left/right wavelength uncertainty, or ``None``.
    flux_scale_mjy : float or None
        Multiplicative scale from one raw per-pixel value to mJy, or ``None``.
    flux_unit : str or None
        Original image unit or calibration source behind the scale.
    label_map : numpy.ndarray or None
        Segmentation labels restricting aperture growth, or ``None``.
    label_allowed : tuple[int, ...] or None
        Labels allowed for aperture growth, or ``None``.
    allow_background : bool
        When a ``label_map`` is present, whether aperture growth may extend off
        the source's own segment into the background label ``0`` (blocked only
        by other sources). ``True`` by default; ignored without a ``label_map``.
    noise_kernel : NoiseAutocovariance or None
        Field-level noise autocovariance on this band's native grid, or ``None``.
        The correlation-aware fallback for the aperture error when the cutout has
        too little local sky to estimate ``C(d)`` (see
        :func:`~noobfriend.extraction.photometry._noise.measure_aperture_noise`).
    """

    name: str
    data: np.ndarray
    wcs: Any
    error: np.ndarray | None
    wavelength: float | None
    wavelength_error: tuple[float, float] | None
    flux_scale_mjy: float | None
    flux_unit: str | None
    label_map: np.ndarray | None
    label_allowed: tuple[int, ...] | None
    allow_background: bool
    noise_kernel: NoiseAutocovariance | None

    @property
    def shape(self) -> tuple[int, int]:
        """Return the native ``(rows, cols)`` image shape."""
        return self.data.shape  # type: ignore[return-value]


def normalize_band(
    name: str,
    spec: Mapping[str, Any],
    *,
    ra: float | None = None,
    dec: float | None = None,
    cutout_size_arcsec: float | None = None,
) -> Band:
    """Validate one public band spec and freeze it into a :class:`Band`.

    Parameters
    ----------
    name : str
        Band name.
    spec : mapping
        A :class:`BandSpec`-shaped mapping: required ``"data"`` plus optional
        ``"wcs"``, ``"error"``, ``"wavelength"``, ``"wavelength_error"``,
        ``"flux_scale_mjy"``, ``"flux_unit"``, ``"label_map"``,
        ``"label_allowed"``, and ``"allow_background"`` (default ``True``).
        Unknown keys are rejected. ``wavelength_error`` must be a tuple of length
        1 or 2; ``(value,)`` is read as symmetric.
    ra, dec : float, optional
        Source world coordinate, used only to synthesise a WCS when ``spec`` has
        no ``"wcs"``.
    cutout_size_arcsec : float, optional
        Cutout angular size, used only to synthesise a WCS when ``spec`` has no
        ``"wcs"`` (see :func:`noobfriend.extraction._wcs.tangent_plane_wcs`).

    Returns
    -------
    Band

    Raises
    ------
    ValueError
        If an unknown key is present, ``"data"`` is missing or malformed, or
        ``"wcs"`` is absent and no ``cutout_size_arcsec`` is available to
        synthesise one.
    """
    unknown = set(spec) - _BAND_KEYS
    if unknown:
        raise ValueError(
            f"band {name!r} has unknown key(s) {sorted(unknown)}; "
            f"allowed keys are {sorted(_BAND_KEYS)}."
        )
    if "data" not in spec:
        raise ValueError(f"band {name!r} is missing required key 'data'.")

    data = np.asarray(spec["data"])
    if data.ndim != 2:
        raise ValueError(f"band {name!r} data must be 2-D; got shape {data.shape}.")

    wcs = spec.get("wcs")
    if wcs is None:
        if cutout_size_arcsec is None:
            raise ValueError(
                f"band {name!r} has no 'wcs'; either provide one, or pass "
                "cutout_size_arcsec to ApertureSED so a tangent-plane WCS can be "
                "synthesised from the source position and the cutout size."
            )
        from noobfriend.extraction._wcs import tangent_plane_wcs

        wcs = tangent_plane_wcs(ra, dec, cutout_size_arcsec, data.shape)

    error = None if spec.get("error") is None else np.asarray(spec["error"])
    if error is not None and error.shape != data.shape:
        raise ValueError(
            f"band {name!r} error shape {error.shape} does not match data "
            f"shape {data.shape}."
        )

    label_map = None if spec.get("label_map") is None else np.asarray(spec["label_map"])
    if label_map is not None and label_map.shape != data.shape:
        raise ValueError(
            f"band {name!r} label_map shape {label_map.shape} does not match "
            f"data shape {data.shape}."
        )

    label_allowed = spec.get("label_allowed")
    label_allowed_out = (
        None if label_allowed is None else tuple(int(label) for label in label_allowed)
    )

    wavelength = None if spec.get("wavelength") is None else float(spec["wavelength"])
    wavelength_error = _normalize_wavelength_error(name, spec.get("wavelength_error"))
    if wavelength_error is not None and wavelength is None:
        raise ValueError(f"band {name!r} provides wavelength_error but no wavelength.")

    flux_scale_mjy = (
        None if spec.get("flux_scale_mjy") is None else float(spec["flux_scale_mjy"])
    )
    if flux_scale_mjy is not None and not np.isfinite(flux_scale_mjy):
        raise ValueError(f"band {name!r} flux_scale_mjy must be finite.")
    flux_unit = None if spec.get("flux_unit") is None else str(spec["flux_unit"])

    allow_background = spec.get("allow_background", True)

    noise_kernel = spec.get("noise_kernel")
    if noise_kernel is not None and not isinstance(noise_kernel, NoiseAutocovariance):
        raise ValueError(
            f"band {name!r} noise_kernel must be a NoiseAutocovariance; "
            f"got {type(noise_kernel).__name__}."
        )

    return Band(
        name=str(name),
        data=data,
        wcs=wcs,
        error=error,
        wavelength=wavelength,
        wavelength_error=wavelength_error,
        flux_scale_mjy=flux_scale_mjy,
        flux_unit=flux_unit,
        label_map=label_map,
        label_allowed=label_allowed_out,
        allow_background=bool(allow_background),
        noise_kernel=noise_kernel,
    )


def _normalize_wavelength_error(band: str, value: object) -> tuple[float, float] | None:
    """Normalize a wavelength-error tuple to ``(left, right)``."""
    if value is None:
        return None
    if not isinstance(value, tuple):
        raise ValueError(
            f"band {band!r} wavelength_error must be a tuple of length 1 or 2."
        )
    if len(value) == 1:
        left = right = float(value[0])
    elif len(value) == 2:
        left, right = float(value[0]), float(value[1])
    else:
        raise ValueError(
            f"band {band!r} wavelength_error must have length 1 or 2; got {len(value)}."
        )
    if not (np.isfinite(left) and np.isfinite(right) and left >= 0 and right >= 0):
        raise ValueError(
            f"band {band!r} wavelength_error values must be finite and non-negative."
        )
    return (left, right)
