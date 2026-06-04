"""High-level multi-band aperture photometry and SED results."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from noobfriend.extraction._wcs import pixel_scale_per_deg, world_detector_transforms
from noobfriend.extraction.photometry._aperture import (
    ApertureMask,
    grow_aperture_mask,
)
from noobfriend.extraction.photometry._array import measurement_image, native_float64
from noobfriend.extraction.photometry._reproject import reproject_image, reproject_mask


@dataclass(frozen=True)
class _BandFrame:
    """Normalized internal representation of one photometry band."""

    name: str
    data: np.ndarray
    wcs: Any
    error: np.ndarray | None
    wavelength: float | None
    wavelength_error: tuple[float, float] | None
    label_map: np.ndarray | None
    label_allowed: tuple[int, ...] | None


@dataclass(frozen=True)
class BandPhotometry:
    """Photometry for one band measured through the union aperture.

    Attributes
    ----------
    band : str
        Band name.
    wavelength : float or None
        Band effective wavelength, if supplied.
    wavelength_error : tuple[float, float] or None
        Left/right wavelength uncertainty, if supplied.
    flux : float
        Sum of valid pixels inside the union aperture.
    error : float or None
        Quadrature sum of per-pixel errors inside the union aperture, or
        ``None`` if no error image was supplied for the band.
    n_pixels : int
        Number of reference-grid pixels in the union aperture.
    n_valid_pixels : int
        Number of those pixels with finite, non-zero band data.
    bad_fraction : float
        Fraction of union-aperture pixels that are ``NaN`` or strict ``0.0`` in
        this band after alignment.
    flagged : bool
        ``True`` when :attr:`bad_fraction` is greater than or equal to the
        configured threshold.
    flag_reason : str or None
        Machine-readable reason for :attr:`flagged`.
    """

    band: str
    wavelength: float | None
    wavelength_error: tuple[float, float] | None
    flux: float
    error: float | None
    n_pixels: int
    n_valid_pixels: int
    bad_fraction: float
    flagged: bool
    flag_reason: str | None


@dataclass(frozen=True)
class ApertureSEDResult:
    """Measured SED plus aperture masks on the reference grid.

    Attributes
    ----------
    measurements : tuple[BandPhotometry, ...]
        Per-band photometry results, sorted by wavelength when all measured
        bands have wavelengths, otherwise in input band order.
    reference_band : str
        Band whose pixel grid was used as the common reference.
    union_mask : numpy.ndarray
        Boolean union aperture on the reference grid.
    band_masks : mapping
        Per-band aperture masks after alignment to the reference grid.
    source_apertures : mapping
        Per-band noobase aperture results on each source band's native grid.
    """

    measurements: tuple[BandPhotometry, ...]
    reference_band: str
    union_mask: np.ndarray
    band_masks: Mapping[str, np.ndarray]
    source_apertures: Mapping[str, ApertureMask]

    def to_table(self) -> Any:
        """Return the SED measurements as an :class:`astropy.table.Table`.

        Returns
        -------
        astropy.table.Table
            Table with one row per band.
        """
        from astropy.table import Table

        rows = [
            {
                "band": m.band,
                "wavelength": m.wavelength,
                "wavelength_error_left": None
                if m.wavelength_error is None
                else m.wavelength_error[0],
                "wavelength_error_right": None
                if m.wavelength_error is None
                else m.wavelength_error[1],
                "flux": m.flux,
                "error": m.error,
                "n_pixels": m.n_pixels,
                "n_valid_pixels": m.n_valid_pixels,
                "bad_fraction": m.bad_fraction,
                "flagged": m.flagged,
                "flag_reason": m.flag_reason,
            }
            for m in self.measurements
        ]
        return Table(rows=rows)

    def plot(
        self,
        *,
        include_flagged: bool = True,
        title: str | None = None,
        display_plot: bool = True,
        height: int = 500,
    ) -> Any:
        """Plot the measured SED with flux and wavelength error bars.

        Parameters
        ----------
        include_flagged : bool, default True
            Include bands flagged for bad pixels. Flagged points are shown in a
            contrasting colour.
        title : str, optional
            Plot title. Defaults to ``"Aperture SED"``.
        display_plot : bool, default True
            When ``True``, return the notebook-display wrapper used by the
            package's plotting helpers. When ``False``, return the raw Bokeh
            figure for tests or further customization.
        height : int, default 500
            Display iframe height when ``display_plot`` is ``True``.

        Returns
        -------
        Any
            A notebook-displayable object or a Bokeh figure.

        Raises
        ------
        ValueError
            If no plotted band has a wavelength.
        """
        from bokeh.models import ColumnDataSource, HoverTool, Whisker
        from bokeh.plotting import figure

        from noobfriend.core.display.plot._bokeh import display

        measurements = [
            m
            for m in self.measurements
            if m.wavelength is not None and (include_flagged or not m.flagged)
        ]
        if not measurements:
            raise ValueError("Cannot plot SED: no included band has a wavelength.")
        measurements = sorted(measurements, key=lambda m: float(m.wavelength))

        x = np.asarray([m.wavelength for m in measurements], dtype=float)
        y = np.asarray([m.flux for m in measurements], dtype=float)
        yerr = np.asarray(
            [np.nan if m.error is None else m.error for m in measurements],
            dtype=float,
        )
        xerr_left = np.asarray(
            [
                np.nan if m.wavelength_error is None else m.wavelength_error[0]
                for m in measurements
            ],
            dtype=float,
        )
        xerr_right = np.asarray(
            [
                np.nan if m.wavelength_error is None else m.wavelength_error[1]
                for m in measurements
            ],
            dtype=float,
        )
        source = ColumnDataSource(
            {
                "band": [m.band for m in measurements],
                "wavelength": x,
                "wave_lower": x - xerr_left,
                "wave_upper": x + xerr_right,
                "flux": y,
                "flux_lower": y - yerr,
                "flux_upper": y + yerr,
                "bad_fraction": [m.bad_fraction for m in measurements],
                "flagged": [m.flagged for m in measurements],
                "color": ["#c43b3b" if m.flagged else "#1f6f8b" for m in measurements],
            }
        )

        p = figure(
            title=title or "Aperture SED",
            x_axis_label="Wavelength",
            y_axis_label="Flux",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            height=420,
            width=680,
        )
        p.line("wavelength", "flux", source=source, color="#5f6b73", line_width=1)
        p.scatter(
            "wavelength",
            "flux",
            source=source,
            marker="circle",
            size=9,
            fill_color="color",
            line_color="white",
            line_width=1,
        )
        p.add_layout(
            Whisker(
                source=source,
                base="wavelength",
                lower="flux_lower",
                upper="flux_upper",
                line_color="#5f6b73",
            )
        )
        p.add_layout(
            Whisker(
                source=source,
                base="flux",
                lower="wave_lower",
                upper="wave_upper",
                dimension="width",
                line_color="#8a9399",
            )
        )
        p.add_tools(
            HoverTool(
                tooltips=[
                    ("band", "@band"),
                    ("wavelength", "@wavelength"),
                    ("flux", "@flux"),
                    ("bad fraction", "@bad_fraction"),
                    ("flagged", "@flagged"),
                ]
            )
        )
        if not display_plot:
            return p
        return display(p, height=height)


class ApertureSED:
    """Build an SED by unioning per-band grown apertures.

    Parameters
    ----------
    bands : mapping
        ``{band_name: spec}``, where each ``spec`` is a plain mapping with
        required ``"data"`` and ``"wcs"`` entries, plus optional ``"error"``,
        ``"wavelength"``, ``"wavelength_error"``, ``"label_map"``, and
        ``"label_allowed"``. ``wavelength_error`` must be a tuple of length 1
        or 2; ``(value,)`` is interpreted as symmetric.
    reference : {"finest"} or str, default "finest"
        Reference grid for unioning masks and measuring all bands. ``"finest"``
        selects the input band with the smallest local pixel scale near the
        measured source. Any other value must be a band name.
    mask_threshold : float, default 0.5
        Fraction threshold used when reprojecting source masks to the reference
        grid.
    flag_bad_fraction : float, default 0.10
        Bands with at least this fraction of union-aperture pixels invalid
        (``NaN`` or strict ``0.0``) are flagged.
    coarse_step : tuple[int, int], optional
        Passed to the photometry-local reprojection helpers.
    """

    def __init__(
        self,
        bands: Mapping[str, Mapping[str, Any]],
        *,
        reference: str = "finest",
        mask_threshold: float = 0.5,
        flag_bad_fraction: float = 0.10,
        coarse_step: tuple[int, int] | None = None,
    ) -> None:
        self._bands = tuple(_normalize_band(name, spec) for name, spec in bands.items())
        if not self._bands:
            raise ValueError("bands must contain at least one band.")
        self._by_name = {band.name: band for band in self._bands}
        if len(self._by_name) != len(self._bands):
            raise ValueError("band names must be unique.")
        if reference != "finest" and reference not in self._by_name:
            raise ValueError(
                f"reference must be 'finest' or one of {sorted(self._by_name)}; "
                f"got {reference!r}."
            )
        if not (0.0 < mask_threshold <= 1.0):
            raise ValueError("mask_threshold must satisfy 0 < threshold <= 1.")
        if not (0.0 <= flag_bad_fraction <= 1.0):
            raise ValueError("flag_bad_fraction must satisfy 0 <= value <= 1.")
        self.reference = reference
        self.mask_threshold = float(mask_threshold)
        self.flag_bad_fraction = float(flag_bad_fraction)
        self.coarse_step = coarse_step

    def measure(
        self,
        *,
        seed_world: tuple[float, float] | None = None,
        seed_xy_by_band: Mapping[str, tuple[float, float]] | None = None,
        grow_kwargs: Mapping[str, Any] | None = None,
    ) -> ApertureSEDResult:
        """Measure all bands through the union of their grown apertures.

        Exactly one seed form is required:

        - ``seed_world=(ra, dec)`` maps the source through each band's WCS.
        - ``seed_xy_by_band={band: (x, y)}`` supplies band-local pixel seeds.

        Parameters
        ----------
        seed_world : tuple[float, float], optional
            Source world coordinate in degrees.
        seed_xy_by_band : mapping, optional
            Per-band ``(x, y)`` source pixels.
        grow_kwargs : mapping, optional
            Extra keyword arguments forwarded to noobase aperture growth.

        Returns
        -------
        ApertureSEDResult
            SED measurements and aperture masks on the selected reference grid.

        Raises
        ------
        ValueError
            If seed inputs are missing, over-specified, incomplete, or the union
            aperture is empty.
        """
        seeds = self._resolve_seeds(seed_world, seed_xy_by_band)
        reference = self._resolve_reference(seeds)
        source_apertures: dict[str, ApertureMask] = {}
        aligned_masks: dict[str, np.ndarray] = {}

        for band in self._bands:
            aperture = grow_aperture_mask(
                band.data,
                seed_xy=seeds[band.name],
                error=band.error,
                label_map=band.label_map,
                label_allowed=band.label_allowed,
                grow_kwargs=grow_kwargs,
            )
            source_apertures[band.name] = aperture
            if band.name == reference.name:
                aligned = aperture.mask.copy()
            else:
                aligned, _ = reproject_mask(
                    aperture.mask,
                    source_wcs=band.wcs,
                    target_wcs=reference.wcs,
                    target_shape=reference.data.shape,
                    threshold=self.mask_threshold,
                    coarse_step=self.coarse_step,
                )
            aligned_masks[band.name] = aligned

        union_mask = np.logical_or.reduce(list(aligned_masks.values()))
        if not bool(union_mask.any()):
            raise ValueError("Union aperture is empty after mask alignment.")

        measurements = tuple(
            self._measure_band(band, reference, union_mask) for band in self._bands
        )
        if all(m.wavelength is not None for m in measurements):
            measurements = tuple(
                sorted(measurements, key=lambda m: float(m.wavelength))
            )

        return ApertureSEDResult(
            measurements=measurements,
            reference_band=reference.name,
            union_mask=union_mask,
            band_masks=aligned_masks,
            source_apertures=source_apertures,
        )

    def _resolve_seeds(
        self,
        seed_world: tuple[float, float] | None,
        seed_xy_by_band: Mapping[str, tuple[float, float]] | None,
    ) -> dict[str, tuple[float, float]]:
        """Resolve the caller's seed input into per-band pixel seeds."""
        if (seed_world is None) == (seed_xy_by_band is None):
            raise ValueError("Provide exactly one of seed_world or seed_xy_by_band.")
        if seed_world is not None:
            ra, dec = seed_world
            seeds: dict[str, tuple[float, float]] = {}
            for band in self._bands:
                world_to_detector, _ = world_detector_transforms(band.wcs)
                x, y = world_to_detector(ra, dec)
                seeds[band.name] = (float(x), float(y))
            return seeds

        assert seed_xy_by_band is not None
        missing = set(self._by_name) - set(seed_xy_by_band)
        extra = set(seed_xy_by_band) - set(self._by_name)
        if missing or extra:
            raise ValueError(
                f"seed_xy_by_band keys must match band names; missing={missing}, "
                f"extra={extra}."
            )
        return {
            name: (float(seed_xy[0]), float(seed_xy[1]))
            for name, seed_xy in seed_xy_by_band.items()
        }

    def _resolve_reference(
        self, seeds: Mapping[str, tuple[float, float]]
    ) -> _BandFrame:
        """Return the reference band for the current measurement."""
        if self.reference != "finest":
            return self._by_name[self.reference]

        def score(band: _BandFrame) -> float:
            _, detector_to_world = world_detector_transforms(band.wcs)
            x, y = seeds[band.name]
            scale_x, scale_y = pixel_scale_per_deg(
                detector_to_world, int(np.round(x)), int(np.round(y))
            )
            positive = [v for v in (scale_x, scale_y) if v > 0]
            return float(np.mean(positive)) if positive else 0.0

        return max(self._bands, key=score)

    def _measure_band(
        self,
        band: _BandFrame,
        reference: _BandFrame,
        union_mask: np.ndarray,
    ) -> BandPhotometry:
        """Measure one band on the reference grid."""
        image = measurement_image(band.data)
        error = None if band.error is None else native_float64(band.error)
        if band.name != reference.name:
            image, error, _, _ = reproject_image(
                image,
                source_wcs=band.wcs,
                target_wcs=reference.wcs,
                target_shape=reference.data.shape,
                error=error,
                coarse_step=self.coarse_step,
            )

        invalid = (~np.isfinite(image)) | (image == 0.0)
        n_pixels = int(union_mask.sum())
        bad_fraction = float(invalid[union_mask].mean()) if n_pixels else 1.0
        valid = union_mask & ~invalid
        n_valid = int(valid.sum())
        flux = float(np.sum(image[valid])) if n_valid else float("nan")

        error_out: float | None
        if error is None:
            error_out = None
        elif n_valid == 0:
            error_out = float("nan")
        else:
            valid_error = valid & np.isfinite(error) & (error >= 0)
            if int(valid_error.sum()) != n_valid:
                error_out = float("nan")
            else:
                error_out = float(np.sqrt(np.sum(error[valid_error] ** 2)))

        flagged = bad_fraction >= self.flag_bad_fraction
        return BandPhotometry(
            band=band.name,
            wavelength=band.wavelength,
            wavelength_error=band.wavelength_error,
            flux=flux,
            error=error_out,
            n_pixels=n_pixels,
            n_valid_pixels=n_valid,
            bad_fraction=bad_fraction,
            flagged=flagged,
            flag_reason="bad_pixels" if flagged else None,
        )


def _normalize_band(name: str, spec: Mapping[str, Any]) -> _BandFrame:
    """Normalize one public band spec into an internal frame."""
    if "data" not in spec:
        raise ValueError(f"band {name!r} is missing required key 'data'.")
    if "wcs" not in spec:
        raise ValueError(f"band {name!r} is missing required key 'wcs'.")

    data = np.asarray(spec["data"])
    if data.ndim != 2:
        raise ValueError(f"band {name!r} data must be 2-D; got shape {data.shape}.")

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

    return _BandFrame(
        name=str(name),
        data=data,
        wcs=spec["wcs"],
        error=error,
        wavelength=wavelength,
        wavelength_error=wavelength_error,
        label_map=label_map,
        label_allowed=label_allowed_out,
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
