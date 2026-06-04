"""Public surface: the blind grism emission-line finder.

:class:`GrismLineFinder` bundles the detection configuration once and exposes
the products as composable steps, mirroring
:class:`~noobfriend.extraction.grism.GrismExtractor`'s style:

* :meth:`~GrismLineFinder.exposure_heatmap` -- one frame's band-pass SNR map
  (1:1 with the frame; hot = likely emission line);
* :meth:`~GrismLineFinder.combine` -- reproject + median-combine same-field
  dither exposures into one union-footprint heatmap (deep, artifacts rejected);
* :meth:`~GrismLineFinder.catalog` -- the auxiliary peak list off any heatmap.

The heatmap is the primary product; the catalog is a convenience for recall.
Source association and redshift are out of scope (left to the user / SED).

This module is internal; :class:`GrismLineFinder` is re-exported from
``noobfriend.extraction.grism.linefind``.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Self

import numpy as np

from noobfriend.extraction.grism._coverage import FrameMeta
from noobfriend.extraction.grism.linefind._combine import (
    CombinedHeatmap,
)
from noobfriend.extraction.grism.linefind._combine import (
    combine_heatmap as _combine_heatmap,
)
from noobfriend.extraction.grism.linefind._combine import (
    exposure_heatmap as _exposure_heatmap,
)
from noobfriend.extraction.grism.linefind._detect import Candidate
from noobfriend.extraction.grism.linefind._detect import detect as _detect
from noobfriend.extraction.grism.linefind._filter import BandPass

_C_KMS = 299792.458
_FWHM_TO_SIGMA = 1.0 / 2.3548200450309493  # 1 / (2 sqrt(2 ln 2))


def _velocity_to_sigma(
    velocity_fwhm_kms: float,
    reference_wavelength: float,
    dispersion_per_pixel: float,
) -> float:
    """Convert a velocity FWHM (km/s) to a dispersion-axis sigma (pixels).

    Parameters
    ----------
    velocity_fwhm_kms : float
        Velocity FWHM in km/s (a line's, or the continuum's smoothing scale).
    reference_wavelength : float
        Reference wavelength (any unit, shared with ``dispersion_per_pixel``).
    dispersion_per_pixel : float
        Dispersion in wavelength-per-pixel (same wavelength unit).

    Returns
    -------
    float
        The Gaussian sigma along the dispersion axis, in pixels.
    """
    wave_fwhm = reference_wavelength * velocity_fwhm_kms / _C_KMS
    return (wave_fwhm / dispersion_per_pixel) * _FWHM_TO_SIGMA


@dataclass(frozen=True)
class GrismLineFinder:
    """A configured blind emission-line finder for WFSS grism frames.

    Build with :meth:`configure`, then call :meth:`exposure_heatmap` /
    :meth:`combine` to make heatmaps and :meth:`catalog` to peak-find them.

    Attributes
    ----------
    dispersion_axis : int
        Array axis the grism disperses along (``1`` row / ``0`` column).
    line_scales : tuple[tuple[float, float], ...]
        ``(sigma_cross, sigma_disp)`` pixel sigma pairs scanned by the band-pass.
    continuum_sigma_disp : float
        Along-dispersion continuum Gaussian sigma (pixels).
    threshold : float
        Catalog seed SNR threshold.
    grow_threshold : float or None
        Lower SNR threshold seeds are grown / deblended to (``None`` ->
        ``0.5 * threshold``).
    min_distance : int
        Minimum separation (pixels) between catalog seeds.
    coarse_step : tuple[int, int] or None
        ``make_pixel_corners`` speedup used by :meth:`combine`.
    """

    dispersion_axis: int
    line_scales: tuple[tuple[float, float], ...]
    continuum_sigma_disp: float
    threshold: float
    grow_threshold: float | None
    min_distance: int
    coarse_step: tuple[int, int] | None

    @classmethod
    def configure(
        cls,
        *,
        dispersion: Literal["row", "column"],
        line_sigmas: Sequence[float] = (1.2, 2.0),
        line_velocities_kms: Sequence[float] | None = None,
        continuum_sigma_disp: float = 8.0,
        continuum_velocity_kms: float | None = None,
        reference_wavelength: float | None = None,
        dispersion_per_pixel: float | None = None,
        threshold: float = 6.0,
        grow_threshold: float | None = None,
        min_distance: int = 3,
        coarse_step: tuple[int, int] | None = (64, 64),
    ) -> Self:
        """Build a finder from a detection configuration.

        ``line_sigmas`` is always the line's *spatial* (cross-dispersion) sigma.
        Without ``line_velocities_kms`` the line is isotropic (along-dispersion
        sigma equals it); with it, the along-dispersion sigma becomes
        ``hypot(line_sigma, sigma_spectral(v))`` -- so the scan is the Cartesian
        product of spatial sizes x velocity widths. The continuum has only an
        along-dispersion scale: ``continuum_sigma_disp`` (px), or
        ``continuum_velocity_kms`` (km/s) which overrides it. Any km/s input
        needs ``reference_wavelength`` and ``dispersion_per_pixel``.

        Parameters
        ----------
        dispersion : {"row", "column"}
            Dispersion direction: ``"row"`` (along x; GRISMR / GR150R) or
            ``"column"`` (along y; GRISMC / GR150C).
        line_sigmas : Sequence[float], optional
            Line cross-dispersion (spatial) Gaussian sigmas (pixels) to scan, by
            default ``(1.2, 2.0)``.
        line_velocities_kms : Sequence[float] or None, optional
            Line velocity FWHMs (km/s) to scan; adds spectral broadening along
            dispersion. ``None`` (default) keeps the line isotropic.
        continuum_sigma_disp : float, optional
            Along-dispersion continuum Gaussian sigma (pixels), by default
            ``8.0``; must exceed every line along-dispersion sigma.
        continuum_velocity_kms : float or None, optional
            Continuum along-dispersion smoothing scale as a velocity (km/s); if
            given it overrides ``continuum_sigma_disp``.
        reference_wavelength, dispersion_per_pixel : float or None, optional
            Reference wavelength and dispersion (wavelength-per-pixel, same unit)
            for any km/s -> pixel conversion; required if a velocity is given.
        threshold : float, optional
            Catalog seed SNR threshold, by default ``6.0``.
        grow_threshold : float or None, optional
            Lower SNR threshold for growing / deblending seeds; ``None``
            (default) uses ``0.5 * threshold``.
        min_distance : int, optional
            Minimum separation (pixels) between seeds, by default ``3``.
        coarse_step : tuple[int, int] or None, optional
            ``make_pixel_corners`` speedup for :meth:`combine`, by default
            ``(64, 64)``.

        Returns
        -------
        GrismLineFinder

        Raises
        ------
        ValueError
            If ``dispersion`` is invalid, a km/s input lacks its conversion
            inputs, or ``continuum_sigma_disp`` does not exceed every line
            along-dispersion sigma.
        """
        if dispersion not in ("row", "column"):
            raise ValueError(
                f"dispersion must be 'row' or 'column', got {dispersion!r}."
            )
        wants_conversion = (
            line_velocities_kms is not None or continuum_velocity_kms is not None
        )
        if wants_conversion and (
            reference_wavelength is None or dispersion_per_pixel is None
        ):
            raise ValueError(
                "a km/s input requires reference_wavelength and dispersion_per_pixel."
            )

        cross = tuple(float(s) for s in line_sigmas)
        if line_velocities_kms is None:
            scales = tuple((s, s) for s in cross)
        else:
            scales = tuple(
                (
                    s,
                    float(
                        np.hypot(
                            s,
                            _velocity_to_sigma(
                                float(v), reference_wavelength, dispersion_per_pixel
                            ),
                        )
                    ),
                )
                for s in cross
                for v in line_velocities_kms
            )

        if continuum_velocity_kms is not None:
            cont_disp = _velocity_to_sigma(
                float(continuum_velocity_kms),
                reference_wavelength,
                dispersion_per_pixel,
            )
        else:
            cont_disp = float(continuum_sigma_disp)

        max_disp = max(d for _, d in scales)
        if cont_disp <= max_disp:
            raise ValueError(
                f"continuum_sigma_disp ({cont_disp:.2f} px) must exceed every "
                f"line sigma_disp (max {max_disp:.2f} px)."
            )

        return cls(
            dispersion_axis=1 if dispersion == "row" else 0,
            line_scales=scales,
            continuum_sigma_disp=cont_disp,
            threshold=float(threshold),
            grow_threshold=grow_threshold,
            min_distance=int(min_distance),
            coarse_step=coarse_step,
        )

    def exposure_heatmap(self, data: np.ndarray, error: np.ndarray) -> np.ndarray:
        """Band-pass SNR heatmap of one frame (1:1 with the input).

        Parameters
        ----------
        data, error : numpy.ndarray
            The 2-D frame and its 1-sigma error.

        Returns
        -------
        numpy.ndarray
            Per-pixel max band-pass SNR over scales; ``NaN`` where masked /
            low-coverage. Hot = likely emission line.
        """
        return _exposure_heatmap(
            data,
            error,
            dispersion_axis=self.dispersion_axis,
            line_scales=self.line_scales,
            continuum_sigma_disp=self.continuum_sigma_disp,
        )

    def combine(
        self,
        frames: Sequence[FrameMeta],
        load: Callable[[str], tuple[np.ndarray, np.ndarray]],
        *,
        reference_index: int = 0,
    ) -> CombinedHeatmap:
        """Combine same-field dither exposures into one union-footprint heatmap.

        Parameters
        ----------
        frames : Sequence[FrameMeta]
            Same-pointing dither exposures (``wcs`` + ``shape`` + ``id``).
        load : collections.abc.Callable
            ``load(id) -> (data, error)``, called once per exposure.
        reference_index : int, optional
            Exposure whose detector frame defines the grid, by default ``0``.

        Returns
        -------
        CombinedHeatmap
            The union heatmap plus combined data / error / coverage and grid.
        """
        return _combine_heatmap(
            frames,
            load,
            dispersion_axis=self.dispersion_axis,
            line_scales=self.line_scales,
            continuum_sigma_disp=self.continuum_sigma_disp,
            reference_index=reference_index,
            coarse_step=self.coarse_step,
        )

    def catalog(self, heatmap: np.ndarray) -> list[Candidate]:
        """Peak-find a heatmap into a candidate list (auxiliary).

        Parameters
        ----------
        heatmap : numpy.ndarray
            A 2-D band-pass SNR map, e.g. from :meth:`exposure_heatmap` or
            :attr:`CombinedHeatmap.heatmap`.

        Returns
        -------
        list[Candidate]
            Candidates above ``threshold``, sorted by descending SNR. Because
            the heatmap is already collapsed over scales, ``Candidate.scale`` is
            the nominal first ``line_scales`` pair, not a per-detection best.
        """
        bp = BandPass(
            snr=heatmap[np.newaxis],
            signal=heatmap[np.newaxis],
            scales=(self.line_scales[0],),
        )
        return _detect(
            bp,
            dispersion_axis=self.dispersion_axis,
            threshold=self.threshold,
            grow_threshold=self.grow_threshold,
            min_distance=self.min_distance,
        )
