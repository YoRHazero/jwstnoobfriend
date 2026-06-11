"""The grown-aperture draft sitting between :class:`ApertureSED` and its result.

:meth:`ApertureSED.draft` grows one binary aperture per band and hands back an
:class:`ApertureSEDDraft`. The draft is "the result minus the photometry": it
owns the per-band grown apertures and can rasterize their union, so the apertures
and candidate union can be previewed (:meth:`plot_apertures`, :meth:`plot_union`)
and one band re-grown (:meth:`regrow`) before any flux is measured. Choosing
which bands *define* the union, and measuring every band through it, both happen
in :meth:`measure` -- the only step that produces calibrated photometry.

Only binary masks are reprojected here (via
:mod:`noobfriend.extraction.photometry._coverage`); science data stays on each
band's native grid, exactly as in
:mod:`noobfriend.extraction.photometry._core`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from noobfriend.extraction.photometry._aperture import ApertureMask, grow_aperture_mask
from noobfriend.extraction.photometry._band import Band
from noobfriend.extraction.photometry._coverage import reproject_coverage
from noobfriend.extraction.photometry._floor import resolve_floor
from noobfriend.extraction.photometry._measure import measure_band
from noobfriend.extraction.photometry._noise import measure_aperture_noise
from noobfriend.extraction.photometry._result import ApertureSEDResult

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class ApertureSEDDraft:
    """Grown per-band apertures plus the geometry to measure them.

    Returned by :meth:`ApertureSED.draft`; not constructed directly.

    Attributes
    ----------
    source_id : str
        Source identifier, carried onto the measured result.
    ra, dec : float
        Source world coordinate in degrees, carried onto the result.
    bands : tuple[Band, ...]
        The normalized bands, each on its own native grid.
    reference_band : str
        Band whose grid rasterizes the union aperture.
    source_apertures : mapping
        Per-band grown aperture on each band's native grid.
    seeds : mapping
        Per-band ``(x, y)`` growth seed (kept so a band can be re-grown).
    scales : mapping
        Per-band local pixel scale (pixels-per-degree) at the seed.
    base_floor : int
        Stop-check floor at the finest band used for the grow (for re-growth).
    label_maps : mapping
        Per-band segmentation map actually used to grow (``None`` where
        unconstrained), so :meth:`regrow` can default to it.
    label_allowed : mapping
        Per-band allowed-label tuple actually used (``None`` where unset).
    union_threshold : float
        Reference-grid coverage fraction at which a pixel joins the union.
    flag_bad_fraction : float
        Aperture non-finite fraction at which a band is flagged when measured.
    coarse_step : tuple[int, int] or None
        Coverage-reprojection coarse step passed through to noobase.
    """

    source_id: str
    ra: float
    dec: float
    bands: tuple[Band, ...]
    reference_band: str
    source_apertures: Mapping[str, ApertureMask]
    seeds: Mapping[str, tuple[float, float]]
    scales: Mapping[str, float]
    base_floor: int
    label_maps: Mapping[str, np.ndarray | None]
    label_allowed: Mapping[str, tuple[int, ...] | None]
    union_threshold: float
    flag_bad_fraction: float
    coarse_step: tuple[int, int] | None

    @property
    def _by_name(self) -> dict[str, Band]:
        """Bands keyed by name."""
        return {band.name: band for band in self.bands}

    @property
    def band_names(self) -> tuple[str, ...]:
        """Band names in input order."""
        return tuple(band.name for band in self.bands)

    @property
    def band_images(self) -> dict[str, np.ndarray]:
        """Per-band native science image, keyed by band name."""
        return {band.name: band.data for band in self.bands}

    def _reference(self) -> Band:
        """Return the band whose grid rasterizes the union."""
        return self._by_name[self.reference_band]

    def _select_union_bands(self, union_bands: Sequence[str] | None) -> list[str]:
        """Validate and return the bands whose apertures define the union."""
        if union_bands is None:
            return list(self.band_names)
        names = [str(b) for b in union_bands]
        if not names:
            raise ValueError("union_bands must name at least one band.")
        unknown = [n for n in names if n not in self._by_name]
        if unknown:
            raise ValueError(
                f"unknown union band(s) {unknown}; available: {sorted(self._by_name)}."
            )
        return names

    def union_mask(self, union_bands: Sequence[str] | None = None) -> np.ndarray:
        """Rasterize the union of the selected bands' apertures on the reference grid.

        Parameters
        ----------
        union_bands : sequence of str, optional
            Bands whose grown apertures contribute to the union. Defaults to all
            bands. This selects which bands *define* the union shape -- every
            band is still measured through it in :meth:`measure`.

        Returns
        -------
        numpy.ndarray
            Boolean union mask on the reference band's grid.
        """
        selected = self._select_union_bands(union_bands)
        reference = self._reference()
        union = np.zeros(reference.shape, dtype=bool)
        for name in selected:
            mask = self.source_apertures[name].mask
            if name == reference.name:
                coverage = mask.astype(np.float64)
            else:
                coverage = reproject_coverage(
                    mask,
                    source_wcs=self._by_name[name].wcs,
                    target_wcs=reference.wcs,
                    target_shape=reference.shape,
                    coarse_step=self.coarse_step,
                )
            union |= coverage >= self.union_threshold
        return union

    def _coverage_map(self, union: np.ndarray) -> dict[str, np.ndarray]:
        """Express a reference-grid union as fractional coverage on every band."""
        reference = self._reference()
        coverage: dict[str, np.ndarray] = {}
        for band in self.bands:
            if band.name == reference.name:
                coverage[band.name] = union.astype(np.float64)
            else:
                coverage[band.name] = reproject_coverage(
                    union,
                    source_wcs=reference.wcs,
                    target_wcs=band.wcs,
                    target_shape=band.shape,
                    coarse_step=self.coarse_step,
                )
        return coverage

    def measure(
        self,
        *,
        union_bands: Sequence[str] | None = None,
        correlated_error: bool = True,
        noise_max_lag: int = 8,
        other_source_dilation: int = 2,
    ) -> ApertureSEDResult:
        """Measure every band through the union and return the SED result.

        Parameters
        ----------
        union_bands : sequence of str, optional
            Bands whose apertures define the union shape. Defaults to all bands.
            Every band is measured through the resulting union regardless -- this
            only controls which bands get a say in the union geometry (e.g.
            exclude a noisy non-detection so it cannot inflate the aperture).
        correlated_error : bool, default True
            Estimate each band's correlation-aware aperture error, median sky
            background, and detection SNR (see
            :func:`noobfriend.extraction.photometry._noise.measure_aperture_noise`).
            When ``False``, ``error`` stays the uncorrelated quadrature value and
            ``background_level`` is ``None``.
        noise_max_lag : int, default 8
            Lag-window half-width (pixels) for the sky autocovariance.
        other_source_dilation : int, default 2
            Dilation (pixels) of non-target sources when masking the sky.

        Returns
        -------
        ApertureSEDResult

        Raises
        ------
        ValueError
            If ``union_bands`` names an unknown band or the union aperture is
            empty.
        """
        union = self.union_mask(union_bands)
        if not bool(union.any()):
            raise ValueError("Union aperture is empty after mask alignment.")

        band_coverage = self._coverage_map(union)
        measurements = []
        for band in self.bands:
            coverage = band_coverage[band.name]
            band_photometry = measure_band(
                band, coverage, flag_bad_fraction=self.flag_bad_fraction
            )
            if correlated_error:
                band_photometry = measure_aperture_noise(
                    band,
                    coverage,
                    band_photometry,
                    max_lag=noise_max_lag,
                    other_source_dilation=other_source_dilation,
                )
            measurements.append(band_photometry)

        measured = tuple(measurements)
        if all(m.wavelength is not None for m in measured):
            measured = tuple(sorted(measured, key=lambda m: float(m.wavelength)))

        return ApertureSEDResult(
            source_id=self.source_id,
            ra=self.ra,
            dec=self.dec,
            measurements=measured,
            reference_band=self.reference_band,
            union_mask=union,
            band_coverage=band_coverage,
            source_apertures=self.source_apertures,
            band_images=self.band_images,
            label_maps=dict(self.label_maps),
        )

    def regrow(
        self,
        band: str,
        *,
        label_map: np.ndarray | None = None,
        label_allowed: Sequence[int] | None = None,
        allow_background: bool | None = None,
        grow_kwargs: Mapping[str, Any] | None = None,
    ) -> "ApertureSEDDraft":
        """Re-grow one band's aperture and return an updated draft.

        For manually tuning a single band after inspecting the preview. Arguments
        left ``None`` keep what that band already used; pass ``label_map`` /
        ``label_allowed`` to constrain its growth to a segment.

        This draft is left unchanged (it is frozen); carry on with the *returned*
        draft, which behaves like any other -- its union (:meth:`union_mask`),
        previews (:meth:`plot_apertures` / :meth:`plot_union`) and
        :meth:`measure` all reflect the re-grown band. Chain further
        :meth:`regrow` calls to tune more bands.

        Parameters
        ----------
        band : str
            Band to re-grow.
        label_map : numpy.ndarray, optional
            Segmentation map to confine growth to; defaults to the map this band
            grew under.
        label_allowed : sequence of int, optional
            Allowed labels; defaults to this band's current setting.
        allow_background : bool, optional
            Whether growth may leave the segment into the background; defaults to
            the band spec's value.
        grow_kwargs : mapping, optional
            Extra keyword arguments for noobase growth; a
            ``"min_pixels_before_stop_check"`` here overrides the base floor.

        Returns
        -------
        ApertureSEDDraft
            A new draft identical to this one except for the re-grown band.

        Raises
        ------
        ValueError
            If ``band`` is not in the draft.
        """
        if band not in self._by_name:
            raise ValueError(
                f"unknown band {band!r}; available: {sorted(self._by_name)}."
            )
        target = self._by_name[band]
        user_grow_kwargs = dict(grow_kwargs or {})
        base = int(
            user_grow_kwargs.pop("min_pixels_before_stop_check", self.base_floor)
        )
        finest_scale = max(self.scales.values(), default=0.0)

        new_label_map = (
            self.label_maps[band] if label_map is None else np.asarray(label_map)
        )
        new_allowed = (
            self.label_allowed[band]
            if label_allowed is None
            else tuple(int(v) for v in label_allowed)
        )
        new_allow_background = (
            target.allow_background
            if allow_background is None
            else bool(allow_background)
        )
        floor = resolve_floor(
            base,
            self.scales[band],
            finest_scale,
            new_label_map,
            target.data,
            self.seeds[band],
        )
        aperture = grow_aperture_mask(
            target.data,
            seed_xy=self.seeds[band],
            error=target.error,
            label_map=new_label_map,
            label_allowed=new_allowed,
            allow_background=new_allow_background,
            grow_kwargs={**user_grow_kwargs, "min_pixels_before_stop_check": floor},
        )
        return replace(
            self,
            source_apertures={**self.source_apertures, band: aperture},
            label_maps={**self.label_maps, band: new_label_map},
            label_allowed={**self.label_allowed, band: new_allowed},
        )

    def _scene(self, union_bands: Sequence[str] | None = None) -> Any:
        """Build the thumbnail scene for the candidate union (default: all bands)."""
        from noobfriend.extraction.photometry._thumbnails import ThumbnailScene

        union = self.union_mask(union_bands)
        return ThumbnailScene(
            band_images=self.band_images,
            source_apertures=self.source_apertures,
            band_coverage=self._coverage_map(union),
            reference_band=self.reference_band,
            order=self.band_names,
            wavelengths={band.name: band.wavelength for band in self.bands},
            flagged={band.name: False for band in self.bands},
        )

    def plot_apertures(
        self, *, union_bands: Sequence[str] | None = None, **kwargs: Any
    ) -> Figure:
        """Montage every band's thumbnail with its aperture and candidate union.

        Drawn before measuring: each band's own grown aperture as a contour and
        the candidate union (of ``union_bands``, default all) as it falls on that
        band as a translucent fill.

        Parameters
        ----------
        union_bands : sequence of str, optional
            Bands whose apertures define the previewed union. Defaults to all.
        **kwargs
            Display options (all keyword-only):

            - ``zoom_in`` (bool, default ``False``) frames each panel on the
              source instead of the full cutout, with a ``zoom_pad`` (default
              ``0.3``) margin as a fraction of the source extent;
            - ``bands`` selects which bands to draw (default all) and ``ncols``
              the montage column count;
            - ``cmap``, ``stretch`` (``"linear"`` or ``"log"``) and
              ``vmin`` / ``vmax`` / ``pmin`` / ``pmax`` set the thumbnail
              background scaling;
            - ``show_aperture`` / ``show_coverage`` / ``show_seed`` toggle the
              overlays and ``aperture_color`` / ``coverage_color`` /
              ``coverage_alpha`` style them;
            - ``panel_size`` is the panel side length in inches;
            - ``title`` and ``save`` set the figure title and an output path.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from noobfriend.extraction.photometry._thumbnails import aperture_montage

        return aperture_montage(self._scene(union_bands), **kwargs)

    def plot_thumbnail(
        self, band: str, *, union_bands: Sequence[str] | None = None, **kwargs: Any
    ) -> Figure:
        """Draw one band's thumbnail with its aperture and the candidate union.

        Parameters
        ----------
        band : str
            Band to draw.
        union_bands : sequence of str, optional
            Bands whose apertures define the previewed union. Defaults to all.
        **kwargs
            Display options (all keyword-only):

            - ``zoom_in`` (bool, default ``False``) frames the source instead of
              the full cutout, with a ``zoom_pad`` (default ``0.3``) margin;
            - ``cmap``, ``stretch`` (``"linear"`` or ``"log"``) and
              ``vmin`` / ``vmax`` / ``pmin`` / ``pmax`` set the background scaling;
            - ``show_aperture`` / ``show_coverage`` / ``show_seed`` toggle the
              overlays and ``aperture_color`` / ``coverage_color`` /
              ``coverage_alpha`` style them;
            - ``size`` is the figure side length in inches;
            - ``title`` and ``save`` set the figure title and an output path.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from noobfriend.extraction.photometry._thumbnails import aperture_thumbnail

        return aperture_thumbnail(self._scene(union_bands), band, **kwargs)

    def plot_union(
        self, *, union_bands: Sequence[str] | None = None, **kwargs: Any
    ) -> Figure:
        """Draw the candidate union on the reference band, enlarged.

        Shows the union (of ``union_bands``, default all) at its finest native
        resolution -- the reference band's panel, where the union fill is the
        rasterized union itself.

        Parameters
        ----------
        union_bands : sequence of str, optional
            Bands whose apertures define the previewed union. Defaults to all.
        **kwargs
            Display options (all keyword-only):

            - ``zoom_in`` (bool, default ``False``) frames the source instead of
              the full cutout, with a ``zoom_pad`` (default ``0.3``) margin;
            - ``cmap``, ``stretch`` (``"linear"`` or ``"log"``) and
              ``vmin`` / ``vmax`` / ``pmin`` / ``pmax`` set the background scaling;
            - ``show_aperture`` / ``show_coverage`` / ``show_seed`` toggle the
              overlays and ``aperture_color`` / ``coverage_color`` /
              ``coverage_alpha`` style them;
            - ``size`` is the figure side length in inches;
            - ``title`` and ``save`` set the figure title and an output path.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from noobfriend.extraction.photometry._thumbnails import aperture_thumbnail

        return aperture_thumbnail(
            self._scene(union_bands), self.reference_band, **kwargs
        )

    def plot_segmentation(
        self,
        band: str,
        *,
        segment_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Figure:
        """Preview ``segment()`` on one band beside its science image.

        Runs :func:`noobfriend.core.imgutils.segment` *live* on the band and
        draws the labels next to the image (seed marked, the seed's segment
        outlined), so you can judge the labels -- and which one the seed lands on
        -- before committing to ``auto_segment`` or a ``label_map``. Pass
        ``segment_kwargs`` (``nsigma`` / ``min_size`` / ``deblend`` /
        ``connectivity``) to match what ``auto_segment=True`` would produce.

        Parameters
        ----------
        band : str
            Band to segment and draw.
        segment_kwargs : mapping, optional
            Keyword arguments forwarded to
            :func:`noobfriend.core.imgutils.segment`.
        **kwargs
            Display options (all keyword-only):

            - ``zoom_in`` (bool, default ``False``) frames both panels on the
              seed's segment instead of the full cutout, with a ``zoom_pad``
              (default ``0.3``) margin;
            - ``cmap``, ``stretch`` (``"linear"`` or ``"log"``) and
              ``vmin`` / ``vmax`` / ``pmin`` / ``pmax`` set the science-image
              scaling, and ``labels_cmap`` the label panel's colormap;
            - ``show_seed`` toggles the seed marker;
            - ``panel_size`` is each panel's side length in inches;
            - ``title`` (default the band name) and ``save`` set the figure title
              and an output path.

        Returns
        -------
        matplotlib.figure.Figure
            The two-panel figure.

        Raises
        ------
        ValueError
            If ``band`` is not in the draft.
        """
        from noobfriend.core.imgutils import segment
        from noobfriend.extraction.photometry._thumbnails import segmentation_montage

        if band not in self._by_name:
            raise ValueError(
                f"unknown band {band!r}; available: {sorted(self._by_name)}."
            )
        target = self._by_name[band]
        labels = segment(
            np.asarray(target.data, dtype=float), **dict(segment_kwargs or {})
        )
        kwargs.setdefault("title", band)
        return segmentation_montage(
            target.data, labels, self.source_apertures[band].seed_xy, **kwargs
        )
