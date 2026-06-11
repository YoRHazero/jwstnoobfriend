"""The measured SED plus the aperture maps it was measured through."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from noobfriend.extraction.photometry._aperture import ApertureMask
from noobfriend.extraction.photometry._measure import BandPhotometry


@dataclass(frozen=True)
class ApertureSEDResult:
    """Measured SED plus the aperture maps used to measure it.

    Attributes
    ----------
    source_id : str
        Identifier of the measured source (used as the default SED plot title).
    ra, dec : float
        Source world coordinate in degrees (ICRS).
    measurements : tuple[BandPhotometry, ...]
        Per-band photometry, sorted by wavelength when every band has one,
        otherwise in input band order.
    reference_band : str
        Band whose native grid was used to rasterize the union aperture.
    union_mask : numpy.ndarray
        Boolean union aperture on the reference grid.
    band_coverage : mapping
        Per-band fractional coverage of the union aperture on each band's own
        native grid (the weight maps the measurement summed over).
    source_apertures : mapping
        Per-band grown apertures on each band's native grid.
    band_images : mapping
        Per-band native science image, kept so the result is self-contained for
        the aperture thumbnails (see :meth:`plot_apertures`); same grid and
        shape as the matching ``band_coverage`` and ``source_apertures`` entry.
    label_maps : mapping
        Per-band segmentation map that actually confined the band's growth
        (``None`` where it grew unconstrained), kept so :meth:`plot_segmentation`
        can show the segmentation that was used.
    """

    source_id: str
    ra: float
    dec: float
    measurements: tuple[BandPhotometry, ...]
    reference_band: str
    union_mask: np.ndarray
    band_coverage: Mapping[str, np.ndarray]
    source_apertures: Mapping[str, ApertureMask]
    band_images: Mapping[str, np.ndarray]
    label_maps: Mapping[str, np.ndarray | None]

    def _default_title(self) -> str:
        """SED plot title from the source id, or a generic fallback."""
        return f"{self.source_id} aperture SED" if self.source_id else "Aperture SED"

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
                "error_uncorrelated": m.error_uncorrelated,
                "flux_mjy": m.flux_mjy,
                "error_mjy": m.error_mjy,
                "flux_scale_mjy": m.flux_scale_mjy,
                "flux_unit": m.flux_unit,
                "covered_area": m.covered_area,
                "valid_area": m.valid_area,
                "bad_fraction": m.bad_fraction,
                "background_level": m.background_level,
                "snr": m.snr,
                "flagged": m.flagged,
            }
            for m in self.measurements
        ]
        return Table(rows=rows)

    def plot(
        self,
        *,
        include_flagged: bool = True,
        detection_snr: float = 2.0,
        upper_limit_sigma: float = 2.0,
        title: str | None = None,
        size: int = 680,
        save: str | Path | None = None,
    ) -> Any:
        """Plot the measured SED as a static matplotlib figure.

        Bands with :attr:`BandPhotometry.snr` at least ``detection_snr`` (or an
        unknown SNR) are drawn as detections with flux and wavelength error bars;
        the rest are downward upper-limit arrows at ``flux + upper_limit_sigma *
        error``. For the interactive Bokeh view with hover, use :meth:`show`.

        Parameters
        ----------
        include_flagged : bool, default True
            Include bands flagged for bad pixels, shown in a contrasting colour.
        detection_snr : float, default 2.0
            Minimum SNR for a band to count as a detection; must be >= 1.
        upper_limit_sigma : float, default 2.0
            Non-detections are drawn at ``flux + upper_limit_sigma * error``;
            must be > 0.
        title : str, optional
            Plot title. Defaults to ``"Aperture SED"``.
        size : int, default 680
            Figure width in pixels; the height follows a fixed aspect ratio.
        save : str or pathlib.Path, optional
            If given, write the figure to this path with ``savefig``.

        Returns
        -------
        matplotlib.figure.Figure

        Raises
        ------
        ValueError
            If no included band has both a wavelength and mJy flux, or if
            ``detection_snr`` < 1 or ``upper_limit_sigma`` <= 0.
        """
        from noobfriend.extraction.photometry._plot import sed_figure_mpl

        return sed_figure_mpl(
            self.measurements,
            include_flagged=include_flagged,
            detection_snr=detection_snr,
            upper_limit_sigma=upper_limit_sigma,
            title=title or self._default_title(),
            size=size,
            save=save,
        )

    def show(
        self,
        *,
        include_flagged: bool = True,
        detection_snr: float = 2.0,
        upper_limit_sigma: float = 2.0,
        title: str | None = None,
        display_plot: bool = True,
        height: int = 500,
    ) -> Any:
        """Show the measured SED as an interactive Bokeh figure (notebook).

        Same detection / upper-limit split as :meth:`plot`, but interactive
        (hover reports band, wavelength, flux, SNR, and flag state). Use
        :meth:`plot` for a static, savable matplotlib figure.

        Parameters
        ----------
        include_flagged, detection_snr, upper_limit_sigma, title
            See :meth:`plot`.
        display_plot : bool, default True
            When ``True``, return the notebook-display wrapper. When ``False``,
            return the raw Bokeh figure for tests or further customization.
        height : int, default 500
            Display iframe height when ``display_plot`` is ``True``.

        Returns
        -------
        Any
            A notebook-displayable object or a Bokeh figure.

        Raises
        ------
        ValueError
            If no included band has both a wavelength and mJy flux, or if
            ``detection_snr`` < 1 or ``upper_limit_sigma`` <= 0.
        """
        from noobfriend.core.display.plot._bokeh import display
        from noobfriend.extraction.photometry._plot import sed_figure

        figure = sed_figure(
            self.measurements,
            include_flagged=include_flagged,
            detection_snr=detection_snr,
            upper_limit_sigma=upper_limit_sigma,
            title=title or self._default_title(),
        )
        if not display_plot:
            return figure
        return display(figure, height=height)

    def plot_apertures(self, **kwargs: Any) -> Any:
        """Montage every band's thumbnail with its aperture and union overlays.

        One matplotlib panel per band, drawn on the band's own native grid: the
        science thumbnail, the band's grown aperture as a contour, and the union
        aperture as measured on this band (:attr:`band_coverage`) as a
        translucent fill whose opacity tracks the coverage fraction. The
        reference band and any flagged band are badged.

        Parameters
        ----------
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
            The assembled montage.
        """
        from noobfriend.extraction.photometry._thumbnails import aperture_montage

        return aperture_montage(self._scene(), **kwargs)

    def plot_thumbnail(self, band: str, **kwargs: Any) -> Any:
        """Draw one band's thumbnail with its aperture and union overlays, enlarged.

        The single-band counterpart of :meth:`plot_apertures` (identical
        overlays) for inspecting one band closely.

        Parameters
        ----------
        band : str
            Band to draw.
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
            The single-panel figure.
        """
        from noobfriend.extraction.photometry._thumbnails import aperture_thumbnail

        return aperture_thumbnail(self._scene(), band, **kwargs)

    def plot_segmentation(self, band: str, **kwargs: Any) -> Any:
        """Draw the segmentation map that confined one band's growth, with the image.

        Shows the ``label_map`` that was *actually used* to grow ``band`` beside
        its science image (seed marked, the seed's segment outlined). The
        segmentation is fixed once measured, so -- unlike the draft's live
        preview -- this takes no ``segment`` parameters. Raises if the band grew
        unconstrained (no segmentation map).

        Parameters
        ----------
        band : str
            Band to draw.
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
            If ``band`` is unknown, or it grew without a segmentation map.
        """
        from noobfriend.extraction.photometry._thumbnails import segmentation_montage

        if band not in self.band_images:
            raise ValueError(
                f"unknown band {band!r}; available: {sorted(self.band_images)}."
            )
        labels = self.label_maps.get(band)
        if labels is None:
            raise ValueError(
                f"band {band!r} grew without a segmentation map; nothing to show."
            )
        kwargs.setdefault("title", band)
        return segmentation_montage(
            self.band_images[band],
            labels,
            self.source_apertures[band].seed_xy,
            **kwargs,
        )

    def _scene(self) -> Any:
        """Build the thumbnail scene from the measured bands."""
        from noobfriend.extraction.photometry._thumbnails import ThumbnailScene

        return ThumbnailScene(
            band_images=self.band_images,
            source_apertures=self.source_apertures,
            band_coverage=self.band_coverage,
            reference_band=self.reference_band,
            order=tuple(m.band for m in self.measurements),
            wavelengths={m.band: m.wavelength for m in self.measurements},
            flagged={m.band: m.flagged for m in self.measurements},
        )
