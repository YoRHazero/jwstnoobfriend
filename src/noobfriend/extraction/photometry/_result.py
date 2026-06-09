"""The measured SED plus the aperture maps it was measured through."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from noobfriend.extraction.photometry._aperture import ApertureMask
from noobfriend.extraction.photometry._measure import BandPhotometry


@dataclass(frozen=True)
class ApertureSEDResult:
    """Measured SED plus the aperture maps used to measure it.

    Attributes
    ----------
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
    source_metadata : mapping
        Optional provenance metadata, e.g. grizli-cutout URLs and filters.
    """

    measurements: tuple[BandPhotometry, ...]
    reference_band: str
    union_mask: np.ndarray
    band_coverage: Mapping[str, np.ndarray]
    source_apertures: Mapping[str, ApertureMask]
    source_metadata: Mapping[str, Any]

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
                "flux_mjy": m.flux_mjy,
                "error_mjy": m.error_mjy,
                "flux_scale_mjy": m.flux_scale_mjy,
                "flux_unit": m.flux_unit,
                "covered_area": m.covered_area,
                "valid_area": m.valid_area,
                "bad_fraction": m.bad_fraction,
                "flagged": m.flagged,
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
        """Plot the measured SED in mJy with flux and wavelength error bars.

        Parameters
        ----------
        include_flagged : bool, default True
            Include bands flagged for bad pixels. Flagged points are shown in a
            contrasting colour.
        title : str, optional
            Plot title. Defaults to ``"Aperture SED"``.
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
            If no included band has both a wavelength and mJy flux.
        """
        from noobfriend.core.display.plot._bokeh import display
        from noobfriend.extraction.photometry._plot import sed_figure

        figure = sed_figure(
            self.measurements, include_flagged=include_flagged, title=title
        )
        if not display_plot:
            return figure
        return display(figure, height=height)
