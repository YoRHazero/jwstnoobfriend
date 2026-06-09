"""Bokeh figure for an aperture SED.

Isolated from :mod:`noobfriend.extraction.photometry._result` so the result
object stays about data and the (heavier) Bokeh import stays local to the one
function that needs it.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from noobfriend.extraction.photometry._measure import BandPhotometry


def sed_figure(
    measurements: Sequence[BandPhotometry],
    *,
    include_flagged: bool = True,
    title: str | None = None,
) -> Any:
    """Build a Bokeh figure of the SED in mJy with flux and wavelength errors.

    Parameters
    ----------
    measurements : sequence of BandPhotometry
        Per-band measurements to plot.
    include_flagged : bool, default True
        Include bands flagged for bad pixels, shown in a contrasting colour.
    title : str, optional
        Plot title. Defaults to ``"Aperture SED"``.

    Returns
    -------
    bokeh.plotting.figure
        The assembled figure.

    Raises
    ------
    ValueError
        If no included band has both a wavelength and an mJy flux.
    """
    from bokeh.models import ColumnDataSource, HoverTool, Whisker
    from bokeh.plotting import figure

    points = [
        m
        for m in measurements
        if m.wavelength is not None
        and m.flux_mjy is not None
        and (include_flagged or not m.flagged)
    ]
    if not points:
        raise ValueError(
            "Cannot plot SED: no included band has both wavelength and flux_mjy."
        )
    points = sorted(points, key=lambda m: float(m.wavelength))

    x = np.asarray([m.wavelength for m in points], dtype=float)
    y = np.asarray([m.flux_mjy for m in points], dtype=float)
    yerr = np.asarray(
        [np.nan if m.error_mjy is None else m.error_mjy for m in points], dtype=float
    )
    xerr_left = np.asarray(
        [
            np.nan if m.wavelength_error is None else m.wavelength_error[0]
            for m in points
        ],
        dtype=float,
    )
    xerr_right = np.asarray(
        [
            np.nan if m.wavelength_error is None else m.wavelength_error[1]
            for m in points
        ],
        dtype=float,
    )
    source = ColumnDataSource(
        {
            "band": [m.band for m in points],
            "wavelength": x,
            "wave_lower": x - xerr_left,
            "wave_upper": x + xerr_right,
            "flux_mjy": y,
            "flux_mjy_lower": y - yerr,
            "flux_mjy_upper": y + yerr,
            "bad_fraction": [m.bad_fraction for m in points],
            "flagged": [m.flagged for m in points],
            "color": ["#c43b3b" if m.flagged else "#1f6f8b" for m in points],
        }
    )

    p = figure(
        title=title or "Aperture SED",
        x_axis_label="Wavelength (µm)",
        y_axis_label="Flux (mJy)",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        height=420,
        width=680,
    )
    p.scatter(
        "wavelength",
        "flux_mjy",
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
            lower="flux_mjy_lower",
            upper="flux_mjy_upper",
            line_color="#5f6b73",
        )
    )
    p.add_layout(
        Whisker(
            source=source,
            base="flux_mjy",
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
                ("flux_mjy", "@flux_mjy"),
                ("bad fraction", "@bad_fraction"),
                ("flagged", "@flagged"),
            ]
        )
    )
    return p
