"""Bokeh and matplotlib figures for an aperture SED.

Isolated from :mod:`noobfriend.extraction.photometry._result` so the result
object stays about data and the (heavier) plotting imports stay local. The
detection / upper-limit split is shared by both backends via
:func:`_classify_points`; :func:`sed_figure` renders the interactive Bokeh view
(``ApertureSEDResult.show``) and :func:`sed_figure_mpl` the static matplotlib
figure (``ApertureSEDResult.plot``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from noobfriend.extraction.photometry._measure import BandPhotometry

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.figure import Figure

#: matplotlib SED figure height as a fraction of its width.
_MPL_ASPECT: float = 0.55


def _classify_points(
    measurements: Sequence[BandPhotometry],
    *,
    include_flagged: bool,
    detection_snr: float,
    upper_limit_sigma: float,
) -> tuple[list[BandPhotometry], list[BandPhotometry]]:
    """Validate plot parameters and split bands into detections and upper limits.

    A band is an upper limit when its :attr:`BandPhotometry.snr` is known and
    below ``detection_snr`` (and it has an ``error_mjy`` to place the limit);
    everything else with a wavelength and mJy flux is a detection. Shared by both
    plot backends.

    Raises
    ------
    ValueError
        If ``detection_snr`` < 1, ``upper_limit_sigma`` <= 0, or no included band
        has both a wavelength and an mJy flux.
    """
    if detection_snr < 1.0:
        raise ValueError(f"detection_snr must be >= 1, got {detection_snr}")
    if upper_limit_sigma <= 0.0:
        raise ValueError(f"upper_limit_sigma must be > 0, got {upper_limit_sigma}")

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

    def is_upper_limit(m: BandPhotometry) -> bool:
        return m.snr is not None and m.snr < detection_snr and m.error_mjy is not None

    detections = [m for m in points if not is_upper_limit(m)]
    uppers = [m for m in points if is_upper_limit(m)]
    return detections, uppers


def sed_figure(
    measurements: Sequence[BandPhotometry],
    *,
    include_flagged: bool = True,
    detection_snr: float = 2.0,
    upper_limit_sigma: float = 2.0,
    title: str | None = None,
) -> Any:
    """Build a Bokeh figure of the SED in mJy, with upper limits for non-detections.

    Bands with :attr:`BandPhotometry.snr` at least ``detection_snr`` (or unknown
    SNR) are drawn as detections with flux and wavelength error bars; the rest
    are drawn as downward upper-limit arrows at ``flux + upper_limit_sigma * error``.

    Parameters
    ----------
    measurements : sequence of BandPhotometry
        Per-band measurements to plot.
    include_flagged : bool, default True
        Include bands flagged for bad pixels, shown in a contrasting colour.
    detection_snr : float, default 2.0
        Minimum SNR for a detection; must be >= 1. A band with ``snr is None``
        (no error) is treated as a detection.
    upper_limit_sigma : float, default 2.0
        Non-detections are drawn at ``flux + upper_limit_sigma * error``; > 0.
    title : str, optional
        Plot title. Defaults to ``"Aperture SED"``.

    Returns
    -------
    bokeh.plotting.figure
        The assembled figure.

    Raises
    ------
    ValueError
        If no included band has both a wavelength and an mJy flux, or if
        ``detection_snr`` < 1 or ``upper_limit_sigma`` <= 0.
    """
    from bokeh.models import ColumnDataSource, HoverTool, Whisker
    from bokeh.plotting import figure

    detections, uppers = _classify_points(
        measurements,
        include_flagged=include_flagged,
        detection_snr=detection_snr,
        upper_limit_sigma=upper_limit_sigma,
    )

    p = figure(
        title=title or "Aperture SED",
        x_axis_label="Wavelength (µm)",
        y_axis_label="Flux (mJy)",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        height=420,
        width=680,
    )

    if detections:
        x = np.asarray([m.wavelength for m in detections], dtype=float)
        y = np.asarray([m.flux_mjy for m in detections], dtype=float)
        yerr = np.asarray(
            [np.nan if m.error_mjy is None else m.error_mjy for m in detections],
            dtype=float,
        )
        xl = np.asarray(
            [
                0.0 if m.wavelength_error is None else m.wavelength_error[0]
                for m in detections
            ],
            dtype=float,
        )
        xr = np.asarray(
            [
                0.0 if m.wavelength_error is None else m.wavelength_error[1]
                for m in detections
            ],
            dtype=float,
        )
        source = ColumnDataSource(
            {
                "band": [m.band for m in detections],
                "wavelength": x,
                "wave_lower": x - xl,
                "wave_upper": x + xr,
                "flux_mjy": y,
                "flux_mjy_lower": y - yerr,
                "flux_mjy_upper": y + yerr,
                "snr": [m.snr for m in detections],
                "bad_fraction": [m.bad_fraction for m in detections],
                "flagged": [m.flagged for m in detections],
                "color": ["#c43b3b" if m.flagged else "#1f6f8b" for m in detections],
            }
        )
        marker = p.scatter(
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
                renderers=[marker],
                tooltips=[
                    ("band", "@band"),
                    ("wavelength", "@wavelength"),
                    ("flux_mjy", "@flux_mjy"),
                    ("snr", "@snr"),
                    ("bad fraction", "@bad_fraction"),
                    ("flagged", "@flagged"),
                ],
            )
        )

    if uppers:
        xu = np.asarray([m.wavelength for m in uppers], dtype=float)
        limit = np.asarray(
            [m.flux_mjy + upper_limit_sigma * m.error_mjy for m in uppers], dtype=float
        )
        tail = np.abs(limit) * 0.3
        ul_source = ColumnDataSource(
            {
                "band": [m.band for m in uppers],
                "wavelength": xu,
                "limit": limit,
                "limit_tail": limit - tail,
                "flux_mjy": [m.flux_mjy for m in uppers],
                "snr": [m.snr for m in uppers],
            }
        )
        p.segment(
            x0="wavelength",
            y0="limit",
            x1="wavelength",
            y1="limit_tail",
            source=ul_source,
            line_color="#7a7a7a",
            line_width=1.5,
        )
        arrow = p.scatter(
            "wavelength",
            "limit",
            source=ul_source,
            marker="inverted_triangle",
            size=11,
            fill_color="#7a7a7a",
            line_color="#4a4a4a",
        )
        p.add_tools(
            HoverTool(
                renderers=[arrow],
                tooltips=[
                    ("band", "@band"),
                    ("wavelength", "@wavelength"),
                    (f"upper limit (+{upper_limit_sigma:g}σ)", "@limit"),
                    ("flux_mjy", "@flux_mjy"),
                    ("snr", "@snr"),
                ],
            )
        )

    return p


def sed_figure_mpl(
    measurements: Sequence[BandPhotometry],
    *,
    include_flagged: bool = True,
    detection_snr: float = 2.0,
    upper_limit_sigma: float = 2.0,
    title: str | None = None,
    size: int = 680,
    save: str | Path | None = None,
) -> Figure:
    """Build a static matplotlib SED in mJy, with upper limits for non-detections.

    The matplotlib counterpart of :func:`sed_figure` (identical detection /
    upper-limit split): detections are points with flux and wavelength error
    bars (flagged bands in a contrasting colour); non-detections are downward
    upper-limit arrows at ``flux + upper_limit_sigma * error`` (drawn with
    matplotlib's native ``errorbar(uplims=True)``).

    Parameters
    ----------
    measurements : sequence of BandPhotometry
        Per-band measurements to plot.
    include_flagged : bool, default True
        Include bands flagged for bad pixels, shown in a contrasting colour.
    detection_snr : float, default 2.0
        Minimum SNR for a detection; must be >= 1.
    upper_limit_sigma : float, default 2.0
        Non-detections are drawn at ``flux + upper_limit_sigma * error``; > 0.
    title : str, optional
        Plot title. Defaults to ``"Aperture SED"``.
    size : int, default 680
        Figure width in pixels; the height follows a fixed aspect ratio.
    save : str or pathlib.Path, optional
        If given, write the figure to this path with ``savefig``.

    Returns
    -------
    matplotlib.figure.Figure
        The assembled figure.

    Raises
    ------
    ValueError
        Same conditions as :func:`_classify_points`.
    """
    import matplotlib.pyplot as plt

    detections, uppers = _classify_points(
        measurements,
        include_flagged=include_flagged,
        detection_snr=detection_snr,
        upper_limit_sigma=upper_limit_sigma,
    )

    width_in = size / 100.0
    fig, ax = plt.subplots(
        figsize=(width_in, width_in * _MPL_ASPECT), layout="constrained"
    )

    for flagged, color, label in (
        (False, "#1f6f8b", None),
        (True, "#c43b3b", "flagged"),
    ):
        group = [m for m in detections if m.flagged == flagged]
        if not group:
            continue
        x = np.asarray([m.wavelength for m in group], dtype=float)
        y = np.asarray([m.flux_mjy for m in group], dtype=float)
        yerr = np.asarray(
            [np.nan if m.error_mjy is None else m.error_mjy for m in group], dtype=float
        )
        xl = np.asarray(
            [
                0.0 if m.wavelength_error is None else m.wavelength_error[0]
                for m in group
            ],
            dtype=float,
        )
        xr = np.asarray(
            [
                0.0 if m.wavelength_error is None else m.wavelength_error[1]
                for m in group
            ],
            dtype=float,
        )
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            xerr=np.vstack([xl, xr]),
            fmt="o",
            color=color,
            ecolor="#8a9399",
            elinewidth=1.0,
            capsize=2,
            markersize=6,
            markeredgecolor="white",
            label=label,
        )

    if uppers:
        x = np.asarray([m.wavelength for m in uppers], dtype=float)
        limit = np.asarray(
            [m.flux_mjy + upper_limit_sigma * m.error_mjy for m in uppers], dtype=float
        )
        ax.errorbar(
            x,
            limit,
            yerr=np.abs(limit) * 0.3,
            uplims=True,
            fmt="o",
            color="#7a7a7a",
            markersize=5,
        )

    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("Flux (mJy)")
    ax.set_title(title or "Aperture SED")
    if any(m.flagged for m in detections):
        ax.legend()
    if save is not None:
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig
