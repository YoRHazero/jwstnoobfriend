"""Matplotlib figures for inspecting a setup (pre-fit) and a result (post-fit).

Two entry points, both composing the public
:func:`~noobfriend.core.display.plot.draw_spectrum` engine onto axes this module
owns (so the data, uncertainty band, and model overlays render identically to
the rest of the library):

* :func:`preview_setup` -- the fitting window and the *initial-guess* model over
  the data, so a bad window or starting point is caught before sampling. It is
  pure NumPy/matplotlib (no PyMC), usable without the ``mcmc`` extra.
* :func:`plot_result` -- the data with the posterior model band, each
  component's median decomposition, and an optional standardized-residual panel.

Reached through :meth:`LineFitSetup.plot` and :meth:`LineFitResult.plot`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from noobfriend.inference.spectrum._result import LineFitResult
    from noobfriend.inference.spectrum._setup import LineFitSetup

_SQRT_2PI: float = math.sqrt(2.0 * math.pi)

#: Decomposition line colours (skip ``C3``, reserved for the total model).
_COMPONENT_COLORS: tuple[str, ...] = (
    "C0",
    "C1",
    "C2",
    "C4",
    "C5",
    "C6",
    "C7",
    "C8",
    "C9",
)


def _initial_params(setup: LineFitSetup, init: Any) -> dict[str, dict[str, float]]:
    """Return per-component initial ``(sign, mu, sigma_w, amp, flux)`` for preview.

    Reuses the shared
    :func:`~noobfriend.inference.spectrum._init.component_geometry` and the
    lstsq per-component fluxes carried on ``init``.
    """
    from noobfriend.inference.spectrum._init import component_geometry

    geometry = component_geometry(setup.components, setup.spectrum.z)
    return {
        cid: {
            **g,
            "amp": init.component_flux[cid] / (g["sigma_w"] * _SQRT_2PI),
            "flux": init.component_flux[cid],
        }
        for cid, g in geometry.items()
    }


def preview_setup(
    setup: LineFitSetup,
    *,
    window: tuple[float, float] | None = None,
    pad_kms: float = 5000.0,
    continuum_degree: int = 1,
    size: int = 680,
    title: str | None = None,
    save: str | None = None,
) -> Figure:
    """Preview the fitting window and initial-guess model over the data.

    Parameters
    ----------
    setup : LineFitSetup
        The compiled model.
    window, pad_kms, continuum_degree
        Mirror :meth:`LineFitSetup.run` so the preview matches what the fit sees.
    size : int, default 680
        Figure width in pixels.
    title : str, optional
        Figure title.
    save : str, optional
        Path to write the figure to.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    from noobfriend.core.display.plot import draw_spectrum
    from noobfriend.inference.spectrum._init import initial_estimates
    from noobfriend.inference.spectrum._window import select_window

    spectrum = setup.spectrum
    comps = setup.components
    mask, (lo, hi) = select_window(spectrum, comps, window=window, pad_kms=pad_kms)
    wl_w = spectrum.wavelength[mask].astype(float)
    fl_w = spectrum.flux[mask].astype(float)
    degree = continuum_degree if wl_w.size >= 4 else 0
    init = initial_estimates(spectrum, comps, wl_w, fl_w, degree=degree)
    params = _initial_params(setup, init)

    def init_model(grid: np.ndarray) -> np.ndarray:
        grid = np.asarray(grid, dtype=float)
        total = init.c0 + (init.c1 * (grid - init.lambda0) if degree >= 1 else 0.0)
        total = np.full_like(grid, total) if np.isscalar(total) else total
        for p in params.values():
            total = total + p["sign"] * p["amp"] * np.exp(
                -0.5 * ((grid - p["mu"]) / p["sigma_w"]) ** 2
            )
        return total

    width_in = size / 100.0
    fig, ax = plt.subplots(figsize=(width_in, width_in * 0.45), layout="constrained")
    model = {
        "flux_func": init_model,
        "wavelength_range": (lo, hi),
        "label": "initial guess",
        "color": "C1",
    }
    draw_spectrum(
        ax,
        spectrum.wavelength,
        spectrum.flux,
        error=spectrum.error,
        colors="0.4",
        models=[model],
        x_label=f"Wavelength [{spectrum.wave_unit}]",
        y_label="Flux",
        title=title or "setup preview (pre-fit)",
    )
    ax.axvspan(lo, hi, color="C0", alpha=0.08, zorder=0)
    span = hi - lo
    ax.set_xlim(lo - 0.3 * span, hi + 0.3 * span)
    if save is not None:
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


def plot_result(
    result: LineFitResult,
    *,
    full: bool = False,
    residual: bool = True,
    decompose: bool = True,
    predictive: bool = False,
    size: int = 680,
    title: str | None = None,
    save: str | None = None,
) -> Figure:
    """Plot the data with the posterior model, decomposition, and residuals.

    Parameters
    ----------
    result : LineFitResult
        The fit to plot.
    full : bool, default False
        Plot the whole spectrum; otherwise just the fitted window.
    residual : bool, default True
        Add a shared-x standardized-residual panel.
    decompose : bool, default True
        Overlay each component's median curve on the continuum.
    predictive : bool, default False
        Shade the 95% posterior predictive band (model plus observation noise)
        behind the data, nested outside the 68% model band. Drawn with a fixed
        seed so the figure is reproducible; see
        :meth:`~noobfriend.inference.spectrum.LineFitResult.predictive_curve`.
    size : int, default 680
        Figure width in pixels.
    title : str, optional
        Figure title.
    save : str, optional
        Path to write the figure to.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    from noobfriend.core.display.plot import draw_spectrum

    spectrum = result.spectrum
    if full:
        wl, fl, er = spectrum.wavelength, spectrum.flux, spectrum.error
    else:
        wl, fl, er = result.window_wl, result.window_flux, result.window_error
    x_label = f"Wavelength [{spectrum.wave_unit}]"
    lo, hi = float(np.nanmin(wl)), float(np.nanmax(wl))
    grid = np.linspace(lo, hi, 512)

    models: list[dict[str, Any]] = [
        {
            "flux_func": result.flux_func,
            "wavelength_range": (lo, hi),
            "label": "model",
            "color": "C3",
        }
    ]
    if decompose and len(result.components) > 1:
        for i, comp in enumerate(result.components):
            models.append(
                {
                    "wavelength": grid,
                    "flux": result.component_curve(comp.id, grid),
                    "label": comp.id,
                    "color": _COMPONENT_COLORS[i % len(_COMPONENT_COLORS)],
                    "linestyle": "--",
                }
            )

    width_in = size / 100.0
    if residual:
        fig, (ax, ax_r) = plt.subplots(
            2,
            1,
            sharex=True,
            height_ratios=(3, 1),
            figsize=(width_in, width_in * 0.5),
            layout="constrained",
        )
        ax.tick_params(labelbottom=False)
    else:
        fig, ax = plt.subplots(
            figsize=(width_in, width_in * 0.45), layout="constrained"
        )

    # Drawn before draw_spectrum (which builds the legend) so it gets an entry,
    # and behind everything via zorder; the data points sit on top.
    if predictive:
        lo_p, _, hi_p = result.predictive_curve(grid, q=(2.5, 50.0, 97.5), seed=0)
        ax.fill_between(
            grid,
            lo_p,
            hi_p,
            color="C3",
            alpha=0.12,
            linewidth=0,
            zorder=0.5,
            label="95% predictive",
        )

    draw_spectrum(
        ax,
        wl,
        fl,
        error=er,
        colors="0.4",
        models=models,
        x_label="" if residual else x_label,
        y_label="Flux",
        title=title,
    )

    if residual:
        with np.errstate(invalid="ignore", divide="ignore"):
            std_res = (fl - result.model_curve(wl)[1]) / er
        ax_r.axhline(0.0, color="C3", lw=1.0)
        for k, ls in ((1.0, "--"), (2.0, ":")):
            ax_r.axhline(k, color="0.6", lw=0.8, ls=ls)
            ax_r.axhline(-k, color="0.6", lw=0.8, ls=ls)
        ax_r.step(wl, std_res, where="mid", color="0.3")
        ax_r.set_xlabel(x_label)
        ax_r.set_ylabel(r"$(d - m)/\sigma$")

    if save is not None:
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig
