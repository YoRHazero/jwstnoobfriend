"""Static figures for MCMC results: decomposition panels and corner plots."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from noobfriend.inference.spectrum.mcmc._decompose import (
        Band,
        FrameDecomposition,
        NoobSpectrumDecomposition,
    )
    from noobfriend.inference.spectrum.mcmc._result import NoobMCMCResult

_DATA = "#1a1a1a"
_MODEL = "#d1495b"
_CONTINUUM = "#7a7a7a"
_MASKED = "#bdbdbd"
_COMPONENTS = ("#00798c", "#edae49", "#66a182", "#8d96a3", "#2e4057", "#a05195")
_HEIGHTS = {"image": 1.4, "spectrum": 3.0, "residual": 1.0}


def plot_decomposition(
    decomposition: NoobSpectrumDecomposition,
    *,
    continuum: str,
    normalize: bool,
    offset: float,
    ylog: bool,
    xlim: tuple[float, float] | None,
) -> Figure:
    """Draw every frame of a decomposition (see its ``plot`` method)."""
    import matplotlib.pyplot as plt

    if continuum not in ("include", "subtract"):
        raise ValueError(
            f"continuum must be 'include' or 'subtract'; got {continuum!r}."
        )
    frames = list(decomposition.values())
    if xlim is None:
        xlim = (
            min(float(f.frame.obs[f.fitted][0]) for f in frames),
            max(float(f.frame.obs[f.fitted][-1]) for f in frames),
        )
    layout: list[tuple[str, FrameDecomposition]] = []
    for frame in frames:
        if frame.frame.source_2d is not None:
            layout.append(("image", frame))
        layout += [("spectrum", frame), ("residual", frame)]
    heights = [_HEIGHTS[kind] for kind, _ in layout]
    figure, axes = plt.subplots(
        len(layout),
        1,
        sharex=True,
        figsize=(9.0, 1.1 * sum(heights)),
        height_ratios=heights,
        layout="constrained",
        squeeze=False,
    )
    titled: set[str] = set()
    for ax, (kind, frame) in zip(axes[:, 0], layout, strict=True):
        if frame.frame.id not in titled:
            titled.add(frame.frame.id)
            ax.set_title(
                f"{frame.frame.id}   chi2 = {frame.chi2:.1f} / {frame.n_fitted}",
                loc="left",
                fontsize=10,
            )
        if kind == "image":
            _draw_image(ax, frame)
        elif kind == "spectrum":
            _draw_spectrum(ax, frame, continuum, normalize, offset, ylog, xlim)
        else:
            _draw_residual(ax, frame, xlim)
    axes[-1, 0].set_xlim(*xlim)
    axes[-1, 0].set_xlabel(f"Observed wavelength ({decomposition.unit})")
    return figure


def plot_corner(result: NoobMCMCResult, params: Sequence[str] | None) -> Figure:
    """Corner plot of posterior parameters by flat name (ArviZ ``plot_pair``)."""
    import arviz as az

    if params is None:
        lines = {line.name for line in result.workspace.lines}
        params = [
            row.name for row in result.workspace.parameters.free() if row.owner in lines
        ]
        if not params:
            raise ValueError("the fit has no free line parameter; pass params=[...].")
    posterior = result.idata["posterior"].dataset
    missing = [name for name in params if name not in posterior.data_vars]
    if missing:
        raise KeyError(f"not in the posterior: {missing}.")
    collection = az.plot_pair(
        result.idata, var_names=list(params), group="posterior", backend="matplotlib"
    )
    return collection.viz["figure"].item()


def _draw_spectrum(
    ax: Any,
    frame: FrameDecomposition,
    continuum: str,
    normalize: bool,
    offset: float,
    ylog: bool,
    xlim: tuple[float, float],
) -> None:
    data = frame.frame
    obs = np.asarray(data.obs)
    inside = (obs >= xlim[0]) & (obs <= xlim[1])
    base_native = np.interp(obs, frame.grid, frame.continuum.median)
    base_grid = frame.continuum.median
    subtract = continuum == "subtract"
    flux = np.asarray(data.flux) - (base_native if subtract else 0.0)
    total = _shift(frame.total, base_grid if subtract else 0.0)
    factor = 1.0
    if normalize:
        peak = float(np.max(np.abs(total.median)))
        factor = 1.0 / peak if peak > 0 else 1.0

    def show(values: np.ndarray) -> np.ndarray:
        shown = values * factor + offset
        return np.where(shown > 0, shown, np.nan) if ylog else shown

    error = np.asarray(data.error) * factor
    good = frame.fitted & inside
    masked = ~frame.fitted & inside
    ax.step(obs[good], show(flux[good]), where="mid", color=_DATA, lw=0.9, label="data")
    ax.fill_between(
        obs[good],
        show(flux[good] - error[good]),
        show(flux[good] + error[good]),
        step="mid",
        color=_DATA,
        alpha=0.12,
        lw=0,
    )
    if masked.any():
        ax.plot(
            obs[masked], show(flux[masked]), ".", color=_MASKED, ms=3, label="masked"
        )
    ax.fill_between(
        frame.grid, show(total.lower), show(total.upper), color=_MODEL, alpha=0.25, lw=0
    )
    ax.plot(frame.grid, show(total.median), color=_MODEL, lw=1.2, label="model")
    if not subtract:
        ax.plot(
            frame.grid,
            show(base_grid),
            color=_CONTINUUM,
            lw=1.0,
            ls="--",
            label="continuum",
        )
    for index, (name, band) in enumerate(frame.components.items()):
        color = _COMPONENTS[index % len(_COMPONENTS)]
        shifted = _shift(band, -base_grid if not subtract else 0.0)
        ax.fill_between(
            frame.grid,
            show(shifted.lower),
            show(shifted.upper),
            color=color,
            alpha=0.2,
            lw=0,
        )
        ax.plot(frame.grid, show(shifted.median), color=color, lw=1.0, label=name)
    if ylog:
        ax.set_yscale("log")
    ax.set_ylabel("Flux" + (" (normalized)" if normalize else ""))
    ax.legend(fontsize=7, loc="upper right", ncols=2)


def _draw_residual(
    ax: Any, frame: FrameDecomposition, xlim: tuple[float, float]
) -> None:
    obs = np.asarray(frame.frame.obs)
    ax.step(obs, frame.residual, where="mid", color=_DATA, lw=0.8)
    ax.axhline(0.0, color=_MODEL, lw=0.8)
    for level in (-1.0, 1.0):
        ax.axhline(level, color=_CONTINUUM, lw=0.6, ls=":")
    ax.set_ylabel("Resid. (sigma)")


def _draw_image(ax: Any, frame: FrameDecomposition) -> None:
    from noobfriend.core.display.plot import draw_spectrum2d

    source = frame.frame.source_2d
    draw_spectrum2d(
        ax,
        frame.frame.obs,
        source.flux,
        spatial=source.spatial,
        collapse_rows=source.collapse_window,
    )
    ax.set_ylabel("Spatial")


def _shift(band: Band, amount: Any) -> Band:
    from noobfriend.inference.spectrum.mcmc._decompose import Band

    return Band(band.median - amount, band.lower - amount, band.upper - amount)
