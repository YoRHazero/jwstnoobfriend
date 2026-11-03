"""The no-argument figure of an MLE result."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from noobfriend.inference.spectrum.mle._result import NoobMLEResult

_COMPONENTS = ("#00798c", "#edae49", "#66a182", "#8d96a3", "#2e4057", "#a05195")


def plot_mle(result: NoobMLEResult) -> Figure:
    """Draw data, selected model, continuum and components, then the residual."""
    import matplotlib.pyplot as plt

    problem = result._problem  # noqa: SLF001 - same subpackage
    x = np.asarray([result.solution.x[name] for name in problem.names])
    wavelength = problem.wavelength
    grid = np.linspace(wavelength[0], wavelength[-1], 8 * wavelength.size)
    smooth = problem.model(x, grid)
    native = np.asarray(problem.model(x).total, dtype=float)
    figure, (top, bottom) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(9.0, 5.0),
        height_ratios=(3.0, 1.0),
        layout="constrained",
    )
    top.step(
        wavelength, problem.data, where="mid", color="#1a1a1a", lw=0.9, label="data"
    )
    top.fill_between(
        wavelength,
        problem.data - problem.error,
        problem.data + problem.error,
        step="mid",
        color="#1a1a1a",
        alpha=0.12,
        lw=0,
    )
    continuum = np.asarray(smooth.continuum + 0.0 * grid, dtype=float)
    top.plot(grid, np.asarray(smooth.total), color="#d1495b", lw=1.2, label="model")
    top.plot(grid, continuum, color="#7a7a7a", lw=1.0, ls="--", label="continuum")
    for index, (name, curve) in enumerate(smooth.components.items()):
        color = _COMPONENTS[index % len(_COMPONENTS)]
        top.plot(grid, continuum + np.asarray(curve), color=color, lw=1.0, label=name)
    top.set_ylabel("Flux")
    top.set_title(
        f"{problem.frame_id}   chi2 = {result.solution.chi2:.1f} / {wavelength.size}",
        loc="left",
        fontsize=10,
    )
    top.legend(fontsize=7, loc="upper right", ncols=2)
    bottom.step(
        wavelength,
        (problem.data - native) / problem.error,
        where="mid",
        color="#1a1a1a",
        lw=0.8,
    )
    bottom.axhline(0.0, color="#d1495b", lw=0.8)
    bottom.set_ylabel("Resid. (sigma)")
    bottom.set_xlabel(f"Observed wavelength ({result.workspace.spectrum.unit})")
    return figure
