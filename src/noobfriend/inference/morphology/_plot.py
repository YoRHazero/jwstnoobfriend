"""Matplotlib montage for previewing and posterior-checking a decomposition.

One row per band: the data, each component's model image, the summed model, and
the normalised residual. The same view serves the pre-sampling sanity check
(:meth:`MorphModel.preview`) and the posterior-predictive check
(:meth:`MorphResult.plot`) -- a residual that looks like noise is the goal. This
module is internal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from noobfriend.inference.morphology._model import Preview


def montage(preview: Preview) -> Any:
    """Render a data / components / model / residual montage.

    Parameters
    ----------
    preview : Preview
        The rendered scene from :meth:`MorphModel.preview`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    comp_names = list(next(iter(preview.components.values())))
    col_titles = ["data", *comp_names, "model", "residual"]
    n_rows = len(preview.bands)
    n_cols = len(col_titles)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.2 * n_cols, 2.2 * n_rows), squeeze=False
    )
    for r, band in enumerate(preview.bands):
        data = preview.data[band]
        vmin, vmax = _limits(data)
        panels = [
            data,
            *(preview.components[band][c] for c in comp_names),
            preview.total[band],
        ]
        for c, img in enumerate(panels):
            _show(axes[r][c], img, vmin, vmax)
        _show_residual(axes[r][n_cols - 1], preview.residual[band])
        axes[r][0].set_ylabel(band, fontsize=9)

    for c, title in enumerate(col_titles):
        axes[0][c].set_title(title, fontsize=9)
    fig.tight_layout()
    return fig


def _limits(data: np.ndarray) -> tuple[float, float]:
    """Robust display limits from the finite pixels."""
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if lo == hi:
        hi = lo + 1.0
    return float(lo), float(hi)


def _show(ax: Any, img: np.ndarray, vmin: float, vmax: float) -> None:
    ax.imshow(img, origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
    ax.set_xticks([])
    ax.set_yticks([])


def _show_residual(ax: Any, residual: np.ndarray) -> None:
    finite = residual[np.isfinite(residual)]
    scale = float(np.percentile(np.abs(finite), 99.0)) if finite.size else 1.0
    scale = scale or 1.0
    ax.imshow(residual, origin="lower", vmin=-scale, vmax=scale, cmap="RdBu_r")
    ax.set_xticks([])
    ax.set_yticks([])
