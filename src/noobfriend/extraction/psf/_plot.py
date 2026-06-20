"""Matplotlib diagnostic figures for built PSFs.

Reached through :meth:`CorePsf.plot` and :meth:`EffectivePsf.plot`: a rendered
native PSF image (asinh stretch, so the faint wings show alongside the bright
core) next to its azimuthally-averaged radial profile. For a hybrid PSF the
core<->wing crossover (``match_radius``) is marked on the profile.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _radial_profile(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(radius, azimuthal mean)`` binned to integer radius from centre."""
    centre = (np.asarray(image.shape) - 1) / 2.0
    yy, xx = np.indices(image.shape)
    r = np.hypot(yy - centre[0], xx - centre[1]).astype(int)
    counts = np.bincount(r.ravel())
    sums = np.bincount(r.ravel(), weights=image.ravel())
    return np.arange(counts.size), sums / np.maximum(counts, 1)


def plot_psf(
    image: np.ndarray,
    *,
    title: str | None = None,
    match_radius: float | None = None,
    save: str | Path | None = None,
) -> "Figure":
    """Plot a rendered PSF image and its azimuthal radial profile.

    Parameters
    ----------
    image : numpy.ndarray
        A 2-D, centred native PSF image (e.g. from ``psf.render()``).
    title : str, optional
        Figure title.
    match_radius : float, optional
        If given (native pixels), drawn as a dashed line on the radial profile to
        mark the core<->wing crossover of a hybrid PSF.
    save : str or Path, optional
        If given, write the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import AsinhNorm

    image = np.asarray(image, dtype=float)
    peak = float(np.nanmax(np.abs(image))) or 1.0

    fig, (ax_img, ax_rad) = plt.subplots(1, 2, figsize=(9.5, 4.2), layout="constrained")

    norm = AsinhNorm(linear_width=peak * 1e-3, vmin=float(np.nanmin(image)), vmax=peak)
    handle = ax_img.imshow(image, origin="lower", cmap="magma", norm=norm)
    fig.colorbar(handle, ax=ax_img, fraction=0.046, pad=0.04)
    ax_img.set_title("PSF image (asinh)")
    ax_img.set_xlabel("x [native pix]")
    ax_img.set_ylabel("y [native pix]")

    radii, profile = _radial_profile(image)
    ax_rad.plot(radii, profile / peak, color="C0", lw=1.5)
    ax_rad.set_yscale("log")
    ax_rad.set_xlabel("radius [native pix]")
    ax_rad.set_ylabel("azimuthal mean / peak")
    ax_rad.set_title("radial profile")
    ax_rad.grid(True, which="both", alpha=0.2)
    if match_radius is not None:
        ax_rad.axvline(
            match_radius,
            ls="--",
            color="C1",
            lw=1.0,
            label=f"match_radius = {match_radius:g}",
        )
        ax_rad.legend(frameon=False)

    if title:
        fig.suptitle(title)
    if save is not None:
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig
