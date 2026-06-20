"""Slice stamps from cached cutouts and drive the noobase ePSF engine.

The two builds are split because the noobase API makes them asymmetric: the
oversampled core comes from :func:`noobase.image.psf.build_epsf` on its own, but
the wings come from :func:`noobase.image.psf.build_extended_psf`, which **requires
a core to stitch onto** -- so :func:`build_wings` takes a :class:`CorePsf` and
cannot stand alone. Both slice their stamp (a small core stamp or a larger native
wing stamp) from the *same* cached cutouts via :func:`noobase.image.build_stamp`,
which also reports the sub-pixel ``delta`` (the phase) consumed by the engine.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from noobase.image import build_stamp
from noobase.image.psf import build_epsf, build_extended_psf

if TYPE_CHECKING:
    from noobfriend.extraction.psf._store import CutoutStore


@dataclass(frozen=True)
class CorePsf:
    """An oversampled effective-PSF core from :func:`build_core`.

    Attributes
    ----------
    core : numpy.ndarray
        ``(oversample * core_size, oversample * core_size)`` oversampled core.
    oversample : int
        Oversampling factor (odd).
    core_size : int
        Detector-pixel edge length of the core stamps (odd).
    flux : numpy.ndarray
        Per-stamp fitted flux from ``build_epsf``.
    converged : bool
        Whether the super-resolution iteration converged.
    n_stars : int
        Number of stamps (detections) that entered the build.
    n_sources : int
        Number of distinct physical sources behind those stamps.
    phase_grid : numpy.ndarray
        ``(oversample, oversample)`` occupancy of the stamps' sub-pixel phases.
    """

    core: np.ndarray
    oversample: int
    core_size: int
    flux: np.ndarray
    converged: bool
    n_stars: int
    n_sources: int
    phase_grid: np.ndarray

    def render(self, size: int | None = None, *, normalize: bool = True) -> np.ndarray:
        """Render a native-resolution core image (see :func:`render_core`)."""
        from noobfriend.extraction.psf._render import render_core

        return render_core(self.core, self.oversample, size=size, normalize=normalize)

    def save(self, path: str | Path, *, overwrite: bool = False) -> None:
        """Write this core to a multi-extension FITS file (see :func:`load`)."""
        from noobfriend.extraction.psf._persist import save_core

        save_core(self, path, overwrite=overwrite)

    @classmethod
    def load(cls, path: str | Path) -> "CorePsf":
        """Read a core back from a file written by :meth:`save`."""
        from noobfriend.extraction.psf._persist import load_core

        return load_core(path)


@dataclass(frozen=True)
class EffectivePsf:
    """A self-contained hybrid effective PSF: oversampled core + native wing.

    Built by :func:`build_wings` (which drives noobase's stitch), then held as
    plain arrays -- no live noobase handle -- so it renders, convolves, and
    round-trips through :meth:`save` / :meth:`load` without re-stitching (which
    would re-process the already-baked wing).

    The noobase sampling contract is
    ``recon(r) = (1 - f_wing(r)) * core_sample(r) + wing_sample(r)`` with a
    raised-cosine ``f_wing`` centred on :attr:`match_radius` of full width
    :attr:`feather_width` (native pixels). The stored :attr:`wing` already has
    ``f_wing`` and the scalar match baked in (it is ~0 in the core region), and
    both planes are encircled-energy normalised within :attr:`ee_aperture_radius`.

    Attributes
    ----------
    core : CorePsf
        The oversampled core the wings were stitched onto (provenance).
    core_plane : numpy.ndarray
        ``(oversample * core_size, oversample * core_size)`` encircled-energy
        normalised oversampled core plane of the stitch -- the contract's
        ``core_sample`` source (differs from ``core.core`` by the EE gauge).
    wing : numpy.ndarray
        ``(wing_size, wing_size)`` native wing plane (``f_wing`` + scalar baked
        in, encircled-energy normalised).
    oversample : int
        Oversampling factor (inherited from ``core``).
    match_radius, feather_width, ee_aperture_radius : float
        Stitch geometry in native pixels: the feather ring centre, the
        raised-cosine feather full width, and the EE-normalisation aperture.
    wing_size : int
        Detector-pixel edge length of the native wing stamps.
    n_wing_stars : int
        Number of wing stamps that entered the build.
    """

    core: CorePsf
    core_plane: np.ndarray
    wing: np.ndarray
    oversample: int
    match_radius: float
    feather_width: float
    ee_aperture_radius: float
    wing_size: int
    n_wing_stars: int

    def render(self, size: int | None = None, *, normalize: bool = True) -> np.ndarray:
        """Render a native-resolution hybrid image (see :func:`render_effective`)."""
        from noobfriend.extraction.psf._render import render_effective

        return render_effective(self, size=size, normalize=normalize)

    def save(self, path: str | Path, *, overwrite: bool = False) -> None:
        """Write this hybrid PSF to a multi-extension FITS file (see :meth:`load`)."""
        from noobfriend.extraction.psf._persist import save_effective

        save_effective(self, path, overwrite=overwrite)

    @classmethod
    def load(cls, path: str | Path) -> "EffectivePsf":
        """Read a hybrid PSF back from a file written by :meth:`save`."""
        from noobfriend.extraction.psf._persist import load_effective

        return load_effective(path)


def _phase_grid(delta: np.ndarray, oversample: int) -> np.ndarray:
    """Bin per-stamp ``delta`` onto the ``(oversample, oversample)`` phase grid."""
    grid = np.zeros((oversample, oversample), dtype=int)
    if delta.shape[0] == 0:
        return grid
    cells = np.clip(np.floor((delta + 0.5) * oversample).astype(int), 0, oversample - 1)
    np.add.at(grid, (cells[:, 0], cells[:, 1]), 1)
    return grid


def _stamp_stack(
    cutouts: "CutoutStore", size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Slice a ``size`` stamp from each cutout via ``build_stamp``.

    Returns ``(data, delta, weight)`` stacks, or ``None`` if no cutout yields a
    usable stamp. ``weight`` is ``valid / error**2`` (DQ-flagged/invalid pixels
    zero), or the validity mask alone when a cutout carries no error.
    """
    stamps: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for cut in cutouts:
        error = None if cut.err is None else cut.err.astype(float)
        result = build_stamp(cut.data.astype(float), size, error=error, mask=cut.bad)
        if result is None:
            continue
        valid = np.asarray(result.valid)
        if result.error is not None:
            err_stamp = np.asarray(result.error, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = np.where(valid & (err_stamp > 0), 1.0 / err_stamp**2, 0.0)
        else:
            weight = valid.astype(float)
        stamps.append(np.asarray(result.stamp, dtype=float))
        deltas.append(np.asarray(result.delta, dtype=float))
        weights.append(weight)
    if not stamps:
        return None
    return np.stack(stamps), np.stack(deltas), np.stack(weights)


def build_core(
    cutouts: "CutoutStore",
    n_sources: int,
    *,
    oversample: int = 3,
    core_size: int = 21,
    max_iter: int = 50,
    tol: float = 1e-4,
    residual_reweight: str = "huber",
) -> CorePsf:
    """Build the oversampled core from cached cutouts via ``build_epsf``.

    Parameters
    ----------
    cutouts : CutoutStore
        Cached cutouts (one per detection) to slice core stamps from.
    n_sources : int
        Number of distinct physical sources behind ``cutouts`` (for the result).
    oversample : int, default 3
        Oversampling factor; must be odd (a noobase precondition).
    core_size : int, default 21
        Odd edge length of the core stamps sliced from each cutout.
    max_iter, tol : int, float
        Super-resolution iteration controls forwarded to ``build_epsf``.
    residual_reweight : str, default "huber"
        Per-pixel M-estimator (``"none"`` / ``"huber"`` / ``"tukey"``).

    Returns
    -------
    CorePsf

    Raises
    ------
    ValueError
        If ``oversample`` or ``core_size`` is not a positive odd integer, or no
        cutout yields a usable stamp.
    """
    if oversample % 2 == 0 or oversample <= 0:
        raise ValueError(
            f"oversample must be a positive odd integer; got {oversample!r}."
        )
    if core_size % 2 == 0 or core_size <= 0:
        raise ValueError(
            f"core_size must be a positive odd integer; got {core_size!r}."
        )
    stack = _stamp_stack(cutouts, core_size)
    if stack is None:
        raise ValueError("no usable core stamps; nothing to build a PSF from.")
    data, delta, weight = stack
    result = build_epsf(
        data,
        delta,
        oversample,
        weight=weight,
        max_iter=max_iter,
        tol=tol,
        residual_reweight=residual_reweight,  # type: ignore[arg-type]
    )
    return CorePsf(
        core=np.asarray(result.epsf),
        oversample=oversample,
        core_size=core_size,
        flux=np.asarray(result.flux),
        converged=bool(result.converged),
        n_stars=int(data.shape[0]),
        n_sources=n_sources,
        phase_grid=_phase_grid(delta, oversample),
    )


def build_wings(
    cutouts: "CutoutStore",
    core: CorePsf,
    *,
    wing_size: int = 51,
    **extended_kwargs: object,
) -> EffectivePsf:
    """Stitch native wings onto ``core`` via ``build_extended_psf``.

    Parameters
    ----------
    cutouts : CutoutStore
        Cached cutouts (typically a brighter, isolated subset) to slice the
        larger native wing stamps from.
    core : CorePsf
        The oversampled core to stitch onto; its ``oversample`` is reused.
    wing_size : int, default 51
        Odd edge length of the native wing stamps.
    **extended_kwargs
        Forwarded to ``build_extended_psf`` (e.g. ``match_radius``,
        ``feather_width``, ``ee_aperture_radius``).

    Returns
    -------
    EffectivePsf

    Raises
    ------
    ValueError
        If ``wing_size`` is not a positive odd integer, or no cutout yields a
        usable wing stamp.

    Notes
    -----
    The stitch couples ``core_size`` to ``match_radius``: ``match_radius *
    oversample`` must sit inside the core's half-extent, so with the default
    ``match_radius`` of 8 the core ``core_size`` must be roughly 17 or more.
    """
    if wing_size % 2 == 0 or wing_size <= 0:
        raise ValueError(
            f"wing_size must be a positive odd integer; got {wing_size!r}."
        )
    stack = _stamp_stack(cutouts, wing_size)
    if stack is None:
        raise ValueError("no usable wing stamps; nothing to build wings from.")
    data, delta, weight = stack
    built = build_extended_psf(
        data,
        delta,
        core.core,
        core.oversample,
        wing_weight=weight,
        **extended_kwargs,  # type: ignore[arg-type]
    )
    extended = built.extended
    return EffectivePsf(
        core=core,
        core_plane=np.asarray(extended.core),
        wing=np.asarray(extended.wing),
        oversample=int(extended.oversample),
        match_radius=float(extended.match_radius),
        feather_width=float(extended.feather_width),
        ee_aperture_radius=float(extended.ee_aperture_radius),
        wing_size=wing_size,
        n_wing_stars=int(data.shape[0]),
    )
