"""Fit-time bad-pixel masking: manual regions and automatic outlier detection.

Both feed :meth:`~noobfriend.inference.spectrum.LineFitSetup.with_mask`, whose
result is the effective ``mask_excluded`` honoured by :func:`select_window`:

* :func:`resolve_mask` turns the polymorphic ``with_mask`` argument -- ``None`` /
  ``"auto"`` / a list of ``[lo, hi]`` wavelength intervals / a length-N boolean
  array -- into a full-length boolean mask (``True`` = excluded).
* :func:`compute_auto_mask` flags artifacts and unrelated emission *outside the
  line cores*: a robust continuum sigma-clip that protects a signal-driven core
  (the initial-guess line model above the noise, plus a small per-line floor) and
  masks isolated spikes (``>= sigma_isolated``) and resolved runs (``>=
  sigma_run`` over a run spanning about the LSF, which subsumes "a peak wider
  than the LSF").

This module is internal; reached only through ``LineFitSetup.with_mask``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from noobfriend.inference.spectrum._init import (
    _robust_std,
    component_geometry,
    initial_estimates,
)
from noobfriend.inference.spectrum._window import C_KMS, select_window

if TYPE_CHECKING:
    from noobfriend.inference.spectrum._setup import LineFitSetup

_SQRT_2PI: float = float(np.sqrt(2.0 * np.pi))

logger = logging.getLogger(__name__)


def resolve_mask(setup: LineFitSetup, mask: Any, **auto_kw: Any) -> np.ndarray | None:
    """Normalise a ``with_mask`` argument into a full-length boolean mask.

    Parameters
    ----------
    setup : LineFitSetup
        The setup (for the spectrum length and, when ``mask="auto"``, the model).
    mask : None or "auto" or array_like
        ``None`` clears the fit-time layer; ``"auto"`` runs
        :func:`compute_auto_mask`; a 2-D ``[[lo, hi], ...]`` is read as
        wavelength intervals; a 1-D length-N array is a boolean mask.
    **auto_kw
        Tuning forwarded to :func:`compute_auto_mask`; only valid with
        ``mask="auto"``.

    Returns
    -------
    numpy.ndarray or None
        The boolean mask (``True`` = excluded), or ``None`` for no layer.

    Raises
    ------
    ValueError
        On an unknown spec, a wrong-length boolean array, or ``auto_kw`` given
        without ``mask="auto"``.
    """
    if mask is None:
        if auto_kw:
            raise ValueError("auto-mask tuning is only valid with mask='auto'.")
        return None
    if isinstance(mask, str):
        if mask != "auto":
            raise ValueError(f"unknown mask spec {mask!r}; use 'auto'.")
        return compute_auto_mask(setup, **auto_kw)
    if auto_kw:
        raise ValueError("auto-mask tuning is only valid with mask='auto'.")

    n = int(setup.spectrum.wavelength.size)
    arr = np.asarray(mask)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return _intervals_to_mask(setup.spectrum.wavelength, arr)
    if arr.ndim == 1:
        m = np.asarray(mask, dtype=bool)
        if m.shape[0] != n:
            raise ValueError(f"mask array length {m.shape[0]} != {n} pixels.")
        return m
    raise ValueError(
        "mask must be None, 'auto', a list of [lo, hi] intervals, or a "
        "length-N bool array."
    )


def _intervals_to_mask(wavelength: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    """Mark every pixel falling inside any ``[lo, hi]`` wavelength interval."""
    wl = np.asarray(wavelength, dtype=float)
    out = np.zeros(wl.shape, dtype=bool)
    for pair in intervals:
        lo, hi = float(pair[0]), float(pair[1])
        if lo > hi:
            lo, hi = hi, lo
        out |= (wl >= lo) & (wl <= hi)
    return out


def _flag_runs(
    candidate: np.ndarray, absz: np.ndarray, run_len: int, sigma_isolated: float
) -> np.ndarray:
    """Flag maximal runs of ``candidate`` pixels by length or peak strength.

    A run spanning ``>= run_len`` pixels is flagged outright (a resolved feature);
    a shorter run is flagged only if its peak ``|z|`` reaches ``sigma_isolated``
    (a strong, near-point spike). ``candidate`` pixels already clear the lower
    ``sigma_run`` threshold.
    """
    out = np.zeros_like(candidate)
    n = candidate.size
    i = 0
    while i < n:
        if not candidate[i]:
            i += 1
            continue
        j = i
        while j < n and candidate[j]:
            j += 1
        if (j - i) >= run_len or float(absz[i:j].max()) >= sigma_isolated:
            out[i:j] = True
        i = j
    return out


def compute_auto_mask(
    setup: LineFitSetup,
    *,
    window: tuple[float, float] | None = None,
    pad_kms: float = 5000.0,
    continuum_degree: int = 1,
    tau: float = 1.0,
    sigma_isolated: float = 5.0,
    sigma_run: float = 3.0,
    run_lsf_frac: float = 0.5,
    iters: int = 2,
) -> np.ndarray:
    """Flag artifacts and unrelated emission outside the fitted line cores.

    Protects a signal-driven *core* region -- the initial-guess line model above
    ``tau`` times the noise, plus a small per-line floor -- then sigma-clips the
    remaining continuum pixels against a robust low-order continuum fit.

    Parameters
    ----------
    setup : LineFitSetup
        The compiled model (for the lines and spectrum).
    window, pad_kms, continuum_degree
        Mirror :meth:`LineFitSetup.run`; pass the same window you will fit on.
    tau : float, default 1.0
        Core threshold: pixels where the initial-guess line model exceeds
        ``tau * sigma`` are protected.
    sigma_isolated : float, default 5.0
        Threshold for short (near-point) deviations -- cosmic rays / hot pixels.
    sigma_run : float, default 3.0
        Lower threshold for *runs* of consecutive deviant pixels.
    run_lsf_frac : float, default 0.5
        A run is treated as resolved (and flagged at ``sigma_run``) once it spans
        ``run_lsf_frac`` of the LSF in pixels; falls back to 3 pixels when ``R``
        is unknown.
    iters : int, default 2
        Continuum-refit / re-clip iterations.

    Returns
    -------
    numpy.ndarray
        Full-length boolean mask (``True`` = flagged), ``False`` outside the
        window, on the line cores, and on already-excluded pixels.
    """
    spectrum = setup.spectrum
    comps = setup.components
    mask_win, _ = select_window(spectrum, comps, window=window, pad_kms=pad_kms)
    idx = np.flatnonzero(mask_win)
    wl = spectrum.wavelength[idx].astype(float)
    fl = spectrum.flux[idx].astype(float)
    n = wl.size
    degree = int(continuum_degree) if n >= 4 else 0

    geom = component_geometry(comps, spectrum.z)
    init = initial_estimates(spectrum, comps, wl, fl, degree=degree)

    # Initial-guess line model (lines only) and continuum, for the core region.
    line = np.zeros_like(wl)
    for comp in comps:
        g = geom[comp.id]
        amp = init.component_flux[comp.id] / (g["sigma_w"] * _SQRT_2PI)
        line += g["sign"] * amp * np.exp(-0.5 * ((wl - g["mu"]) / g["sigma_w"]) ** 2)
    cont0 = init.c0 + (init.c1 * (wl - init.lambda0) if degree >= 1 else 0.0)
    sigma0 = _robust_std(fl - cont0 - line) or 1.0

    dwl = float(np.median(np.diff(wl))) if n > 1 else 1.0
    lsf_kms = None if spectrum.R is None else C_KMS / spectrum.R

    # Core: signal-driven (line model > tau*sigma) plus a small per-line floor so
    # a faint, near-undetected line core is never clipped.
    core = np.abs(line) > tau * sigma0
    floor_wl = 3.0 * (lsf_kms or 0.0) / C_KMS  # fractional half-width
    for comp in comps:
        mu = geom[comp.id]["mu"]
        half = max(mu * floor_wl, 2.0 * dwl)
        core |= np.abs(wl - mu) <= half

    if lsf_kms is not None:
        lsf_pix = (lsf_kms / C_KMS) * float(np.median(wl)) / dwl
        run_len = max(3, int(round(run_lsf_frac * lsf_pix)))
    else:
        run_len = 3

    flagged = np.zeros(n, dtype=bool)
    for _ in range(max(1, iters)):
        cont_region = (~core) & (~flagged)
        if int(cont_region.sum()) < degree + 2:
            break
        coef = np.polyfit(wl[cont_region], fl[cont_region], degree)
        resid = fl - np.polyval(coef, wl)
        sigma = _robust_std(resid[cont_region]) or 1.0
        absz = np.abs(resid) / sigma
        candidate = (~core) & (absz >= sigma_run)
        new = _flag_runs(candidate, absz, run_len, sigma_isolated)
        if np.array_equal(new, flagged):
            break
        flagged = new

    full = np.zeros(spectrum.wavelength.shape, dtype=bool)
    full[idx[flagged]] = True
    n_runs = int(np.count_nonzero(flagged[1:] & ~flagged[:-1])) + int(
        flagged[0] if n else 0
    )
    logger.info("auto_mask: flagged %d px in %d run(s)", int(full.sum()), n_runs)
    return full
