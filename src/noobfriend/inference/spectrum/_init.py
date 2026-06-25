"""Data-driven initial estimates feeding the priors.

The priors are weakly informative but *centred on the data*. One linear
least-squares solve (:func:`numpy.linalg.lstsq`) fits the continuum and every
component's amplitude together -- at the initial line centres and template
widths -- so the per-component flux estimates *add up to the data* even when
lines overlap (a shallow broad under a narrow core, where a per-line peak guess
double-counts). Widths and velocities are not solved here -- that is the
sampler's job, bounded by the priors -- so a free width stays at its template
value; if the true width differs, a wide component can absorb a narrower one's
unfit wings and read a little high, which is harmless for a starting point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from noobfriend.inference.spectrum._window import C_KMS

if TYPE_CHECKING:
    from noobfriend.inference.spectrum._setup import ResolvedAxis, ResolvedComponent

_FWHM_TO_SIGMA: float = 1.0 / 2.3548200450309493
_SQRT_2PI: float = float(np.sqrt(2.0 * np.pi))


@dataclass(frozen=True)
class InitialEstimates:
    """Prior-centring estimates derived from the window.

    Attributes
    ----------
    c0, c1 : float
        Continuum level at ``lambda0`` and slope (``c1 = 0`` for degree 0).
    c0_scale, c1_scale : float
        Weakly-informative prior scales for ``c0`` / ``c1``.
    lambda0 : float
        Continuum reference wavelength (window mean).
    flux_scale : float
        A global positive flux scale (fallback for any component not solved).
    component_flux : dict
        Per-component initial integrated flux (positive), from the lstsq solve.
    """

    c0: float
    c1: float
    c0_scale: float
    c1_scale: float
    lambda0: float
    flux_scale: float
    component_flux: dict[str, float]


def _robust_std(values: np.ndarray) -> float:
    """Median-absolute-deviation standard deviation."""
    if values.size == 0:
        return 0.0
    mad = float(np.median(np.abs(values - np.median(values))))
    return 1.4826 * mad


def _offset_init_kms(axis: ResolvedAxis, centre_sys: float) -> float:
    """Return the initial velocity offset (km/s) for a centre axis (own part)."""
    to_kms = (
        (lambda v: v) if axis.unit == "km/s" else (lambda v: C_KMS * v / centre_sys)
    )
    if axis.kind == "fixed":
        return float(to_kms(axis.value))
    if axis.bounds is None:
        return 0.0
    return float(to_kms(0.5 * (axis.bounds[0] + axis.bounds[1])))


def component_geometry(
    components: tuple[ResolvedComponent, ...],
) -> dict[str, dict[str, float]]:
    """Per-component ``{sign, dv, fwhm, mu, sigma_w}`` at the initial guess.

    Centres use the initial velocity offset (0 unless the centre is fixed/bounded
    off systemic) applied to each component's ``centre_wavelength``; widths use
    the resolved FWHM init (template default, an absolute override, or the
    parent's when tied). Shared by the flux solve and the pre-fit preview.
    """
    from noobfriend.inference.spectrum._pymc_model import _topo_order

    out: dict[str, dict[str, float]] = {}
    for comp in _topo_order(components):
        centre_sys = comp.centre_wavelength
        base = comp.centre.base_id
        base_dv = out[base]["dv"] if base is not None else 0.0
        dv = base_dv + _offset_init_kms(comp.centre, centre_sys)
        mu = comp.centre_wavelength * (1.0 + dv / C_KMS)
        wax = comp.width
        if wax.kind == "tied":
            fwhm = out[wax.base_id]["fwhm"]
        elif wax.kind == "fixed":
            fwhm = float(wax.value)
        else:
            lo, hi = wax.bounds
            fwhm = float(np.clip(comp.template.fwhm_kms_init, lo, hi))
        out[comp.id] = {
            "sign": float(comp.template.sign),
            "dv": dv,
            "fwhm": fwhm,
            "mu": mu,
            "sigma_w": mu * (fwhm / C_KMS) * _FWHM_TO_SIGMA,
        }
    return out


def _solve_amplitudes(
    components: tuple[ResolvedComponent, ...],
    geometry: dict[str, dict[str, float]],
    wl: np.ndarray,
    flux: np.ndarray,
    lambda0: float,
    degree: int,
) -> tuple[float, float, dict[str, float], np.ndarray]:
    """Least-squares fit of continuum + amplitudes; return c0, c1, fluxes, model.

    Each free-flux component is a design column ``sign * profile``; a fixed flux
    is subtracted from the right-hand side; a fixed-ratio (tied) flux is folded
    into its root free ancestor's column and derived afterwards.
    """
    by_id = {c.id: c for c in components}
    prof = {
        c.id: np.exp(
            -0.5 * ((wl - geometry[c.id]["mu"]) / geometry[c.id]["sigma_w"]) ** 2
        )
        for c in components
    }
    cols: list[np.ndarray] = [np.ones_like(wl)]
    if degree >= 1:
        cols.append(wl - lambda0)
    rhs = np.asarray(flux, dtype=float).copy()
    col_index: dict[str, int] = {}

    def root_ratio(comp: ResolvedComponent) -> tuple[ResolvedComponent, float]:
        ratio, cur = 1.0, comp
        while cur.flux.base_id is not None and cur.flux.kind == "fixed":
            ratio *= cur.flux.value
            cur = by_id[cur.flux.base_id]
        return cur, ratio

    for c in components:
        fx = c.flux
        if fx.kind == "free":
            col_index[c.id] = len(cols)
            cols.append(geometry[c.id]["sign"] * prof[c.id])
        elif fx.base_id is None:  # fixed absolute -> known, subtract
            amp = fx.value / (geometry[c.id]["sigma_w"] * _SQRT_2PI)
            rhs = rhs - geometry[c.id]["sign"] * amp * prof[c.id]
    for c in components:
        fx = c.flux
        if fx.base_id is not None and fx.kind == "fixed":  # fixed-ratio -> fold
            root, ratio = root_ratio(c)
            factor = (
                geometry[c.id]["sign"]
                * ratio
                * (geometry[root.id]["sigma_w"] / geometry[c.id]["sigma_w"])
            )
            if root.id in col_index:
                cols[col_index[root.id]] = (
                    cols[col_index[root.id]] + factor * prof[c.id]
                )
            else:  # root flux also fixed-absolute
                amp_root = root.flux.value / (geometry[root.id]["sigma_w"] * _SQRT_2PI)
                rhs = rhs - factor * amp_root * prof[c.id]

    design = np.vstack(cols).T
    coef = np.linalg.lstsq(design, rhs, rcond=None)[0]
    c0 = float(coef[0])
    c1 = float(coef[1]) if degree >= 1 else 0.0
    model = design @ coef

    fluxes: dict[str, float] = {}
    for c in components:
        if c.id in col_index:
            amp = max(float(coef[col_index[c.id]]), 0.0)
            fluxes[c.id] = amp * geometry[c.id]["sigma_w"] * _SQRT_2PI
        elif c.flux.base_id is None and c.flux.kind == "fixed":
            fluxes[c.id] = float(c.flux.value)
    for c in components:
        if c.flux.base_id is not None and c.flux.kind == "fixed":
            root, ratio = root_ratio(c)
            fluxes[c.id] = ratio * fluxes[root.id]
    return c0, c1, fluxes, model


def initial_estimates(
    components: tuple[ResolvedComponent, ...],
    wl: np.ndarray,
    flux: np.ndarray,
    *,
    degree: int,
) -> InitialEstimates:
    """Estimate the continuum and per-component flux over the window.

    Parameters
    ----------
    components : tuple of ResolvedComponent
        The compiled components.
    wl, flux : numpy.ndarray
        The in-window wavelength and flux.
    degree : int
        Continuum polynomial degree (0 or 1).

    Returns
    -------
    InitialEstimates
    """
    wl = np.asarray(wl, dtype=float)
    flux = np.asarray(flux, dtype=float)
    lambda0 = float(np.mean(wl))
    geometry = component_geometry(components)
    c0, c1, fluxes, model = _solve_amplitudes(
        components, geometry, wl, flux, lambda0, degree
    )

    noise = _robust_std(flux - model)
    if noise <= 0:
        noise = max(_robust_std(flux - float(np.median(flux))), 1e-30)

    # Floor each flux at a faint (one-sigma) line, so the prior scale is positive
    # and a non-detection's prior still permits a weak line.
    component_flux = {
        c.id: max(fluxes.get(c.id, 0.0), noise * geometry[c.id]["sigma_w"] * _SQRT_2PI)
        for c in components
    }
    span = float(wl.max() - wl.min()) or 1.0
    return InitialEstimates(
        c0=c0,
        c1=c1,
        c0_scale=max(5.0 * noise, abs(c0) + 1e-30),
        c1_scale=max(5.0 * noise / span, 1e-30),
        lambda0=lambda0,
        flux_scale=max(max(component_flux.values(), default=0.0), 1e-30),
        component_flux=component_flux,
    )
