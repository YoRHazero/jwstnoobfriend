"""Compile a :class:`LineFitSetup` into a PyMC model.

:func:`build_model` walks the resolved component graph in parent-first order and
turns each component's three axes into PyTensor expressions: a sampled velocity
offset / FWHM / flux for *free* axes, a constant for *fixed* ones, and a
reference to the parent's expression for *tied* ones. The interpretable
quantities (``dv``, observed ``mu``, ``fwhm_kms``, wavelength ``sigma_w``,
integrated ``flux``, peak ``amp``) are registered as ``pm.Deterministic`` so the
result can summarise them and rebuild the model curve. Velocity widths are
sampled with a truncated-normal prior over the resolved bounds, floored by the
instrumental resolution; emission fluxes positive (half-normal or bounded).

PyMC and its dependencies are imported here lazily so importing
:mod:`noobfriend.inference.spectrum` never requires the optional ``mcmc`` extra.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from noobfriend.inference.spectrum._init import initial_estimates
from noobfriend.inference.spectrum._window import C_KMS, select_window

if TYPE_CHECKING:
    from noobfriend.inference.spectrum._setup import (
        LineFitSetup,
        ResolvedAxis,
        ResolvedComponent,
    )

_FWHM_TO_SIGMA: float = 1.0 / 2.3548200450309493
_SQRT_2PI: float = math.sqrt(2.0 * math.pi)


@dataclass(frozen=True)
class ComponentMeta:
    """Minimal per-component metadata the result needs to rebuild the curve."""

    id: str
    sign: int


@dataclass(frozen=True)
class ModelBundle:
    """A built PyMC model plus what the result needs after sampling.

    Attributes
    ----------
    model : pymc.Model
        The assembled model.
    components : tuple of ComponentMeta
        Component ids and signs in build order.
    continuum_degree : int
        Continuum polynomial degree actually used.
    lambda0 : float
        Continuum reference wavelength.
    window_wl, window_flux, window_error : numpy.ndarray
        The in-window data the model was conditioned on.
    """

    model: Any
    components: tuple[ComponentMeta, ...]
    continuum_degree: int
    lambda0: float
    window_wl: np.ndarray
    window_flux: np.ndarray
    window_error: np.ndarray


def _topo_order(
    components: tuple[ResolvedComponent, ...],
) -> list[ResolvedComponent]:
    """Order components parent-first (a derived line's centre names its parent)."""
    remaining = list(components)
    done: set[str] = set()
    ordered: list[ResolvedComponent] = []
    while remaining:
        progressed = False
        for comp in list(remaining):
            dep = comp.centre.base_id
            if dep is None or dep in done:
                ordered.append(comp)
                done.add(comp.id)
                remaining.remove(comp)
                progressed = True
        if not progressed:
            raise ValueError("could not order components parent-first (cycle?).")
    return ordered


def _truncated(
    pm: Any, name: str, lo: float, hi: float, mu: float | None = None
) -> Any:
    """Make a truncated-normal over ``[lo, hi]`` centred at ``mu`` (default mid)."""
    centre = 0.5 * (lo + hi) if mu is None else float(np.clip(mu, lo, hi))
    sigma = max((hi - lo) / 4.0, 1e-9)
    return pm.TruncatedNormal(name, mu=centre, sigma=sigma, lower=lo, upper=hi)


def _build_offset(
    pm: Any, pt: Any, axis: ResolvedAxis, centre_sys: float, cid: str
) -> Any:
    """Build the velocity offset (km/s) for one centre axis (own part only)."""
    to_kms = (
        (lambda v: v) if axis.unit == "km/s" else (lambda v: C_KMS * v / centre_sys)
    )
    if axis.kind == "fixed":
        return pt.as_tensor_variable(float(to_kms(axis.value)))
    assert axis.bounds is not None  # noqa: S101  (free axis)
    lo, hi = to_kms(axis.bounds[0]), to_kms(axis.bounds[1])
    lo, hi = (lo, hi) if lo < hi else (hi, lo)
    return _truncated(pm, f"{cid}__dv_off", lo, hi, mu=0.0)


def _build_fwhm(
    pm: Any,
    pt: Any,
    axis: ResolvedAxis,
    exprs: dict[str, dict[str, Any]],
    fwhm_inst: float | None,
    init: float,
    cid: str,
) -> Any:
    """Build the velocity-FWHM (km/s) for one width axis."""
    if axis.kind == "tied":
        assert axis.base_id is not None  # noqa: S101
        return exprs[axis.base_id]["fwhm"]
    if axis.kind == "fixed":
        return pt.as_tensor_variable(float(axis.value))
    assert axis.bounds is not None  # noqa: S101
    lo, hi = axis.bounds
    if fwhm_inst is not None:
        lo = max(lo, fwhm_inst)
    if not lo < hi:
        lo = 0.9 * hi
    return _truncated(pm, f"{cid}__fwhm_free", lo, hi, mu=init)


def _build_flux(
    pm: Any,
    pt: Any,
    axis: ResolvedAxis,
    exprs: dict[str, dict[str, Any]],
    flux_scale: float,
    cid: str,
) -> Any:
    """Build the integrated flux for one flux axis."""
    if axis.base_id is not None:  # ratio to parent
        parent_flux = exprs[axis.base_id]["flux"]
        if axis.kind == "fixed":
            return float(axis.value) * parent_flux
        assert axis.bounds is not None  # noqa: S101
        ratio = _truncated(pm, f"{cid}__flux_ratio", axis.bounds[0], axis.bounds[1])
        return ratio * parent_flux
    if axis.kind == "fixed":
        return pt.as_tensor_variable(float(axis.value))
    if axis.bounds is not None:
        return _truncated(pm, f"{cid}__flux_free", axis.bounds[0], axis.bounds[1])
    return pm.HalfNormal(f"{cid}__flux_raw", sigma=flux_scale)


def build_model(
    setup: LineFitSetup,
    *,
    window: tuple[float, float] | None = None,
    pad_kms: float = 5000.0,
    continuum_degree: int = 1,
    jitter: bool = False,
) -> ModelBundle:
    """Compile ``setup`` into a PyMC model over the fitting window.

    Parameters
    ----------
    setup : LineFitSetup
        The compiled model graph.
    window : tuple of float, optional
        Explicit ``(lo, hi)`` wavelength window; else auto from the line span.
    pad_kms : float, default 5000.0
        Velocity padding for the automatic window.
    continuum_degree : int, default 1
        Local continuum polynomial degree (forced to 0 for tiny windows).
    jitter : bool, default False
        Add a fractional error-inflation term to the likelihood.

    Returns
    -------
    ModelBundle
    """
    import pymc as pm
    import pytensor.tensor as pt

    spectrum = setup.spectrum
    comps = setup.components
    mask, _ = select_window(spectrum, comps, window=window, pad_kms=pad_kms)
    wl = spectrum.wavelength[mask].astype(float)
    fl = spectrum.flux[mask].astype(float)
    er = spectrum.error[mask].astype(float)
    degree = int(continuum_degree) if wl.size >= 4 else 0
    init = initial_estimates(spectrum, comps, wl, fl, er, degree=degree)
    z = spectrum.z
    fwhm_inst = None if spectrum.R is None else C_KMS / spectrum.R

    exprs: dict[str, dict[str, Any]] = {}
    metas: list[ComponentMeta] = []
    model = pm.Model()
    with model:
        lam = pt.as_tensor_variable(wl)
        c0 = pm.Normal("continuum__c0", mu=init.c0, sigma=init.c0_scale)
        if degree >= 1:
            c1 = pm.Normal("continuum__c1", mu=init.c1, sigma=init.c1_scale)
            model_flux = c0 + c1 * (lam - init.lambda0)
        else:
            model_flux = c0 + 0.0 * lam

        for comp in _topo_order(comps):
            cid = comp.id
            centre_sys = (1.0 + z) * comp.line.rest_wavelength
            base_dv = (
                exprs[comp.centre.base_id]["dv"]
                if comp.centre.base_id is not None
                else pt.as_tensor_variable(0.0)
            )
            dv = pm.Deterministic(
                f"{cid}__dv",
                base_dv + _build_offset(pm, pt, comp.centre, centre_sys, cid),
            )
            mu = pm.Deterministic(
                f"{cid}__mu", (1.0 + z) * comp.line.rest_wavelength * (1.0 + dv / C_KMS)
            )
            fwhm = pm.Deterministic(
                f"{cid}__fwhm_kms",
                _build_fwhm(
                    pm,
                    pt,
                    comp.width,
                    exprs,
                    fwhm_inst,
                    comp.template.fwhm_kms_init,
                    cid,
                ),
            )
            sigma_w = pm.Deterministic(
                f"{cid}__sigma_w", mu * (fwhm / C_KMS) * _FWHM_TO_SIGMA
            )
            flux = pm.Deterministic(
                f"{cid}__flux",
                _build_flux(pm, pt, comp.flux, exprs, init.flux_scale, cid),
            )
            amp = pm.Deterministic(f"{cid}__amp", flux / (sigma_w * _SQRT_2PI))
            model_flux = model_flux + comp.template.sign * amp * pt.exp(
                -0.5 * ((lam - mu) / sigma_w) ** 2
            )
            exprs[cid] = {"dv": dv, "fwhm": fwhm, "flux": flux}
            metas.append(ComponentMeta(cid, comp.template.sign))

        sigma_obs: Any = pt.as_tensor_variable(er)
        if jitter:
            log_f = pm.Normal("log_jitter", mu=-3.0, sigma=1.0)
            sigma_obs = pt.sqrt(er**2 + (pt.exp(log_f) * model_flux) ** 2)
        pm.Normal("obs", mu=model_flux, sigma=sigma_obs, observed=fl)

    return ModelBundle(
        model=model,
        components=tuple(metas),
        continuum_degree=degree,
        lambda0=init.lambda0,
        window_wl=wl,
        window_flux=fl,
        window_error=er,
    )
