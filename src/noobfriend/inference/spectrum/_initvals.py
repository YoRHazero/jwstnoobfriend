"""Manual sampling start values (``initvals``), inverted from physical quantities.

A user (typically the frontend, hand-tuning a curve over a feature) can hand the
sampler a better NUTS starting point for a stubborn fit -- e.g. the broad/narrow
degeneracy where the data-driven start scatters chains into divergences. The
input is purely physical and per component: ``flux`` / ``fwhm_kms`` / ``dv_kms``
on a line, ``c0`` / ``c1`` on the continuum (the reserved ``"continuum"`` key).

The inversion to the model's free random variables is essentially the identity,
because :mod:`noobfriend.inference.spectrum._pymc_model` builds those RVs *as*
the physical quantities: ``{cid}__dv_off`` is the velocity offset in km/s,
``{cid}__fwhm_free`` is the velocity FWHM in km/s, ``{cid}__flux_raw`` /
``{cid}__flux_free`` is the integrated flux, and ``continuum__c0`` / ``__c1`` are
the continuum level / slope. The only work is (1) selecting the RV that exists
for each axis (a value targeting a tied / fixed / ratio axis cannot seed a start
and is skipped with a warning) and (2) clipping into each RV's *realised* support
-- the FWHM lower bound after the instrumental-resolution floor, an open interval
for a truncated normal -- so PyMC's transforms stay finite.

This is a sampling *start* only: it is fed to ``pm.sample(initvals=...)`` and
**never** changes a prior, so it cannot bias what the user is testing (e.g. the
reality of a broad line). This module is internal and pure (no PyMC).
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from noobfriend.inference.spectrum._setup import ResolvedComponent

from noobfriend.inference.spectrum._window import C_KMS

__all__ = ["normalize_init", "resolve_initvals"]

#: The reserved mapping key carrying continuum (not per-line) start values.
CONTINUUM_KEY: str = "continuum"

#: Accepted physical quantity keys for a line block and the continuum block.
_LINE_QUANTITIES: frozenset[str] = frozenset({"flux", "fwhm_kms", "dv_kms"})
_CONTINUUM_QUANTITIES: frozenset[str] = frozenset({"c0", "c1"})

#: Fractional inset used to keep a start strictly inside an open truncation.
_EDGE: float = 1e-6
#: A tiny positive floor for the (lower-unbounded) half-normal flux start.
_TINY: float = 1e-30


def normalize_init(
    init: Mapping[str, Mapping[str, float]], component_ids: list[str]
) -> dict[str, dict[str, float]]:
    """Validate a user ``init_guess`` mapping and return plain float dicts.

    Parameters
    ----------
    init : mapping
        ``{component_id: {quantity: value}}`` with an optional reserved
        ``"continuum"`` key (``{"c0": ..., "c1": ...}``). Line quantities are
        ``flux`` / ``fwhm_kms`` / ``dv_kms``.
    component_ids : list of str
        The compiled component ids; any key outside this set (other than
        ``"continuum"``) is a typo and raises.

    Returns
    -------
    dict
        The same structure with values coerced to ``float``.

    Raises
    ------
    ValueError
        On an unknown component id, an unknown quantity key, a non-mapping block,
        or a non-numeric (or boolean) value.
    """
    from collections.abc import Mapping as _Mapping

    ids = set(component_ids)
    out: dict[str, dict[str, float]] = {}
    for key, block in init.items():
        if key == CONTINUUM_KEY:
            allowed = _CONTINUUM_QUANTITIES
        elif key in ids:
            allowed = _LINE_QUANTITIES
        else:
            raise ValueError(
                f"init_guess: unknown component id {key!r}; known ids are "
                f"{sorted(ids)} (or {CONTINUUM_KEY!r} for the continuum)."
            )
        if not isinstance(block, _Mapping):
            raise ValueError(
                f"init_guess[{key!r}] must be a mapping of quantity -> value."
            )
        clean: dict[str, float] = {}
        for quantity, value in block.items():
            if quantity not in allowed:
                raise ValueError(
                    f"init_guess[{key!r}]: unknown quantity {quantity!r}; allowed: "
                    f"{sorted(allowed)}."
                )
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"init_guess[{key!r}][{quantity!r}] must be a number, got "
                    f"{value!r}."
                )
            clean[quantity] = float(value)
        out[key] = clean
    return out


def resolve_initvals(
    components: tuple[ResolvedComponent, ...],
    init: Mapping[str, Mapping[str, float]],
    *,
    fwhm_inst: float | None,
    continuum_degree: int,
) -> dict[str, float]:
    """Invert physical start values into free-RV start values for ``pm.sample``.

    Parameters
    ----------
    components : tuple of ResolvedComponent
        The compiled components (giving each axis's resolution and bounds).
    init : mapping
        A validated mapping from :func:`normalize_init`.
    fwhm_inst : float or None
        Instrumental velocity FWHM ``c / R`` (km/s), the floor applied to a free
        width's lower bound; ``None`` when the spectrum has no ``R``.
    continuum_degree : int
        The continuum polynomial degree actually used (``c1`` is only seedable at
        degree >= 1).

    Returns
    -------
    dict
        ``{rv_name: start_value}`` for the free RVs that could be seeded, each
        clipped into its realised support. Quantities targeting a non-free axis
        are skipped (with a warning).
    """
    initvals: dict[str, float] = {}
    by_id = {c.id: c for c in components}

    cont = init.get(CONTINUUM_KEY)
    if cont is not None:
        if "c0" in cont:
            initvals["continuum__c0"] = cont["c0"]
        if "c1" in cont:
            if continuum_degree >= 1:
                initvals["continuum__c1"] = cont["c1"]
            else:
                warnings.warn(
                    "init_guess['continuum']['c1'] skipped: the continuum is "
                    "degree 0 over this window.",
                    stacklevel=2,
                )

    for cid, block in init.items():
        if cid == CONTINUUM_KEY:
            continue
        comp = by_id[cid]
        if "dv_kms" in block:
            _set_dv(initvals, comp, block["dv_kms"])
        if "fwhm_kms" in block:
            _set_fwhm(initvals, comp, block["fwhm_kms"], fwhm_inst)
        if "flux" in block:
            _set_flux(initvals, comp, block["flux"])
    return initvals


def _clip_open(value: float, lo: float, hi: float) -> float:
    """Clip ``value`` strictly inside the open interval ``(lo, hi)``."""
    span = hi - lo
    pad = _EDGE * span if span > 0 else 0.0
    return float(min(max(value, lo + pad), hi - pad))


def _set_dv(initvals: dict[str, float], comp: ResolvedComponent, dv_kms: float) -> None:
    """Seed ``{cid}__dv_off`` (km/s) for a free centre axis."""
    axis = comp.centre
    if axis.kind != "free":
        warnings.warn(
            f"init_guess[{comp.id!r}]['dv_kms'] skipped: the centre is not a free "
            "axis (it is fixed or co-moving with a parent).",
            stacklevel=3,
        )
        return
    assert axis.bounds is not None  # noqa: S101  (a free axis is bounded)
    # The dv_off RV is always km/s; a wavelength-unit offset bound is converted
    # exactly as the model builds it (divide by the systemic centre).
    if axis.unit == "km/s":
        lo, hi = axis.bounds
    else:
        lo = C_KMS * axis.bounds[0] / comp.centre_wavelength
        hi = C_KMS * axis.bounds[1] / comp.centre_wavelength
    lo, hi = (lo, hi) if lo < hi else (hi, lo)
    initvals[f"{comp.id}__dv_off"] = _clip_open(dv_kms, lo, hi)


def _set_fwhm(
    initvals: dict[str, float],
    comp: ResolvedComponent,
    fwhm_kms: float,
    fwhm_inst: float | None,
) -> None:
    """Seed ``{cid}__fwhm_free`` (km/s) for a free width axis."""
    axis = comp.width
    if axis.kind != "free":
        warnings.warn(
            f"init_guess[{comp.id!r}]['fwhm_kms'] skipped: the width is not a free "
            "axis (it is tied to a parent or fixed).",
            stacklevel=3,
        )
        return
    assert axis.bounds is not None  # noqa: S101
    lo, hi = axis.bounds
    if fwhm_inst is not None:
        lo = max(lo, fwhm_inst)
    if not lo < hi:
        lo = 0.9 * hi
    initvals[f"{comp.id}__fwhm_free"] = _clip_open(fwhm_kms, lo, hi)


def _set_flux(initvals: dict[str, float], comp: ResolvedComponent, flux: float) -> None:
    """Seed the free *absolute* flux RV (half-normal or bounded) for a component."""
    axis = comp.flux
    if axis.kind == "free" and axis.base_id is None and axis.bounds is None:
        initvals[f"{comp.id}__flux_raw"] = max(flux, _TINY)  # half-normal, >0
    elif axis.kind == "free" and axis.base_id is None and axis.bounds is not None:
        initvals[f"{comp.id}__flux_free"] = _clip_open(
            flux, axis.bounds[0], axis.bounds[1]
        )
    else:
        warnings.warn(
            f"init_guess[{comp.id!r}]['flux'] skipped: the flux is tied, fixed, or "
            "a ratio to a parent, not a free absolute parameter.",
            stacklevel=3,
        )
