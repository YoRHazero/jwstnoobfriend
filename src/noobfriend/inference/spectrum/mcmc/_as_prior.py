"""Turn part of a posterior into a joint prior for a new fit (R1).

The selected components' sampled parameters are moment-matched by one
multivariate normal in unconstrained coordinates (``log`` for positive
parameters, ``logit`` for fractions, identity otherwise), keeping their
cross-correlations. New declarations of the same structure are built by
calling the held declarations' own methods, so this module never imports the
declaration classes.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from noobfriend.inference.spectrum._xp import numpy_namespace
from noobfriend.inference.spectrum.priors import JointNormal, PriorSource

if TYPE_CHECKING:
    from noobfriend.inference.spectrum.mcmc._result import NoobMCMCResult
    from noobfriend.inference.spectrum.workspace import LineHandle, NoobFitWorkspace

#: Marginal skewness and excess kurtosis beyond which a warning is raised.
SKEW_LIMIT = 1.0
KURTOSIS_LIMIT = 2.0


class NonNormalPriorWarning(UserWarning):
    """A converted posterior is far from normal in unconstrained coordinates."""


def as_prior(result: NoobMCMCResult, names: Sequence[str]) -> dict[str, Any]:
    """Convert components of a fit into declarations sharing one joint prior.

    See :meth:`NoobMCMCResult.as_prior`.
    """
    workspace = result.workspace
    if isinstance(names, str):
        names = [names]
    selected = list(dict.fromkeys(names))
    lines = {line.name: line for line in workspace.lines}
    for name in selected:
        if name != "continuum" and name not in lines:
            raise KeyError(f"no component {name!r} in this fit.")
    for name in selected:
        base = lines[name].base if name in lines else None
        while base is not None:
            if base not in selected:
                raise ValueError(
                    f"{name!r} is derived from {base!r}; include {base!r} in as_prior too."
                )
            base = lines[base].base
    rows = _sampled_rows(workspace, selected)
    if not rows:
        raise ValueError("the selected components have no sampled parameter.")
    joint = _joint_prior(result, rows)
    declarations: dict[str, Any] = {}
    for name in _base_first(selected, lines):
        if name == "continuum":
            declarations[name] = _continuum(workspace, joint)
        else:
            declarations[name] = _line(lines[name], workspace, joint, declarations)
    return {name: declarations[name] for name in selected}


def _sampled_rows(workspace: NoobFitWorkspace, selected: list[str]) -> list[Any]:
    rows = []
    for row in workspace.parameters.values():
        if row.owner in selected and row.owner != "continuum" and row.is_free:
            rows.append(row)
        elif row.owner == "continuum" and "continuum" in selected and row.is_free:
            if not row.role.endswith(".spread"):
                rows.append(row)
        elif row.owner.startswith("frame.") and "continuum" in selected:
            if row.role.startswith("continuum.") and row.kind == "free":
                raise ValueError(
                    "as_prior cannot convert an 'independent' multi-frame continuum: "
                    "its coefficients are per frame."
                )
    return rows


def _joint_prior(result: NoobMCMCResult, rows: list[Any]) -> JointNormal:
    xp = numpy_namespace()
    posterior = result.idata["posterior"].dataset
    columns = []
    for row in rows:
        draws = np.asarray(posterior[row.name].values, dtype=float).reshape(-1)
        columns.append(row.domain.to_unconstrained(xp, draws))
    matrix = np.column_stack(columns)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("some draws sit on a domain edge; cannot transform them.")
    _check_normality(matrix, [row.name for row in rows])
    workspace = result.workspace
    sources = tuple(
        PriorSource(workspace.frame_digests[frame_id], workspace.fitted[frame_id])
        for frame_id in workspace.frame_ids
    )
    covariance = np.atleast_2d(np.cov(matrix, rowvar=False))
    return JointNormal(
        names=tuple(row.name for row in rows),
        domains=tuple(row.domain for row in rows),
        mean=matrix.mean(axis=0),
        covariance=covariance,
        sources=sources,
    )


def _check_normality(matrix: np.ndarray, names: list[str]) -> None:
    centered = matrix - matrix.mean(axis=0)
    std = centered.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    z = centered / std
    skew = np.mean(z**3, axis=0)
    kurtosis = np.mean(z**4, axis=0) - 3.0
    bad = [
        f"{name} (skew {s:.2f}, excess kurtosis {k:.2f})"
        for name, s, k in zip(names, skew, kurtosis, strict=True)
        if abs(s) > SKEW_LIMIT or abs(k) > KURTOSIS_LIMIT
    ]
    if bad:
        warnings.warn(
            "posterior is clearly non-normal in unconstrained coordinates for "
            + "; ".join(bad)
            + ". The joint normal prior will misrepresent it.",
            NonNormalPriorWarning,
            stacklevel=4,
        )


def _base_first(selected: list[str], lines: Mapping[str, LineHandle]) -> list[str]:
    ordered: list[str] = []

    def visit(name: str) -> None:
        if name in ordered:
            return
        if name in lines and lines[name].base is not None:
            visit(lines[name].base)  # type: ignore[arg-type]
        ordered.append(name)

    for name in selected:
        visit(name)
    return ordered


def _member(
    workspace: NoobFitWorkspace, joint: JointNormal, name: str, declared: Any
) -> Any:
    """Return the joint member for a free row, the original value for a fixed one."""
    row = workspace.parameters.get(name)
    if row is not None and row.is_free:
        return joint.member(name)
    return declared


def _line(
    handle: LineHandle,
    workspace: NoobFitWorkspace,
    joint: JointNormal,
    built: Mapping[str, Any],
) -> Any:
    declaration = handle.declaration
    name = handle.name
    wavelength = (
        {"rest": declaration.rest}
        if declaration.rest is not None
        else {"obs": declaration.obs}
    )
    shape = {
        "component": declaration.component,
        "contribution": declaration.contribution,
        "profile": declaration.profile,
    }
    if handle.base is None:
        line = type(declaration)(name, unit=declaration.unit, **wavelength, **shape)
    else:
        line = built[handle.base].derive(
            name, unit=declaration.unit, **wavelength, **shape
        )
    center = declaration.center_rule
    if center.kind in ("default", "delta_v_kms"):
        value = _member(workspace, joint, f"{name}.delta_v", center.value)
        line = line.center(delta_v_kms=value)
    elif center.kind == "delta_wavelength":
        value = _member(workspace, joint, f"{name}.delta_wavelength", center.value)
        line = line.center(delta_wavelength=value)
    else:
        value = _member(workspace, joint, f"{name}.delta_v_from_base", center.value)
        line = line.center(delta_v_from_base_kms=value)
    for role in ("fwhm", "flux"):
        rule = declaration.fwhm_rule if role == "fwhm" else declaration.flux_rule
        method = getattr(line, role)
        if rule.kind == "ratio_to_base":
            value = _member(
                workspace, joint, f"{name}.{role}_ratio_to_base", rule.value
            )
            line = method(ratio_to_base=value)
        else:
            line = method(_member(workspace, joint, f"{name}.{role}", rule.value))
    for kernel in declaration.kernels:
        if kernel.fwhm is None and kernel.fraction is None:
            continue
        line = line.convolve(
            kernel.kind,
            _member(workspace, joint, f"{name}.{kernel.kind}.fwhm", kernel.fwhm),
            fraction=_member(
                workspace, joint, f"{name}.{kernel.kind}.fraction", kernel.fraction
            ),
        )
    return line


def _continuum(workspace: NoobFitWorkspace, joint: JointNormal) -> Any:
    declaration = workspace.continuum.declaration
    sharing = declaration.sharing
    if declaration.kind == "power_law":
        continuum = type(declaration).power_law(
            lambda_0=declaration.lambda_0, sharing=sharing
        )
    else:
        continuum = type(declaration).polynomial(
            declaration.order, lambda_0=declaration.lambda_0, sharing=sharing
        )
    for parameter in declaration.parameter_names:
        declared = declaration.declarations.get(parameter)
        value = _member(workspace, joint, f"continuum.{parameter}", declared)
        if value is None:
            continue
        if parameter == "level":
            continuum = continuum.level(value)
        elif parameter == "index":
            continuum = continuum.index(value)
        else:
            continuum = continuum.coefficient(int(parameter[1:]), value)
    return continuum
