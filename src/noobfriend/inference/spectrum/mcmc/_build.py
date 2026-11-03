"""Build the NumPyro model of a workspace.

The model honours two documented contracts: every physical parameter is a
named ``numpyro.deterministic`` site (flat workspace names), and every
likelihood term is an observed sample site ``frame.<id>.flux`` with a real
distribution. Everything else is internal and ends in ``.latent``:

- free parameters are sampled as ``value / scale`` (the table's natural
  scale) under the correspondingly scaled prior;
- joint-prior members are non-centred, ``u = mean + L z`` in unconstrained
  coordinates, and take part in no other reparameterization;
- a free line width under an LSF is sampled as the log of its effective
  (quadrature) width, with a prior-plus-Jacobian correction;
- an emission/absorption pair at one wavelength (same component and profile,
  both fluxes independent and on ``[0, inf)``) is sampled in polar
  net/cancellation coordinates;
- pooled continuum parameters are sampled centred, each frame's value
  ``~ Normal(mean, spread)`` (the power-law level in log space): every frame
  pins its own continuum with many pixels, where the non-centred form
  ``mean + spread * z`` turns into a funnel.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import inf, isclose, isfinite, log, pi, sqrt
from typing import TYPE_CHECKING, Any

import numpy as np

from noobfriend.core.jaxruntime import enable_x64
from noobfriend.inference.spectrum._xp import jax_namespace
from noobfriend.inference.spectrum.mcmc._names import (
    latent,
    observation_site,
    physical_names,
)
from noobfriend.inference.spectrum.priors import (
    LogUniform,
    Normal,
    Prior,
    TruncatedNormal,
    Uniform,
)

if TYPE_CHECKING:
    from noobfriend.inference.spectrum.workspace import NoobFitWorkspace, Parameter

_SQRT2 = sqrt(2.0)
_HALF_LOG_TWO_OVER_PI = 0.5 * log(2.0 / pi)


@dataclass(frozen=True, slots=True)
class Reparameterizations:
    """Which internal reparameterizations a built model uses.

    Attributes
    ----------
    flux_pairs
        ``(emission, absorption)`` line pairs sampled in net/cancellation
        coordinates.
    effective_fwhm
        Lines whose width is sampled as the effective width, with the
        representative LSF FWHM (km/s) used.
    joint_priors
        Parameter names of each non-centred joint prior.
    pooled
        Pooled continuum means.
    """

    flux_pairs: tuple[tuple[str, str], ...]
    effective_fwhm: Mapping[str, float]
    joint_priors: tuple[tuple[str, ...], ...]
    pooled: tuple[str, ...]

    def describe(self) -> tuple[str, ...]:
        """One line per reparameterization, for summaries and saved files."""
        lines = [
            f"flux pair {e} / {a}: net and cancellation" for e, a in self.flux_pairs
        ]
        lines += [
            f"{name}.fwhm: log effective width (LSF {lsf:.4g} km/s)"
            for name, lsf in self.effective_fwhm.items()
        ]
        lines += [
            f"joint prior: non-centred over {', '.join(names)}"
            for names in self.joint_priors
        ]
        lines += [f"{name}: pooled (centred) over frames" for name in self.pooled]
        return tuple(lines)


def build_numpyro_model(
    workspace: NoobFitWorkspace,
) -> tuple[Callable[[], None], Reparameterizations]:
    """Return the zero-argument NumPyro model and its reparameterizations.

    Parameters
    ----------
    workspace
        The resolved fit plan.

    Returns
    -------
    model : callable
        A plain NumPyro model taking no arguments.
    reparameterizations : Reparameterizations
        What the model samples internally.
    """
    import numpyro
    import numpyro.distributions as dist

    enable_x64()
    xp = jax_namespace()
    table = workspace.parameters
    joint_names = {
        name for block in workspace.joint_priors for name in block.parameters
    }
    pairs = _flux_pairs(workspace, joint_names)
    effective = _effective_fwhm(workspace, joint_names)
    names = physical_names(workspace)
    choleskies = [
        np.linalg.cholesky(block.covariance) for block in workspace.joint_priors
    ]
    frames = {}
    for frame_id, frame in workspace.spectrum.frames.items():
        mask = workspace.fitted[frame_id]
        frames[frame_id] = (np.asarray(frame.obs)[mask], np.asarray(frame.flux)[mask])

    def model() -> None:
        values: dict[str, Any] = {}
        for index, (block, cholesky) in enumerate(
            zip(workspace.joint_priors, choleskies, strict=True)
        ):
            size = len(block.parameters)
            z = numpyro.sample(
                latent(f"joint.{index}"),
                dist.Normal(0.0, 1.0).expand([size]).to_event(1),
            )
            u = block.mean + xp.matmul(cholesky, z)
            for position, (name, domain) in enumerate(
                zip(block.parameters, block.domains, strict=True)
            ):
                values[name] = domain.from_unconstrained(xp, u[position])
        for emission, absorption in pairs:
            _sample_pair(
                numpyro,
                dist,
                xp,
                table[f"{emission}.flux"],
                table[f"{absorption}.flux"],
                values,
            )
        for line_name, lsf in effective.items():
            row = table[f"{line_name}.fwhm"]
            values[row.name] = _sample_effective_fwhm(numpyro, dist, xp, row, lsf)
        for row in table.free():
            if row.name not in values:
                values[row.name] = _sample_prior(numpyro, dist, xp, row)
        for row in table.pooled():
            mean = values[row.pooled_from]
            spread = values[f"{row.pooled_from}.spread"]
            if row.log_pooled:
                log_value = numpyro.sample(
                    latent(row.name), dist.Normal(xp.log(mean), spread)
                )
                values[row.name] = xp.exp(log_value)
            else:
                raw = numpyro.sample(
                    latent(row.name),
                    dist.Normal(mean / row.scale, spread / row.scale),
                )
                values[row.name] = row.scale * raw
        physical = workspace.physical(values)
        for name in names:
            numpyro.deterministic(name, xp.asarray(physical[name]))
        for frame_id, (wavelength, flux) in frames.items():
            mean = workspace.frame_model(xp, physical, frame_id, wavelength).total
            noise = workspace.frame_noise[frame_id]
            distribution = noise.distribution(
                mean,
                physical[f"frame.{frame_id}.noise.scale"],
                physical.get(f"frame.{frame_id}.noise.rho"),
            )
            numpyro.sample(observation_site(frame_id), distribution, obs=flux)

    reparameterizations = Reparameterizations(
        flux_pairs=pairs,
        effective_fwhm=effective,
        joint_priors=tuple(block.parameters for block in workspace.joint_priors),
        pooled=tuple(dict.fromkeys(row.pooled_from for row in table.pooled())),
    )
    return model, reparameterizations


def _flux_pairs(
    workspace: NoobFitWorkspace, joint_names: set[str]
) -> tuple[tuple[str, str], ...]:
    """Emission/absorption pairs eligible for the polar reparameterization."""
    table = workspace.parameters

    def eligible(line: Any) -> bool:
        row = table.get(f"{line.name}.flux")
        if row is None or not row.is_free or row.name in joint_names:
            return False
        return row.prior.bounds() == (0.0, inf)  # type: ignore[union-attr]

    emissions = [line for line in workspace.lines if line.sign > 0 and eligible(line)]
    absorptions = [line for line in workspace.lines if line.sign < 0 and eligible(line)]
    used: set[str] = set()
    pairs = []
    for emission in emissions:
        for absorption in absorptions:
            if absorption.name in used:
                continue
            if (
                emission.component == absorption.component
                and emission.profile == absorption.profile
                and isclose(emission.nominal, absorption.nominal, rel_tol=1e-10)
            ):
                pairs.append((emission.name, absorption.name))
                used.add(absorption.name)
                break
    return tuple(pairs)


def _effective_fwhm(
    workspace: NoobFitWorkspace, joint_names: set[str]
) -> dict[str, float]:
    """Lines with a free width and an LSF: representative LSF per line."""
    effective = {}
    for line in workspace.lines:
        row = workspace.parameters.get(f"{line.name}.fwhm")
        widths = [width for width in line.lsf_fwhm_kms.values() if width is not None]
        if row is None or not row.is_free or row.name in joint_names or not widths:
            continue
        effective[line.name] = float(np.median(widths))
    return effective


def _support(dist: Any, low: float, high: float) -> Any:
    constraints = dist.constraints
    if isfinite(low) and isfinite(high):
        return constraints.interval(low, high)
    if isfinite(low):
        return constraints.greater_than(low)
    if isfinite(high):
        return constraints.less_than(high)
    return constraints.real


def _flat(numpyro: Any, dist: Any, name: str, low: float, high: float) -> Any:
    """Sample a flat site on ``(low, high)``, proper when both ends are finite."""
    if isfinite(low) and isfinite(high):
        return numpyro.sample(name, dist.Uniform(low, high))
    return numpyro.sample(name, dist.ImproperUniform(_support(dist, low, high), (), ()))


def _sample_prior(numpyro: Any, dist: Any, xp: Any, row: Parameter) -> Any:
    """Sample ``value / scale`` under the scaled prior and return the value."""
    scale = row.scale
    prior = row.prior.scaled(1.0 / scale)  # type: ignore[union-attr]
    low, high = prior.bounds()
    name = latent(row.name)
    if isinstance(prior, Normal):
        raw = numpyro.sample(name, dist.Normal(prior.mu, prior.sigma))
    elif isinstance(prior, TruncatedNormal):
        raw = numpyro.sample(
            name,
            dist.TruncatedNormal(
                prior.mu,
                prior.sigma,
                low=low if isfinite(low) else None,
                high=high if isfinite(high) else None,
            ),
        )
    elif isinstance(prior, Uniform):
        raw = _flat(numpyro, dist, name, low, high)
    elif isinstance(prior, LogUniform):
        log_low = log(low) if low > 0.0 else -inf
        raw = xp.exp(
            _flat(numpyro, dist, name, log_low, log(high) if isfinite(high) else inf)
        )
    else:
        raise TypeError(f"no sampling rule for prior {type(prior).__name__}.")
    return scale * raw


def _sample_effective_fwhm(
    numpyro: Any, dist: Any, xp: Any, row: Parameter, lsf: float
) -> Any:
    """Sample a width as ``log(effective / scale)`` and correct to its prior."""
    prior: Prior = row.prior  # type: ignore[assignment]
    low, high = prior.bounds()
    scale = row.scale
    t_low = log(sqrt(low**2 + lsf**2) / scale)
    t_high = log(sqrt(high**2 + lsf**2) / scale) if isfinite(high) else inf
    t = _flat(numpyro, dist, latent(row.name), t_low, t_high)
    effective = scale * xp.exp(t)
    fwhm = xp.sqrt(effective**2 - lsf**2)
    correction = prior.log_density(xp, fwhm) + 2.0 * xp.log(effective) - xp.log(fwhm)
    numpyro.factor(latent(f"{row.name}.reparam"), correction)
    return fwhm


def _sample_pair(
    numpyro: Any,
    dist: Any,
    xp: Any,
    emission: Parameter,
    absorption: Parameter,
    values: dict[str, Any],
) -> None:
    """Sample two fluxes in net/cancellation coordinates.

    With unit half-normal fluxes ``e, a``, the map ``n = (e - a) / sqrt 2``,
    ``c = sqrt(2 e a)`` gives ``n ~ Normal(0, 1/sqrt 2)``,
    ``c ~ Weibull(2, sqrt 2)`` and a ``-log r`` Jacobian (``r**2 = n**2 +
    c**2``). A flux prior other than ``TruncatedNormal(0, sigma, low=0)`` is
    restored exactly by a density-ratio factor.
    """
    prefix = f"pair.{emission.owner}.{absorption.owner}"
    net = numpyro.sample(latent(f"{prefix}.net"), dist.Normal(0.0, 1.0 / _SQRT2))
    cancellation = numpyro.sample(
        latent(f"{prefix}.cancellation"), dist.Weibull(_SQRT2, 2.0)
    )
    radius = xp.sqrt(net**2 + cancellation**2)
    numpyro.factor(latent(f"{prefix}.jacobian"), -xp.log(radius))
    units = {
        emission.name: (radius + net) / _SQRT2,
        absorption.name: (radius - net) / _SQRT2,
    }
    correction = 0.0
    for row in (emission, absorption):
        prior: Prior = row.prior  # type: ignore[assignment]
        unit = units[row.name]
        if isinstance(prior, TruncatedNormal) and prior.mu == 0.0:
            values[row.name] = prior.sigma * unit
            continue
        value = row.scale * unit
        half_normal = _HALF_LOG_TWO_OVER_PI - 0.5 * unit**2
        correction = (
            correction + prior.log_density(xp, value) + log(row.scale) - half_normal
        )
        values[row.name] = value
    if not isinstance(correction, float):
        numpyro.factor(latent(f"{prefix}.prior"), correction)
