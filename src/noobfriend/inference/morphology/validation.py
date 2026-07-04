"""Forward simulation and injection-recovery validation.

``simulate`` renders a model at known physical parameters onto the grids,
error maps and PSFs of an existing image set and adds Gaussian noise —
giving fake data whose truth is known exactly. ``recovery`` compares a
posterior fit against that truth, reporting posterior quantile ranks so that
calibration failures (not just biases) are visible.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from noobfriend.inference.morphology.data import NoobImageSet
from noobfriend.inference.morphology.models import ModelSpec
from noobfriend.inference.morphology.results import ModelFit

__all__ = ["RecoveryReport", "RecoveryRow", "recovery", "simulate"]


def simulate(
    images: NoobImageSet,
    model: ModelSpec | str,
    *,
    params: dict[str, float],
    fluxes: dict[str, dict[str, float]],
    backgrounds: dict[str, float] | None = None,
    noise: bool = True,
    oversample: int = 4,
    seed: int = 0,
) -> NoobImageSet:
    """Replace image data with a rendered model plus optional noise.

    Parameters
    ----------
    images : NoobImageSet
        Template images providing grids, error maps, masks and PSFs.
    model : ModelSpec or str
        Model to render.
    params : dict of str to float
        Physical parameters. Point sources need ``point{k}_x``/``point{k}_y``;
        a host needs ``host_r_eff``, ``host_n``, ``host_q``,
        ``host_theta_deg``, ``host_x`` and ``host_y``.
    fluxes : dict
        ``component -> band -> total flux`` for every source component
        (``point0``, ..., ``host``).
    backgrounds : dict of str to float, optional
        Per-band constant background (default 0).
    noise : bool
        Add ``error``-scaled Gaussian noise.
    oversample : int
        Sersic sub-pixel evaluation factor per axis.
    seed : int
        Noise seed.

    Returns
    -------
    NoobImageSet
        Copies of the images with simulated ``data``.
    """
    import jax.numpy as jnp

    from noobfriend.inference.morphology._render import prepare_static, source_images

    spec = ModelSpec.parse(model)
    images = images if isinstance(images, NoobImageSet) else NoobImageSet(images)
    backgrounds = backgrounds or {}
    phys: dict[str, Any] = {key: jnp.asarray(value) for key, value in params.items()}
    rng = np.random.default_rng(seed)

    simulated = []
    for image in images:
        static = prepare_static(image, oversample=oversample)
        sources = np.asarray(source_images(static, spec, phys))
        total = np.full(image.shape, float(backgrounds.get(image.name, 0.0)))
        for index, component in enumerate(spec.component_names):
            total = total + fluxes[component][image.name] * sources[index]
        if noise:
            total = total + image.error * rng.standard_normal(image.shape)
        simulated.append(replace(image, data=total))
    return NoobImageSet(simulated)


@dataclass(frozen=True)
class RecoveryRow:
    """Recovery diagnostics for one parameter.

    ``rank`` is the fraction of posterior draws below the truth; well
    calibrated posteriors give ranks that avoid 0 and 1. ``z_score`` is the
    truth's offset from the posterior median in posterior standard
    deviations.
    """

    name: str
    truth: float
    median: float
    q16: float
    q84: float
    rank: float
    z_score: float

    @property
    def covered_1sigma(self) -> bool:
        """Return whether the truth lies inside the central 68% interval."""
        return self.q16 <= self.truth <= self.q84


@dataclass(frozen=True)
class RecoveryReport:
    """Injection-recovery comparison for one fit."""

    rows: tuple[RecoveryRow, ...]

    @property
    def table(self) -> str:
        """Return a plain-text truth-vs-posterior table."""
        header = (
            f"{'parameter':<24}{'truth':>12}{'median':>12}{'q16':>12}"
            f"{'q84':>12}{'rank':>8}{'z':>8}"
        )
        lines = [header]
        for row in self.rows:
            lines.append(
                f"{row.name:<24}{row.truth:>12.4g}{row.median:>12.4g}"
                f"{row.q16:>12.4g}{row.q84:>12.4g}{row.rank:>8.3f}"
                f"{row.z_score:>8.2f}"
            )
        return "\n".join(lines)

    @property
    def max_abs_z(self) -> float:
        """Return the worst absolute posterior z-score."""
        return max(abs(row.z_score) for row in self.rows)


def recovery(fit: ModelFit, truth: dict[str, float]) -> RecoveryReport:
    """Compare a fit against known injected parameters.

    Parameters
    ----------
    fit : ModelFit
        The posterior fit.
    truth : dict of str to float
        True values keyed by physical parameter name (e.g. ``host_r_eff``)
        or amplitude name ``{component}_flux_{band}`` (e.g.
        ``host_flux_f444w``, ``background_flux_f444w``).

    Returns
    -------
    RecoveryReport
        Per-parameter quantile ranks and z-scores.
    """
    rows = []
    for name, value in truth.items():
        draws = _find_draws(fit, name)
        if draws is None:
            continue
        flat = np.asarray(draws).ravel()
        std = float(flat.std())
        rows.append(
            RecoveryRow(
                name=name,
                truth=float(value),
                median=float(np.median(flat)),
                q16=float(np.quantile(flat, 0.16)),
                q84=float(np.quantile(flat, 0.84)),
                rank=float(np.mean(flat < value)),
                z_score=float((np.median(flat) - value) / std) if std > 0 else np.inf,
            )
        )
    return RecoveryReport(rows=tuple(rows))


def _find_draws(fit: ModelFit, name: str) -> np.ndarray | None:
    """Locate posterior draws for a truth key, or return ``None``."""
    if name in fit.params:
        return fit.params[name]
    for component, bands in fit.fluxes.items():
        for band, draws in bands.items():
            if name == f"{component}_flux_{band}":
                return draws
    return None
