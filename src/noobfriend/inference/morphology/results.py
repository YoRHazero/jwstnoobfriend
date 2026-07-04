"""Result containers for morphology decomposition fits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from noobfriend.inference.morphology.data import NoobImageSet
from noobfriend.inference.morphology.models import ModelSpec

if TYPE_CHECKING:
    import arviz

__all__ = [
    "Comparison",
    "DecompositionResult",
    "ModelFit",
    "PredictiveMetric",
    "SamplerHealth",
]


@dataclass(frozen=True)
class SamplerHealth:
    """Convergence diagnostics for one fit.

    ``max_rhat`` and the ESS values are rank-normalized (Vehtari et al.
    2021); ``messages`` lists every threshold violation, so an empty tuple
    means the run looks healthy.
    """

    max_rhat: float
    min_ess_bulk: float
    min_ess_tail: float
    divergences: int
    messages: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Return whether no diagnostic threshold fired."""
        return not self.messages


@dataclass(frozen=True)
class PredictiveMetric:
    """One predictive model-comparison metric (PSIS-LOO)."""

    name: str
    value: float | None
    valid: bool
    reason: str | None = None
    details: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Comparison:
    """PSIS-LOO comparison across fitted models."""

    metrics: dict[str, PredictiveMetric]

    def best(self) -> str:
        """Return the valid model with the largest ELPD-LOO.

        Raises
        ------
        ValueError
            If no model has a valid metric.
        """
        valid = {
            name: metric.value
            for name, metric in self.metrics.items()
            if metric.valid and metric.value is not None
        }
        if not valid:
            raise ValueError("no valid PSIS-LOO metric is available.")
        return max(valid, key=lambda name: valid[name])

    @property
    def table(self) -> str:
        """Return a plain-text ELPD table, best model first."""
        rows = sorted(
            self.metrics.items(),
            key=lambda item: -(item[1].value if item[1].value is not None else -np.inf),
        )
        lines = [f"{'model':<16}{'elpd_loo':>12}{'se':>10}  status"]
        for name, metric in rows:
            value = f"{metric.value:.1f}" if metric.value is not None else "-"
            se = metric.details.get("se")
            se_text = f"{se:.1f}" if se is not None else "-"
            status = "ok" if metric.valid else f"invalid: {metric.reason}"
            lines.append(f"{name:<16}{value:>12}{se_text:>10}  {status}")
        return "\n".join(lines)


@dataclass(frozen=True)
class ModelFit:
    """Posterior fit of one model.

    Attributes
    ----------
    spec : ModelSpec
        The fitted model.
    params : dict of str to numpy.ndarray
        Physical parameter draws, each of shape ``(chains, draws)``.
    fluxes : dict
        ``component -> band -> draws`` amplitude posteriors, components being
        ``point0, ..., host, background``.
    z : numpy.ndarray
        Latent draws, shape ``(chains, draws, dim)``.
    health : SamplerHealth
        Convergence diagnostics.
    loo : PredictiveMetric or None
        PSIS-LOO metric when computed.
    rendered : dict
        ``band -> {component: image, "total": image}`` at the posterior
        median.
    map_modes : tuple
        Deduplicated ``(z, log_density)`` MAP modes found by the initializer.
    chain_weights : numpy.ndarray or None
        Per-chain stacking weights (Yao, Vehtari & Gelman 2022) computed
        from per-chain PSIS-LOO when the pointwise log-likelihood is
        available; ``None`` otherwise. Near-uniform weights mean the chains
        agree; strongly uneven weights mean some chains sampled
        configurations that predict the data poorly.
    timings : dict of str to float
        Per-stage wall times in seconds.
    images : NoobImageSet
        The fitted images (kept for residual accessors).
    """

    spec: ModelSpec
    params: dict[str, np.ndarray]
    fluxes: dict[str, dict[str, np.ndarray]]
    z: np.ndarray
    health: SamplerHealth
    loo: PredictiveMetric | None
    rendered: dict[str, dict[str, np.ndarray]]
    map_modes: tuple[tuple[np.ndarray, float], ...]
    timings: dict[str, float]
    images: NoobImageSet
    chain_weights: np.ndarray | None = None

    @property
    def name(self) -> str:
        """Return the canonical model name."""
        return self.spec.name

    def median_params(self) -> dict[str, float]:
        """Return posterior medians of all physical parameters."""
        return {name: float(np.median(draws)) for name, draws in self.params.items()}

    def flux(self, component: str, band: str) -> np.ndarray:
        """Return amplitude posterior draws for one component and band."""
        return self.fluxes[component][band]

    def residual_sigma(self, band: str) -> np.ndarray:
        """Return the whitened residual map at the posterior median model."""
        image = self.images[band]
        resid = (image.data - self.rendered[band]["total"]) / image.error
        return np.where(image.mask, resid, np.nan)

    def stacked(self, *, seed: int = 0) -> "ModelFit":
        """Return a copy whose draws are stacking-resampled across chains.

        Chains are resampled with probability proportional to
        ``chain_weights`` and draws uniformly within each chain, giving a
        single merged pseudo-chain of the same total size. Use when the raw
        fit's R-hat flags chain disagreement: the mixture weighs posterior
        configurations by predictive quality instead of by the number of
        chains that landed in each.

        Notes
        -----
        ``health`` is carried over unchanged as the record of the raw-run
        diagnosis (R-hat/ESS do not apply to the resampled mixture), and
        ``rendered`` still shows the raw-run posterior-median model.

        Raises
        ------
        ValueError
            If stacking weights are unavailable (fit run without LOO).
        """
        if self.chain_weights is None:
            raise ValueError("stacking weights unavailable; fit with compute_loo=True.")
        from dataclasses import replace

        rng = np.random.default_rng(seed)
        chains, draws = self.z.shape[:2]
        total = chains * draws
        chain_idx = rng.choice(chains, size=total, p=self.chain_weights)
        draw_idx = rng.integers(0, draws, size=total)

        def resample(array: np.ndarray) -> np.ndarray:
            flat = array[chain_idx, draw_idx]
            return flat.reshape(1, total, *array.shape[2:])

        return replace(
            self,
            params={name: resample(v) for name, v in self.params.items()},
            fluxes={
                comp: {band: resample(v) for band, v in bands.items()}
                for comp, bands in self.fluxes.items()
            },
            z=resample(self.z),
        )

    def to_arviz(self) -> "arviz.InferenceData":
        """Return an ArviZ ``InferenceData`` with all named posteriors."""
        import arviz as az

        posterior: dict[str, Any] = dict(self.params)
        for component, bands in self.fluxes.items():
            for band, draws in bands.items():
                posterior[f"{component}_flux_{band}"] = draws
        return az.from_dict(posterior=posterior)


@dataclass(frozen=True)
class DecompositionResult:
    """Bundle of fits and their predictive comparison."""

    fits: dict[str, ModelFit]
    comparison: Comparison | None
    timings: dict[str, float]

    def __getitem__(self, name: str) -> ModelFit:
        """Return one fit by canonical model name."""
        return self.fits[name]

    @property
    def best(self) -> str | None:
        """Return the best valid model name, when comparison is available."""
        if self.comparison is None:
            return None
        try:
            return self.comparison.best()
        except ValueError:
            return None
