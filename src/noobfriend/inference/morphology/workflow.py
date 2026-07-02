"""End-to-end morphology decomposition workflow."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from noobfriend.inference.morphology.backend import NumpyroNUTSBackend
from noobfriend.inference.morphology.baseline import (
    BaselineDecision,
    WingSNRBaselineSelector,
)
from noobfriend.inference.morphology.comparison import (
    LOOComparison,
    compare_psis_loo,
)
from noobfriend.inference.morphology.data import NoobImageSet
from noobfriend.inference.morphology.fit import (
    AnchorSlopeMultiStartInitializer,
    MultiStartPreviewResult,
)
from noobfriend.inference.morphology.gate import PSFGate, PSFGateResult
from noobfriend.inference.morphology.introspection import scene_parameters
from noobfriend.inference.morphology.result import FitResult
from noobfriend.inference.morphology.scene import Scene


class SamplerBackend(Protocol):
    """Protocol consumed by the workflow for posterior sampling."""

    def fit(
        self,
        images: NoobImageSet,
        scene: Scene,
        *,
        fixed_params: dict[str, float] | None = None,
        init_params: dict[str, float] | None = None,
        seed: int = 0,
    ) -> FitResult:
        """Fit one scene and return a posterior result."""


@dataclass(frozen=True)
class MorphologyWorkflowConfig:
    """Configuration for the full morphology workflow."""

    gate: PSFGate = PSFGate()
    baseline_selector: WingSNRBaselineSelector = WingSNRBaselineSelector()
    initializer: AnchorSlopeMultiStartInitializer | None = (
        AnchorSlopeMultiStartInitializer()
    )
    sampler: SamplerBackend = NumpyroNUTSBackend()
    baseline_sampler: SamplerBackend | None = None
    run_baseline: bool = True
    run_full: bool = True
    compute_loo: bool = True


@dataclass(frozen=True)
class MorphologyWorkflowResult:
    """Result bundle from the end-to-end morphology workflow."""

    gate: PSFGateResult
    baseline: BaselineDecision
    initialization: MultiStartPreviewResult | None
    full_fit: FitResult | None
    baseline_fit: FitResult | None
    comparison: LOOComparison | None
    timings: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def fits(self) -> dict[str, FitResult]:
        """Return completed fits by comparison name."""
        results: dict[str, FitResult] = {}
        if self.baseline_fit is not None:
            results[f"baseline:{self.baseline.kind}"] = self.baseline_fit
        if self.full_fit is not None:
            results["full"] = self.full_fit
        return results

    @property
    def best_model(self) -> str | None:
        """Return the best valid LOO model name, when comparison is available."""
        if self.comparison is None:
            return None
        try:
            return self.comparison.best()
        except ValueError:
            return None

    @property
    def total_seconds(self) -> float:
        """Return total workflow runtime in seconds."""
        return self.timings.get("total_s", sum(self.timings.values()))


@dataclass(frozen=True)
class MorphologyWorkflow:
    """Run the gate, initialization, posterior fits, and model comparison."""

    config: MorphologyWorkflowConfig = MorphologyWorkflowConfig()

    def run(
        self,
        images: NoobImageSet,
        *,
        full_scene: Scene,
        point_scene: Scene,
        sersic_scene: Scene,
        fixed_params: dict[str, float] | None = None,
        init_params: dict[str, float] | None = None,
        baseline_init_params: dict[str, float] | None = None,
        seed: int = 0,
    ) -> MorphologyWorkflowResult:
        """Run the configured workflow and record stage timings."""
        images = images if isinstance(images, NoobImageSet) else NoobImageSet(images)
        timings: dict[str, float] = {}
        total_start = time.perf_counter()

        gate, elapsed = _timed(lambda: self.config.gate.fit(images))
        timings["psf_gate_s"] = elapsed

        baseline, elapsed = _timed(
            lambda: self.config.baseline_selector.choose(
                gate,
                point_scene=point_scene,
                sersic_scene=sersic_scene,
            )
        )
        timings["baseline_selection_s"] = elapsed

        initialization = None
        full_seed: dict[str, float] = {}
        if self.config.run_full and self.config.initializer is not None:
            initialization, elapsed = _timed(
                lambda: self.config.initializer.preview(images, full_scene, gate)
            )
            timings["initialization_s"] = elapsed
            for key, value in initialization.timings.items():
                timings[f"initialization.{key}"] = value
            if initialization.selected:
                full_seed.update(initialization.best.params)
        full_seed.update(init_params or {})

        full_fit = None
        if self.config.run_full:
            sampler = self.config.sampler
            scene_init = _filter_init_params(full_scene, full_seed)
            full_fit, elapsed = _timed(
                lambda: sampler.fit(
                    images,
                    full_scene,
                    fixed_params=fixed_params,
                    init_params=scene_init,
                    seed=seed,
                )
            )
            timings["full_fit_s"] = elapsed
            full_fit = _attach_workflow_metadata(
                full_fit,
                stage="full",
                elapsed=elapsed,
                initialization=initialization,
            )

        baseline_fit = None
        if self.config.run_baseline:
            sampler = self.config.baseline_sampler or self.config.sampler
            baseline_seed = _baseline_seed(gate, full_seed)
            baseline_seed.update(baseline_init_params or {})
            scene_init = _filter_init_params(baseline.scene, baseline_seed)
            baseline_fit, elapsed = _timed(
                lambda: sampler.fit(
                    images,
                    baseline.scene,
                    fixed_params=fixed_params,
                    init_params=scene_init,
                    seed=seed + 1,
                )
            )
            timings["baseline_fit_s"] = elapsed
            baseline_fit = _attach_workflow_metadata(
                baseline_fit,
                stage=f"baseline:{baseline.kind}",
                elapsed=elapsed,
                initialization=None,
            )

        comparison = None
        if self.config.compute_loo:
            fit_results = {}
            if baseline_fit is not None:
                fit_results[f"baseline:{baseline.kind}"] = baseline_fit
            if full_fit is not None:
                fit_results["full"] = full_fit
            if fit_results:
                comparison, elapsed = _timed(lambda: compare_psis_loo(fit_results))
                timings["loo_comparison_s"] = elapsed

        timings["total_s"] = time.perf_counter() - total_start
        return MorphologyWorkflowResult(
            gate=gate,
            baseline=baseline,
            initialization=initialization,
            full_fit=full_fit,
            baseline_fit=baseline_fit,
            comparison=comparison,
            timings=timings,
        )


def _baseline_seed(gate: PSFGateResult, full_seed: dict[str, float]) -> dict[str, float]:
    """Return a baseline seed from gate output and full-scene initialization."""
    seed = dict(full_seed)
    seed.update(gate.point_params)
    return seed


def _filter_init_params(scene: Scene, params: dict[str, float]) -> dict[str, float]:
    """Keep only initialization values matching sampled scene parameters."""
    names = {param.name for param in scene_parameters(scene)}
    if not names:
        return {}
    return {name: float(params[name]) for name in names if name in params}


def _attach_workflow_metadata(
    result: FitResult,
    *,
    stage: str,
    elapsed: float,
    initialization: MultiStartPreviewResult | None,
) -> FitResult:
    """Attach workflow provenance to one fit result."""
    metadata = dict(result.metadata)
    metadata["workflow_stage"] = stage
    metadata["workflow_elapsed_s"] = elapsed
    if initialization is not None:
        metadata["initialization"] = initialization
    return result.with_metadata(metadata)


def _timed(func: Any) -> tuple[Any, float]:
    """Run ``func`` and return ``(result, elapsed_seconds)``."""
    start = time.perf_counter()
    result = func()
    return result, time.perf_counter() - start
