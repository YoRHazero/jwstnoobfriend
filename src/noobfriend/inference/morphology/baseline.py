"""Baseline scene selection from deterministic PSF-gate diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field

from noobfriend.inference.morphology.gate import PSFGateResult
from noobfriend.inference.morphology.scene import Scene


@dataclass(frozen=True)
class BaselineDecision:
    """Chosen component-only baseline for model comparison."""

    kind: str
    scene: Scene
    reason: str
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class WingSNRBaselineSelector:
    """Default modular baseline selector using positive wing SNR."""

    combined_threshold: float = 8.0
    single_band_threshold: float = 6.0

    def choose(
        self,
        gate: PSFGateResult,
        *,
        point_scene: Scene,
        sersic_scene: Scene,
    ) -> BaselineDecision:
        """Choose point-only or Sersic-only baseline scenes."""
        combined = gate.combined_positive_wing_snr
        max_band = gate.max_positive_wing_snr
        metrics = {"combined_positive_wing_snr": combined, "max_positive_wing_snr": max_band}
        if combined >= self.combined_threshold or max_band >= self.single_band_threshold:
            return BaselineDecision(
                kind="sersic",
                scene=sersic_scene,
                reason="positive PSF-wing residual exceeds extended-source threshold",
                metrics=metrics,
            )
        return BaselineDecision(
            kind="point",
            scene=point_scene,
            reason="PSF gate is point-like under positive wing-SNR thresholds",
            metrics=metrics,
        )
