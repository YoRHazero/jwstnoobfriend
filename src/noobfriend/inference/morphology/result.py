"""Result and diagnostic containers for morphology workflows."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RenderedScene:
    """Rendered model images for one scene."""

    total: dict[str, np.ndarray]
    components: dict[str, dict[str, np.ndarray]]


@dataclass(frozen=True)
class FitDiagnostics:
    """Sampler health summary."""

    max_rhat: float | None = None
    min_ess: float | None = None
    divergences: int | None = None
    max_tree_depth_hits: int | None = None


@dataclass(frozen=True)
class PredictiveMetric:
    """Model-comparison metric with validity information."""

    name: str
    value: float | None
    valid: bool
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FitResult:
    """A completed morphology fit."""

    posterior: Any
    diagnostics: FitDiagnostics
    rendered: RenderedScene | None = None
    log_likelihood: np.ndarray | None = None
    derived: dict[str, np.ndarray] = field(default_factory=dict)
    metrics: dict[str, PredictiveMetric] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def metric(self, name: str) -> PredictiveMetric:
        """Return a named predictive metric."""
        return self.metrics[name]

    def with_metadata(self, metadata: dict[str, Any]) -> "FitResult":
        """Return a copy with updated workflow metadata."""
        return replace(self, metadata=metadata)
