"""Composable morphology inference contracts.

The current package is a clean rewrite of the morphology API. New code should
use explicit image, scene, backend, and result objects rather than
environment-variable controlled workflows.
"""

from noobfriend.inference.morphology.backend import (
    MissingSamplerBackendError,
    NumpyroNUTSBackend,
    choose_jax_platform,
)
from noobfriend.inference.morphology.baseline import (
    BaselineDecision,
    WingSNRBaselineSelector,
)
from noobfriend.inference.morphology.components import (
    Background,
    EllipticalOffsetFrom,
    FixedCenter,
    FreeCenter,
    Point,
    Sersic,
    SersicShape,
)
from noobfriend.inference.morphology.comparison import (
    LOOComparison,
    compare_psis_loo,
    psis_loo,
)
from noobfriend.inference.morphology.data import KernelPSF, NoobImage, NoobImageSet
from noobfriend.inference.morphology.fit import (
    AnchorSlopeMultiStartConfig,
    AnchorSlopeMultiStartInitializer,
    FitConfig,
    MorphologyFitter,
    MultiStartPreviewResult,
    PreviewCandidate,
)
from noobfriend.inference.morphology.gate import (
    PSFGate,
    PSFGateConfig,
    PSFGateResult,
    RadialResidual,
)
from noobfriend.inference.morphology.parameters import (
    LogUniform,
    Normal,
    Parameter,
    PerBand,
    Prior,
    TotalFractionFlux,
    TruncNormal,
    Uniform,
)
from noobfriend.inference.morphology.render import inject_scene, render_scene
from noobfriend.inference.morphology.result import (
    FitDiagnostics,
    FitResult,
    PredictiveMetric,
    RenderedScene,
)
from noobfriend.inference.morphology.scene import Scene
from noobfriend.inference.morphology.introspection import scene_parameters
from noobfriend.inference.morphology.workflow import (
    MorphologyWorkflow,
    MorphologyWorkflowConfig,
    MorphologyWorkflowResult,
)

__all__ = [
    "AnchorSlopeMultiStartConfig",
    "AnchorSlopeMultiStartInitializer",
    "Background",
    "BaselineDecision",
    "EllipticalOffsetFrom",
    "FitConfig",
    "FitDiagnostics",
    "FitResult",
    "FixedCenter",
    "FreeCenter",
    "KernelPSF",
    "LOOComparison",
    "LogUniform",
    "MissingSamplerBackendError",
    "MorphologyFitter",
    "MorphologyWorkflow",
    "MorphologyWorkflowConfig",
    "MorphologyWorkflowResult",
    "MultiStartPreviewResult",
    "NoobImage",
    "NoobImageSet",
    "Normal",
    "NumpyroNUTSBackend",
    "PSFGateResult",
    "PSFGate",
    "PSFGateConfig",
    "Parameter",
    "PerBand",
    "Point",
    "PreviewCandidate",
    "PredictiveMetric",
    "Prior",
    "RadialResidual",
    "RenderedScene",
    "Scene",
    "Sersic",
    "SersicShape",
    "TotalFractionFlux",
    "TruncNormal",
    "Uniform",
    "WingSNRBaselineSelector",
    "choose_jax_platform",
    "compare_psis_loo",
    "inject_scene",
    "psis_loo",
    "render_scene",
    "scene_parameters",
]
