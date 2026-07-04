"""Bayesian spatial decomposition of AGN and host galaxies.

The public surface follows a collapsed-sampler design: MCMC explores only the
nonlinear morphology parameters (in a standardized, singularity-free latent
space), while component fluxes and backgrounds are marginalized analytically
and recovered exactly after sampling. Model presence ("is there a host?",
"is this a dual AGN?") is decided by predictive comparison across the model
family, never by boundary-hugging parameters inside one model.

Typical use::

    from noobfriend.inference.morphology import decompose

    result = decompose(images, models=("point", "point+host"))
    print(result.comparison.table)
    fit = result["point+host"]
    fit.flux("host", "f444w")        # exact amplitude posterior draws
    fit.health                       # rank-normalized R-hat / ESS gates

Sampling needs the optional MCMC stack (``uv add "jax[cpu]" numpyro arviz``);
everything else (data containers, priors, model specs) imports without it.
"""

from noobfriend.inference.morphology._map import MultistartConfig
from noobfriend.inference.morphology._nuts import NutsConfig
from noobfriend.inference.morphology.data import (
    KernelPSF,
    NoobImage,
    NoobImageSet,
    PSFProvider,
)
from noobfriend.inference.morphology.decompose import (
    MissingSamplerBackendError,
    SamplerConfig,
    decompose,
    fit_model,
)
from noobfriend.inference.morphology.models import ModelSpec
from noobfriend.inference.morphology.priors import (
    AmplitudePriors,
    CenterPrior,
    HostPriors,
    PointPriors,
    Priors,
)
from noobfriend.inference.morphology.results import (
    Comparison,
    DecompositionResult,
    ModelFit,
    PredictiveMetric,
    SamplerHealth,
)
from noobfriend.inference.morphology.validation import (
    RecoveryReport,
    RecoveryRow,
    recovery,
    simulate,
)

__all__ = [
    "AmplitudePriors",
    "CenterPrior",
    "Comparison",
    "DecompositionResult",
    "HostPriors",
    "KernelPSF",
    "MissingSamplerBackendError",
    "ModelFit",
    "ModelSpec",
    "MultistartConfig",
    "NoobImage",
    "NoobImageSet",
    "NutsConfig",
    "PSFProvider",
    "PointPriors",
    "PredictiveMetric",
    "Priors",
    "RecoveryReport",
    "RecoveryRow",
    "SamplerConfig",
    "SamplerHealth",
    "decompose",
    "fit_model",
    "recovery",
    "simulate",
]
