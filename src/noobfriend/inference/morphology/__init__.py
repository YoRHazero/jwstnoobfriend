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

Sampling needs the optional MCMC stack (the ``mcmc`` extra, plus ``cuda`` for
a GPU); everything else (data containers, priors, model specs) imports
without it.

Fits run on a GPU when JAX has one (``SamplerConfig.device="auto"``) and then
compute in float32; on the CPU they compute in float64. Run one GPU fit at a
time: each GPU process reserves 75% of the GPU memory on its first GPU array,
so a second concurrent one may not even start.

Importing this package loads no JAX. Its one side effect is writing
``NOOB_MCMC_CPU_CORES`` (default: every core) into ``XLA_FLAGS`` as the host
device count, unless the user already set it, so CPU chains run in parallel.
"""

from noobfriend.core.jaxruntime import configure_host_devices

configure_host_devices()

from noobfriend.inference.morphology._map import MultistartConfig  # noqa: E402
from noobfriend.inference.morphology._nuts import NutsConfig  # noqa: E402
from noobfriend.inference.morphology.data import (  # noqa: E402
    KernelPSF,
    NoobImage,
    NoobImageSet,
    PSFProvider,
)
from noobfriend.inference.morphology.decompose import (  # noqa: E402
    MissingSamplerBackendError,
    SamplerConfig,
    decompose,
    fit_model,
)
from noobfriend.inference.morphology.models import ModelSpec  # noqa: E402
from noobfriend.inference.morphology.priors import (  # noqa: E402
    AmplitudePriors,
    CenterPrior,
    HostPriors,
    PointPriors,
    Priors,
)
from noobfriend.inference.morphology.results import (  # noqa: E402
    Comparison,
    DecompositionResult,
    FitRuntime,
    ModelFit,
    PredictiveMetric,
    SamplerHealth,
)
from noobfriend.inference.morphology.validation import (  # noqa: E402
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
    "FitRuntime",
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
