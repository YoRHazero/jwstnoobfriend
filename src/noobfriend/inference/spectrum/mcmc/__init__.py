"""MCMC backend: NumPyro model, sampling, posterior results.

``ws.model()`` returns a :class:`NoobSpectrumModel` (with the plain NumPyro
model as an escape hatch); ``.sample()`` returns a :class:`NoobMCMCResult`
whose values are :class:`ParameterPosterior` objects, whose
``decompose()`` gives a :class:`NoobSpectrumDecomposition`, and whose
``as_prior`` turns components into joint-prior declarations for a new fit.
"""

from noobfriend.inference.spectrum.mcmc._as_prior import NonNormalPriorWarning
from noobfriend.inference.spectrum.mcmc._build import Reparameterizations
from noobfriend.inference.spectrum.mcmc._decompose import (
    Band,
    FrameDecomposition,
    NoobSpectrumDecomposition,
)
from noobfriend.inference.spectrum.mcmc._diagnostics import (
    BFMI_MIN,
    ESS_MIN,
    RHAT_MAX,
    ConvergenceVerdict,
    MCMCConvergenceWarning,
    ParameterDiagnostics,
    SamplingReport,
)
from noobfriend.inference.spectrum.mcmc._model import NoobSpectrumModel
from noobfriend.inference.spectrum.mcmc._persist import load
from noobfriend.inference.spectrum.mcmc._posterior import (
    ComponentPosterior,
    ContinuumPosterior,
    FramePosterior,
    FramePosteriors,
    KernelPosterior,
    NoisePosterior,
    ParameterPosterior,
)
from noobfriend.inference.spectrum.mcmc._result import NoobMCMCResult

__all__ = [
    "BFMI_MIN",
    "ESS_MIN",
    "RHAT_MAX",
    "Band",
    "ComponentPosterior",
    "ContinuumPosterior",
    "ConvergenceVerdict",
    "FrameDecomposition",
    "FramePosterior",
    "FramePosteriors",
    "KernelPosterior",
    "MCMCConvergenceWarning",
    "NoisePosterior",
    "NonNormalPriorWarning",
    "NoobMCMCResult",
    "NoobSpectrumDecomposition",
    "NoobSpectrumModel",
    "ParameterDiagnostics",
    "ParameterPosterior",
    "Reparameterizations",
    "SamplingReport",
    "load",
]
