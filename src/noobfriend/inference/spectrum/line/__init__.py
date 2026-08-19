"""Line contract for spectrum fitting."""

from noobfriend.inference.spectrum.line.distributions import (
    LogUniform,
    Normal,
    Prior,
    TruncatedNormal,
    Uniform,
    prior_support,
)
from noobfriend.inference.spectrum.line.kernels import (
    BUILTIN_KERNELS,
    KernelFunction,
    KernelName,
    LineKernel,
    gaussian,
    laplace,
)
from noobfriend.inference.spectrum.line.line import NoobLine
from noobfriend.inference.spectrum.line.types import (
    DEFAULT_CENTER_RANGE_KMS,
    DEFAULT_FWHM_RANGES,
    NOMINAL_FWHM_KMS,
    ComponentName,
    ContributionName,
    FixedOrBounded,
    ParameterSpec,
    ProfileName,
)

__all__ = [
    "BUILTIN_KERNELS",
    "ComponentName",
    "ContributionName",
    "DEFAULT_CENTER_RANGE_KMS",
    "DEFAULT_FWHM_RANGES",
    "FixedOrBounded",
    "KernelFunction",
    "KernelName",
    "LineKernel",
    "LogUniform",
    "NOMINAL_FWHM_KMS",
    "NoobLine",
    "Normal",
    "ParameterSpec",
    "Prior",
    "ProfileName",
    "TruncatedNormal",
    "Uniform",
    "gaussian",
    "laplace",
    "prior_support",
]
