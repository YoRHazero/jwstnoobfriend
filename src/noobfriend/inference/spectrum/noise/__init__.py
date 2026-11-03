"""Frame noise models: declarations, likelihoods and leave-one-out terms.

Scripts declare :class:`Noise` with an optional :class:`AR1` or
:class:`Autocorrelation` correlation and attach it with
``NoobSpectrum.with_noise``. :class:`FrameNoise` is the precomputed per-frame
structure the fit backends evaluate. The estimators themselves live in
:mod:`noobfriend.core.specutils`; :func:`estimated_noise` only wraps their
output.
"""

from noobfriend.inference.spectrum.noise._frame import (
    FrameNoise,
    NoiseKind,
    ar1_couplings,
)
from noobfriend.inference.spectrum.noise._models import (
    AR1,
    Autocorrelation,
    Noise,
    estimated_noise,
)
from noobfriend.inference.spectrum.noise._numpyro import ar1_distribution

__all__ = [
    "AR1",
    "Autocorrelation",
    "FrameNoise",
    "Noise",
    "NoiseKind",
    "ar1_couplings",
    "ar1_distribution",
    "estimated_noise",
]
