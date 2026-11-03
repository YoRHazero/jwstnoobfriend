"""Spectral line fitting: declare, prepare, fit, read.

The chain::

    spec = NoobSpectrum(obs, flux, error, unit="angstrom", resolving_power=1600, z=z)
    ws = spec.prepare([NoobLine("Ha", rest=6564.6, unit="angstrom")],
                      continuum=NoobContinuum.polynomial(1, lambda_0=...))
    result = ws.model().sample()        # or ws.mle()

This top level exports every name a script needs to *declare* a fit. Objects
obtained along the chain (workspace, model, results, decompositions) are
imported from their subpackages when needed for typing.

Importing this package loads neither JAX, NumPyro, ArviZ nor matplotlib. Its
one side effect is writing ``NOOB_MCMC_CPU_CORES`` (default: every core) into
``XLA_FLAGS`` as the host device count, so MCMC chains can run in parallel.
"""

from noobfriend.core.jaxruntime import configure_host_devices

configure_host_devices()

from noobfriend.inference.spectrum.continuum import NoobContinuum  # noqa: E402
from noobfriend.inference.spectrum.data import NoobSpectrum  # noqa: E402
from noobfriend.inference.spectrum.line import NoobLine  # noqa: E402
from noobfriend.inference.spectrum.noise import AR1, Autocorrelation, Noise  # noqa: E402
from noobfriend.inference.spectrum.priors import (  # noqa: E402
    LogUniform,
    Normal,
    TruncatedNormal,
    Uniform,
)

__all__ = [
    "AR1",
    "Autocorrelation",
    "LogUniform",
    "Noise",
    "NoobContinuum",
    "NoobLine",
    "NoobSpectrum",
    "Normal",
    "TruncatedNormal",
    "Uniform",
]
