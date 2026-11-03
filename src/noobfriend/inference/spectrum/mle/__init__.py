"""Maximum-likelihood backend for quick single-frame fits.

``ws.mle()`` returns a :class:`NoobMLEResult`; its values are
:class:`ParameterEstimate` objects with Jacobian error bars.
"""

from noobfriend.inference.spectrum.mle._fit import fit
from noobfriend.inference.spectrum.mle._result import (
    CONDITION_LIMIT,
    ComponentEstimate,
    ContinuumEstimate,
    MLESolution,
    NoobMLEResult,
    ParameterEstimate,
)

__all__ = [
    "CONDITION_LIMIT",
    "ComponentEstimate",
    "ContinuumEstimate",
    "MLESolution",
    "NoobMLEResult",
    "ParameterEstimate",
    "fit",
]
