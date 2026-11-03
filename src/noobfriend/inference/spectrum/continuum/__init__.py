"""Continuum declarations.

:class:`NoobContinuum` is built by ``polynomial`` or ``power_law``, anchored at
a required ``lambda_0``, and evaluated by :meth:`NoobContinuum.evaluate` with
NumPy or JAX.
"""

from noobfriend.inference.spectrum.continuum._continuum import (
    ContinuumKind,
    ContinuumSharing,
    NoobContinuum,
)

__all__ = ["ContinuumKind", "ContinuumSharing", "NoobContinuum"]
