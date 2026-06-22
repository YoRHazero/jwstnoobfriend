"""MCMC spatial decomposition of high-redshift galaxies.

Forward-model a galaxy's multi-band imaging as a sum of analytic surface-brightness
profiles, each convolved with the per-band PSF, and sample the parameters with
numpyro/JAX NUTS.

Workflow: build a :class:`NoobImage` from one target's per-band cutouts (each a
plain :class:`BandSpec`-shaped dict); declare profile components (:class:`Sersic`,
:class:`Exponential`, :class:`PointSource`), giving each parameter nothing (a
data-driven default), a fixed number, an ``(lo, hi)`` interval, an explicit
:class:`Prior`, or a shared :class:`Param`, and tie components by reusing a
:class:`Param` or by an ordering expression such as ``r_eff + gap``; then
``model = image.model([...])``, inspect with ``model.preview()``, and
``result = model.sample()``.

``MorphModel`` and ``MorphResult`` are imported lazily (they pull in JAX/numpyro
from the optional ``mcmc`` extra), so importing this package needs only NumPy.
"""

from noobfriend.inference.morphology._band import BandSpec
from noobfriend.inference.morphology._image import NoobImage
from noobfriend.inference.morphology._param import (
    LogUniform,
    Normal,
    Param,
    Prior,
    Spec,
    TruncNormal,
    Uniform,
)
from noobfriend.inference.morphology._priors import HighzPriorPolicy, PriorPolicy
from noobfriend.inference.morphology._profile import (
    Exponential,
    PointSource,
    Profile,
    Sersic,
)

__all__ = [
    "BandSpec",
    "Exponential",
    "HighzPriorPolicy",
    "LogUniform",
    "MorphModel",
    "MorphResult",
    "NoobImage",
    "Normal",
    "Param",
    "PointSource",
    "Prior",
    "PriorPolicy",
    "Profile",
    "Sersic",
    "Spec",
    "TruncNormal",
    "Uniform",
]


def __getattr__(name: str) -> object:
    """Lazily expose the JAX/numpyro-backed model and result classes."""
    if name == "MorphModel":
        from noobfriend.inference.morphology._model import MorphModel

        return MorphModel
    if name == "MorphResult":
        from noobfriend.inference.morphology._result import MorphResult

        return MorphResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
