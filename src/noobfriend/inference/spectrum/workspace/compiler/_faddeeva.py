"""Differentiable Voigt profile for the PyTensor backend.

A lorentzian line seen through a gaussian instrumental line-spread function is
a Voigt profile ``V(x; sigma, gamma) = Re[w(z)] / (sigma sqrt(2 pi))`` with
``z = (x + i gamma) / (sigma sqrt(2))``, where ``w`` is the Faddeeva function.
The numeric path evaluates it with :func:`scipy.special.voigt_profile`, but
PyTensor ships no Faddeeva op, so sampling needs one: this module wraps
:func:`scipy.special.wofz` in a custom Op whose gradients follow from the
closed form ``w'(z) = -2 z w(z) + 2i / sqrt(pi)`` — the derivative of ``w`` is
expressed through ``w`` itself, so the backward pass reuses the forward
outputs and stays exact.

PyTensor is an optional dependency (the ``mcmc`` extra), so it is imported
lazily inside the Op factory; importing this module stays cheap and safe.
"""

from __future__ import annotations

from functools import cache
from math import pi, sqrt
from typing import Any

import numpy as np
from scipy.special import wofz

_SQRT_TWO = sqrt(2.0)
_SQRT_TWO_PI = sqrt(2.0 * pi)
_TWO_OVER_SQRT_PI = 2.0 / sqrt(pi)


@cache
def _wofz_op() -> Any:
    """Build (once) the PyTensor Op for the Faddeeva function.

    Returns
    -------
    Any
        An Op mapping real tensors ``(u, v)`` of one shared shape to
        ``(Re[w(u + iv)], Im[w(u + iv)])``.
    """
    import pytensor.tensor as pt
    from pytensor.gradient import DisconnectedType
    from pytensor.graph.basic import Apply
    from pytensor.graph.op import Op

    class Wofz(Op):
        """Real and imaginary parts of the Faddeeva function ``w(u + iv)``."""

        __props__ = ()

        def make_node(self, u: Any, v: Any) -> Any:
            u = pt.as_tensor_variable(u)
            v = pt.as_tensor_variable(v)
            if u.type.broadcastable != v.type.broadcastable:
                raise ValueError(
                    "wofz expects u and v pre-broadcast to one shared shape."
                )
            return Apply(self, [u, v], [u.type(), u.type()])

        def perform(self, node: Any, inputs: Any, output_storage: Any) -> None:
            u, v = inputs
            w = wofz(u + 1j * v)
            output_storage[0][0] = np.ascontiguousarray(w.real)
            output_storage[1][0] = np.ascontiguousarray(w.imag)

        def infer_shape(self, fgraph: Any, node: Any, input_shapes: Any) -> Any:
            return [input_shapes[0], input_shapes[0]]

        def pullback(self, inputs: Any, outputs: Any, output_grads: Any) -> Any:
            u, v = inputs
            real, imag = outputs
            # w'(z) = -2 z w(z) + 2i / sqrt(pi), split into real/imaginary
            # parts; the Cauchy-Riemann equations give the v-derivatives.
            real_u = -2.0 * (u * real - v * imag)
            imag_u = _TWO_OVER_SQRT_PI - 2.0 * (u * imag + v * real)
            grad_u = pt.zeros_like(u)
            grad_v = pt.zeros_like(v)
            grad_real, grad_imag = output_grads
            if not isinstance(grad_real.type, DisconnectedType):
                grad_u = grad_u + grad_real * real_u
                grad_v = grad_v - grad_real * imag_u
            if not isinstance(grad_imag.type, DisconnectedType):
                grad_u = grad_u + grad_imag * imag_u
                grad_v = grad_v + grad_imag * real_u
            return [grad_u, grad_v]

    return Wofz()


def symbolic_voigt(pt: Any, delta: Any, sigma: Any, gamma: Any) -> Any:
    """Build the PyTensor graph of a unit-integral Voigt profile.

    The symbolic counterpart of :func:`scipy.special.voigt_profile`, exact and
    differentiable in all three arguments.

    Parameters
    ----------
    pt
        The ``pytensor.tensor`` namespace.
    delta
        Offsets from the line centre.
    sigma
        Standard deviation of the gaussian component, in the units of
        ``delta``.
    gamma
        Half width at half maximum of the lorentzian component, in the units
        of ``delta``.

    Returns
    -------
    Any
        A PyTensor expression for the Voigt profile values, integrating to one
        over ``delta``.
    """
    u = delta / (sigma * _SQRT_TWO)
    v = gamma / (sigma * _SQRT_TWO)
    u, v = pt.broadcast_arrays(u, v)
    real, _ = _wofz_op()(u, v)
    return real / (sigma * _SQRT_TWO_PI)
