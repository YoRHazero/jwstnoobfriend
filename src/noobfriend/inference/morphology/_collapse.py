"""Analytic marginalization of the linear amplitudes.

For one band, the model is ``y = X^T a + eps`` with ``eps ~ N(0, diag(sig^2))``
and Gaussian amplitude priors ``a ~ N(0, diag(v^2))``. Working with whitened
quantities (``y_w = y / sig``, rows of ``X_w`` likewise), both the marginal
likelihood ``p(y | theta) = N(0, X_w^T V X_w + I)`` and the conditional
amplitude posterior ``p(a | y, theta)`` are closed-form via a K x K Woodbury
solve (K = number of amplitudes, <= n_points + 2), costing O(N K^2) on top of
rendering.

This is Rao-Blackwellization / a collapsed sampler: MCMC explores only the
nonlinear parameters, and exact amplitude posterior draws are recovered
afterwards from the stored chain.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, cholesky, solve_triangular

from noobfriend.inference.morphology import _x64  # noqa: F401

__all__ = ["conditional_amplitudes", "marginal_loglik"]


def _cho_factor(x_w: Any, v: Any) -> tuple[Any, Any]:
    """Return ``(L, w)`` with ``L L^T = I + B B^T`` and ``w = B y_w``-ready ``B``.

    ``B = diag(v) X_w`` has shape ``(K, n)``.
    """
    b = x_w * v[:, None]
    m = jnp.eye(v.shape[0]) + b @ b.T
    return cholesky(m, lower=True), b


def marginal_loglik(y_w: Any, x_w: Any, v: Any, const: float) -> Any:
    """Return ``log p(y | theta)`` with amplitudes marginalized.

    Parameters
    ----------
    y_w : jax.Array
        Whitened flattened data, masked pixels zeroed, shape ``(n,)``.
    x_w : jax.Array
        Whitened design rows, shape ``(K, n)``.
    v : jax.Array
        Amplitude prior standard deviations, shape ``(K,)``.
    const : float
        The data-only constant ``-(n log 2pi + sum log sig^2) / 2`` over
        masked pixels.

    Returns
    -------
    jax.Array
        Scalar marginal log-likelihood.
    """
    chol, b = _cho_factor(x_w, v)
    w = b @ y_w
    solved = cho_solve((chol, True), w)
    # Residual form of y_w.y_w - w.M^-1.w: both terms are residual-scale,
    # avoiding the large-number cancellation of the direct expression and
    # keeping the marginal likelihood accurate in float32.
    resid = y_w - b.T @ solved
    quad = resid @ resid + solved @ solved
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diagonal(chol)))
    return const - 0.5 * (logdet + quad)


def conditional_amplitudes(y_w: Any, x_w: Any, v: Any) -> tuple[Any, Any]:
    """Return the exact Gaussian posterior of the amplitudes given ``theta``.

    Parameters
    ----------
    y_w, x_w, v : jax.Array
        As in :func:`marginal_loglik`.

    Returns
    -------
    tuple
        ``(mean, factor)`` where ``mean`` has shape ``(K,)`` and ``factor``
        satisfies ``factor @ factor.T = Cov[a | y, theta]``, so a posterior
        draw is ``mean + factor @ eps`` with ``eps ~ N(0, I_K)``.
    """
    chol, b = _cho_factor(x_w, v)
    w = b @ y_w
    mean = v * cho_solve((chol, True), w)
    inv_l = solve_triangular(chol, jnp.eye(v.shape[0]), lower=True)
    factor = v[:, None] * inv_l.T
    return mean, factor
