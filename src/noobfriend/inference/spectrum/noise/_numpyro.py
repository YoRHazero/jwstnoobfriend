"""NumPyro distribution for AR(1)-correlated pixels, built on first use.

The class is created inside :func:`ar1_distribution` so importing the noise
package never imports NumPyro or JAX.
"""

from __future__ import annotations

from functools import cache
from typing import Any

from noobfriend.inference.spectrum._xp import jax_namespace
from noobfriend.inference.spectrum.noise._frame import ar1_couplings


@cache
def ar1_distribution() -> type:
    """Return the ``AR1Normal`` NumPyro distribution class.

    ``AR1Normal(loc, scale, rho, gaps)`` is the Gaussian Markov chain
    ``x_0 ~ N(0, 1)``, ``x_i | x_{i-1} ~ N(phi_i x_{i-1}, 1 - phi_i**2)`` with
    ``phi_i = rho ** gaps[i - 1]`` and ``y = loc + scale * x``: an AR(1)
    process with unit marginal variance observed at integer pixel positions
    that may skip masked pixels. Its event shape is the pixel axis, so
    ``log_prob`` returns one value per frame.

    Returns
    -------
    type
        The distribution class.
    """
    import jax
    import jax.numpy as jnp
    from numpyro.distributions import Distribution, constraints

    xp = jax_namespace()

    class AR1Normal(Distribution):
        """Unit-variance AR(1) normal on possibly gapped pixels."""

        arg_constraints = {
            "loc": constraints.real_vector,
            "scale": constraints.positive,
            "rho": constraints.open_interval(-1.0, 1.0),
        }
        support = constraints.real_vector
        reparametrized_params = ["loc", "scale"]

        def __init__(
            self,
            loc: Any,
            scale: Any,
            rho: Any,
            gaps: Any,
            *,
            validate_args: bool | None = None,
        ) -> None:
            self.loc = jnp.asarray(loc)
            self.scale = jnp.broadcast_to(jnp.asarray(scale), self.loc.shape[-1:])
            self.rho = jnp.asarray(rho)
            self.gaps = jnp.asarray(gaps)
            super().__init__(
                batch_shape=(),
                event_shape=self.loc.shape[-1:],
                validate_args=validate_args,
            )

        def _phi(self) -> Any:
            return ar1_couplings(xp, self.rho, self.gaps)

        def sample(self, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
            noise = jax.random.normal(key, sample_shape + self.event_shape)
            phi = self._phi()
            innovation = jnp.sqrt(1.0 - phi**2)

            def step(previous: Any, inputs: tuple[Any, Any, Any]) -> tuple[Any, Any]:
                coupling, width, eps = inputs
                current = coupling * previous + width * eps
                return current, current

            first = noise[..., 0]
            rest = jnp.moveaxis(noise[..., 1:], -1, 0)
            _, tail = jax.lax.scan(step, first, (phi, innovation, rest))
            x = jnp.concatenate([first[..., None], jnp.moveaxis(tail, 0, -1)], axis=-1)
            return self.loc + self.scale * x

        def log_prob(self, value: Any) -> Any:
            x = (value - self.loc) / self.scale
            phi = self._phi()
            variance = 1.0 - phi**2
            innovations = x[..., 1:] - phi * x[..., :-1]
            quadratic = x[..., 0] ** 2 + jnp.sum(innovations**2 / variance, axis=-1)
            n = x.shape[-1]
            return (
                -0.5 * quadratic
                - 0.5 * n * jnp.log(2.0 * jnp.pi)
                - 0.5 * jnp.sum(jnp.log(variance), axis=-1)
                - jnp.sum(jnp.log(self.scale), axis=-1)
            )

    return AR1Normal
