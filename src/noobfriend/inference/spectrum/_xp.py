"""Array-namespace switching for the spectrum package.

Line profiles, continua, priors and likelihoods are written once against an
:class:`ArrayNamespace` ``xp`` and evaluated with either NumPy (prepare, MLE,
plotting) or JAX (MCMC). ``xp`` behaves like the array module itself for array
functions (``xp.exp``, ``xp.where``, ...) and additionally carries the special
functions and linear algebra the formulas need, taken from SciPy or from
``jax.scipy`` so the two paths share call signatures.

Both namespaces are built lazily: importing this module loads neither SciPy's
submodules nor JAX.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["ArrayNamespace", "jax_namespace", "numpy_namespace", "readonly"]


@dataclass(frozen=True, slots=True)
class ArrayNamespace:
    """An array module bundled with matching special functions.

    Attribute access falls through to :attr:`module`, so ``xp.exp`` is
    ``numpy.exp`` or ``jax.numpy.exp``.

    Attributes
    ----------
    name
        ``"numpy"`` or ``"jax"``.
    module
        ``numpy`` or ``jax.numpy``.
    special
        ``scipy.special`` or ``jax.scipy.special`` (``erfcx``, ``erfc``,
        ``wofz``, ``log_ndtr``, ...).
    linalg
        ``scipy.linalg`` or ``jax.scipy.linalg`` (``cholesky``,
        ``solve_triangular``, ...).
    """

    name: str
    module: ModuleType
    special: ModuleType
    linalg: ModuleType

    def __getattr__(self, attribute: str) -> Any:
        """Delegate array functions to :attr:`module`."""
        return getattr(self.module, attribute)


@cache
def numpy_namespace() -> ArrayNamespace:
    """Return the NumPy/SciPy namespace."""
    import scipy.linalg
    import scipy.special

    return ArrayNamespace("numpy", np, scipy.special, scipy.linalg)


@cache
def jax_namespace() -> ArrayNamespace:
    """Return the JAX namespace (imports jax on first call)."""
    import jax.numpy as jnp
    import jax.scipy.linalg
    import jax.scipy.special

    return ArrayNamespace("jax", jnp, jax.scipy.special, jax.scipy.linalg)


def readonly(value: ArrayLike, *, dtype: Any = float) -> np.ndarray:
    """Return a read-only NumPy copy of ``value``.

    Parameters
    ----------
    value
        Any array-like input.
    dtype
        Target dtype, ``float`` by default.

    Returns
    -------
    numpy.ndarray
        A fresh array with ``writeable=False``.
    """
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array
