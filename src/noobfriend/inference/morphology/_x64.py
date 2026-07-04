"""Enable JAX float64 for all morphology internals.

Imported for its side effect by every internal module that touches JAX. The
collapsed log-posterior sums tens of thousands of squared residuals; in
float32 that sum carries absolute error comparable to the O(1) energy
differences NUTS accepts or rejects on, which destroys step-size adaptation.
"""

import jax

jax.config.update("jax_enable_x64", True)
