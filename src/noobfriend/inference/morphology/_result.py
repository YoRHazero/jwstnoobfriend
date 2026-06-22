"""``MorphResult``: the posterior, its diagnostics, and derived quantities.

Wraps the numpyro ``MCMC`` run in an arviz :class:`~arviz.InferenceData` and adds
the morphology-specific surface: convergence diagnostics as first-class output
(R-hat, ESS, divergences -- never a point estimate from an unconverged posterior),
posterior-mean model / residual images, and derived quantities (per-band component
flux, half-light radius in kpc) obtained by evaluating a component's parameter
expression across the posterior draws. This module is internal; ``MorphResult`` is
reached from :meth:`MorphModel.sample`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from noobfriend.inference.morphology._geometry import kpc_per_arcsec

if TYPE_CHECKING:
    from noobfriend.inference.morphology._model import MorphModel, Preview


class MorphResult:
    """Posterior and derived quantities from a fitted :class:`MorphModel`.

    Attributes
    ----------
    model : MorphModel
        The model that was fit.
    idata : arviz.InferenceData
        The posterior and sample statistics.
    """

    def __init__(self, model: MorphModel, mcmc: Any) -> None:
        import arviz as az

        self.model = model
        self.mcmc = mcmc
        self.idata = az.from_numpyro(mcmc)
        self._comp = {cr.name: cr for cr in model._comp_render}

    # ------------------------------------------------------------ posterior

    def posterior_mean(self) -> dict[str, float]:
        """Posterior mean of every sampled site."""
        post = self.idata.posterior
        return {name: float(post[name].mean()) for name in post.data_vars}

    def summary(self) -> Any:
        """Return the arviz summary table (means, HDIs, R-hat, ESS)."""
        import arviz as az

        return az.summary(self.idata)

    def diagnostics(self) -> dict[str, float]:
        """Convergence diagnostics: worst R-hat, smallest ESS, divergence count."""
        summary = self.summary()
        diverging = (
            int(np.asarray(self.idata.sample_stats["diverging"]).sum())
            if "diverging" in self.idata.sample_stats
            else 0
        )
        return {
            "max_r_hat": float(summary["r_hat"].max()),
            "min_ess_bulk": float(summary["ess_bulk"].min()),
            "divergences": diverging,
        }

    # ------------------------------------------------------------ derived

    def _draws_env(self) -> dict[Any, np.ndarray]:
        """Map each free parameter to its flattened posterior draws."""
        post = self.idata.posterior
        env: dict[Any, np.ndarray] = {}
        for p in self.model._free:
            name = self.model._names[id(p)]
            env[p] = np.asarray(post[name].values).reshape(-1)
        return env

    def flux(self, component: str, band: str) -> np.ndarray:
        """Posterior draws of one component's flux in one band."""
        cr = self._component(component)
        if band not in cr.flux_by_band:
            raise ValueError(f"band {band!r} not in the model.")
        return np.broadcast_to(
            np.asarray(cr.flux_by_band[band].evaluate(self._draws_env())),
            self._n_draws(),
        )

    def r_eff_kpc(self, component: str) -> np.ndarray:
        """Posterior draws of one component's half-light radius, in proper kpc."""
        cr = self._component(component)
        if "r_eff" not in cr.geom:
            raise ValueError(f"component {component!r} has no r_eff.")
        scale = kpc_per_arcsec(self.model.image.z, self.model.cosmology)
        arcsec = np.broadcast_to(
            np.asarray(cr.geom["r_eff"].evaluate(self._draws_env())), self._n_draws()
        )
        return arcsec * scale

    def magnitude(self, component: str, band: str, zeropoint: float) -> np.ndarray:
        """Posterior draws of a component's AB-like magnitude given a zeropoint."""
        return -2.5 * np.log10(self.flux(component, band)) + zeropoint

    # ------------------------------------------------------------ images

    def model_image(self, band: str) -> np.ndarray:
        """Posterior-mean model image for a band."""
        return self._mean_preview().total[band]

    def residual(self, band: str) -> np.ndarray:
        """Posterior-mean residual (``data - model``) for a band."""
        return self._mean_preview().residual[band]

    def plot(self) -> Any:
        """Posterior-predictive montage: data | components | model | residual."""
        from noobfriend.inference.morphology._plot import montage

        return montage(self._mean_preview())

    # ------------------------------------------------------------ internals

    def _mean_preview(self) -> Preview:
        return self.model.preview(self.posterior_mean())

    def _component(self, name: str) -> Any:
        if name not in self._comp:
            raise ValueError(f"unknown component {name!r}; have {list(self._comp)}.")
        return self._comp[name]

    def _n_draws(self) -> tuple[int]:
        post = self.idata.posterior
        return (post.sizes["chain"] * post.sizes["draw"],)
