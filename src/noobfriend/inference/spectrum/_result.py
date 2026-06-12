"""``LineFitResult``: the posterior of a line fit and its conveniences.

Wraps the sampler's :class:`~xarray.DataTree` ``idata`` and the metadata needed
to interpret it: per-component summaries (:meth:`summary`, :meth:`interval`), a
posterior model curve for overlaying on a spectrum (:meth:`model_curve` /
:attr:`flux_func`, consumed by ``plot_spectrum1d``), convergence
:meth:`diagnostics`, and a one-line :meth:`plot`. Cross-model selection (single
vs double) is :func:`compare_models`.

ArviZ / xarray are imported lazily inside the methods, so holding a result never
requires the optional ``mcmc`` extra at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from matplotlib.figure import Figure

    from noobfriend.inference.spectrum._pymc_model import ComponentMeta
    from noobfriend.inference.spectrum._spectrum import NoobSpectrum


@dataclass(frozen=True)
class LineFitResult:
    """Posterior and conveniences for one line fit.

    Returned by :meth:`LineFitSetup.run`; not constructed directly.

    Attributes
    ----------
    idata : xarray.DataTree
        The sampler output (posterior, sample_stats, log_likelihood).
    components : tuple of ComponentMeta
        Component ids and signs, in build order.
    continuum_degree : int
        Continuum polynomial degree used.
    lambda0 : float
        Continuum reference wavelength.
    window_wl, window_flux, window_error : numpy.ndarray
        The fitted in-window data.
    spectrum : NoobSpectrum
        The source spectrum.
    """

    idata: Any
    components: tuple[ComponentMeta, ...]
    continuum_degree: int
    lambda0: float
    window_wl: np.ndarray
    window_flux: np.ndarray
    window_error: np.ndarray
    spectrum: NoobSpectrum

    @property
    def component_ids(self) -> tuple[str, ...]:
        """Component ids in build order."""
        return tuple(c.id for c in self.components)

    def _flat(self, name: str) -> np.ndarray:
        """Flatten a posterior variable's ``(chain, draw)`` samples to ``(S,)``."""
        return np.asarray(self.idata.posterior[name].values).reshape(-1)

    def interval(
        self, name: str, q: tuple[float, float, float] = (16.0, 50.0, 84.0)
    ) -> tuple[float, float, float]:
        """Return ``(low, median, high)`` percentiles of a posterior variable.

        Parameters
        ----------
        name : str
            A posterior variable name, e.g. ``"Halpha.broad__flux"``. The
            convenience accessors build these from a component id.
        q : tuple of float, default (16, 50, 84)
            The ``(low, mid, high)`` percentiles.

        Returns
        -------
        tuple of float
        """
        lo, mid, hi = np.percentile(self._flat(name), q)
        return float(lo), float(mid), float(hi)

    def flux(self, component_id: str) -> tuple[float, float, float]:
        """Integrated-flux ``(low, median, high)`` for a component."""
        return self.interval(f"{component_id}__flux")

    def fwhm_kms(self, component_id: str) -> tuple[float, float, float]:
        """Velocity-FWHM (km/s) ``(low, median, high)`` for a component."""
        return self.interval(f"{component_id}__fwhm_kms")

    def velocity_kms(self, component_id: str) -> tuple[float, float, float]:
        """Velocity offset (km/s) ``(low, median, high)`` for a component."""
        return self.interval(f"{component_id}__dv")

    def model_curve(
        self,
        grid: np.ndarray,
        *,
        q: tuple[float, float, float] = (16.0, 50.0, 84.0),
        max_draws: int = 400,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the posterior model curve on ``grid``.

        Parameters
        ----------
        grid : numpy.ndarray
            Wavelengths to evaluate at.
        q : tuple of float, default (16, 50, 84)
            Percentiles for the ``(low, mid, high)`` band.
        max_draws : int, default 400
            Cap on posterior draws used (subsampled) for speed.

        Returns
        -------
        low, mid, high : numpy.ndarray
            The percentile band of the total model flux at each grid point.
        """
        grid = np.asarray(grid, dtype=float)
        c0 = self._flat("continuum__c0")
        n = c0.size
        idx = (
            np.linspace(0, n - 1, max_draws).astype(int)
            if n > max_draws
            else np.arange(n)
        )
        c0 = c0[idx]
        curves = np.broadcast_to(c0[:, None], (idx.size, grid.size)).copy()
        if self.continuum_degree >= 1:
            c1 = self._flat("continuum__c1")[idx]
            curves += c1[:, None] * (grid[None, :] - self.lambda0)
        for comp in self.components:
            mu = self._flat(f"{comp.id}__mu")[idx]
            sw = self._flat(f"{comp.id}__sigma_w")[idx]
            amp = self._flat(f"{comp.id}__amp")[idx]
            curves += (
                comp.sign
                * amp[:, None]
                * np.exp(-0.5 * ((grid[None, :] - mu[:, None]) / sw[:, None]) ** 2)
            )
        low, mid, high = np.percentile(curves, q, axis=0)
        return low, mid, high

    @property
    def flux_func(
        self,
    ) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """A ``flux_func`` returning ``(low, mid, high)`` for ``plot_spectrum1d``."""

        def _f(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            low, mid, high = self.model_curve(grid)
            return low, mid, high

        return _f

    def summary(self, var_names: Sequence[str] | None = None) -> Any:
        """Return an ArviZ summary table (a :class:`pandas.DataFrame`).

        Parameters
        ----------
        var_names : sequence of str, optional
            Posterior variables to summarise. Defaults to each component's
            ``dv`` / ``fwhm_kms`` / ``flux``.

        Returns
        -------
        pandas.DataFrame
        """
        import arviz as az

        if var_names is None:
            var_names = [
                f"{c.id}__{q}"
                for c in self.components
                for q in ("dv", "fwhm_kms", "flux")
            ]
        return az.summary(self.idata, var_names=list(var_names))

    def diagnostics(self) -> dict[str, float]:
        """Return ``{divergences, max_rhat, min_ess_bulk}`` convergence flags."""
        import arviz as az

        stats = self.idata.sample_stats
        diverging = (
            int(np.asarray(stats["diverging"].values).sum())
            if "diverging" in stats.data_vars
            else 0
        )
        table = az.summary(self.idata)
        max_rhat = (
            float(np.nanmax(table["r_hat"].to_numpy()))
            if "r_hat" in table
            else float("nan")
        )
        min_ess = (
            float(np.nanmin(table["ess_bulk"].to_numpy()))
            if "ess_bulk" in table
            else float("nan")
        )
        return {"divergences": diverging, "max_rhat": max_rhat, "min_ess_bulk": min_ess}

    def plot(self, *, full: bool = False, **kwargs: Any) -> Figure:
        """Overlay the posterior model band on the data.

        Parameters
        ----------
        full : bool, default False
            Plot the whole spectrum; otherwise just the fitted window.
        **kwargs
            Forwarded to
            :func:`~noobfriend.core.display.plot._spectrum1d.plot_spectrum1d`.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from noobfriend.core.display.plot._spectrum1d import plot_spectrum1d

        if full:
            wl, fl, er = (
                self.spectrum.wavelength,
                self.spectrum.flux,
                self.spectrum.error,
            )
        else:
            wl, fl, er = self.window_wl, self.window_flux, self.window_error
        kwargs.setdefault("x_label", f"Wavelength [{self.spectrum.wave_unit}]")
        model = {"flux_func": self.flux_func, "label": "model", "color": "C3"}
        return plot_spectrum1d(wl, fl, error=er, models=[model], **kwargs)

    def __repr__(self) -> str:
        """Concise representation: component ids and draw count."""
        n = int(np.asarray(self.idata.posterior["continuum__c0"].values).size)
        return f"LineFitResult({len(self.components)} components, {n} draws)"


def compare_models(results: Mapping[str, LineFitResult]) -> Any:
    """Compare fits by expected log predictive density (PSIS-LOO).

    Parameters
    ----------
    results : mapping of str to LineFitResult
        Named fits (e.g. ``{"single": ..., "double": ...}``); each must carry a
        ``log_likelihood`` group (the default when sampled via
        :meth:`LineFitSetup.run`).

    Returns
    -------
    pandas.DataFrame
        ArviZ's ranked comparison table.
    """
    import arviz as az

    return az.compare({name: res.idata for name, res in results.items()})
