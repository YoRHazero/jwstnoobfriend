"""``LineFitResult``: the posterior of a line fit and its conveniences.

Wraps the sampler's :class:`~xarray.DataTree` ``idata`` and the metadata needed
to interpret it: per-component summaries (:meth:`summary`, :meth:`interval`), a
posterior model curve for overlaying on a spectrum (:meth:`model_curve` /
:attr:`flux_func`, consumed by ``plot_spectrum1d``), convergence
:meth:`diagnostics`, and a one-line :meth:`plot`. Cross-model selection (single
vs double) is :meth:`LineFitResult.compare`.

ArviZ / xarray are imported lazily inside the methods, so holding a result never
requires the optional ``mcmc`` extra at import time.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from matplotlib.figure import Figure

    from noobfriend.inference.spectrum._pymc_model import BoundedParam, ComponentMeta
    from noobfriend.inference.spectrum._spectrum import NoobSpectrum


@dataclass(frozen=True)
class LineFitResult:
    """Posterior and conveniences for one line fit.

    Returned by :meth:`LineFitSetup.run`; not constructed directly.

    Attributes
    ----------
    idata : xarray.DataTree
        The sampler output (posterior, sample_stats, log_likelihood).
    model_name : str
        The model's label (from :meth:`NoobSpectrum.setup`), used by
        :meth:`compare`.
    components : tuple of ComponentMeta
        Component ids and signs, in build order.
    bounded : tuple of BoundedParam
        Every truncated free parameter and its bounds (for :meth:`boundary_report`).
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
    model_name: str
    components: tuple[ComponentMeta, ...]
    bounded: tuple[BoundedParam, ...]
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

    def boundary_report(
        self, *, rtol: float = 0.02, frac_threshold: float = 0.1
    ) -> Any:
        """Flag free parameters whose posterior piles up at a prior bound.

        Mass stacked against a truncation bound means the data wants to leave the
        allowed range -- e.g. a broad component pinned at the FWHM floor (it is
        not really broad) or a flux pressed to its lower limit. This is evidence a
        component is unwarranted, not a sampling artifact; weigh it with
        :meth:`significance_report` and
        :meth:`compare`.

        Parameters
        ----------
        rtol : float, default 0.02
            Edge band width as a fraction of each parameter's ``upper - lower``.
        frac_threshold : float, default 0.1
            Posterior fraction within an edge band above which the parameter is
            flagged.

        Returns
        -------
        pandas.DataFrame
            One row per bounded free parameter: its component, quantity, bounds,
            the posterior fraction at each edge, and a ``flagged`` boolean.
            Empty when the model has no truncated free parameters.
        """
        import pandas as pd

        rows = []
        for bp in self.bounded:
            samples = self._flat(bp.name)
            tol = rtol * (bp.upper - bp.lower)
            at_lower = float(np.mean(samples <= bp.lower + tol))
            at_upper = float(np.mean(samples >= bp.upper - tol))
            rows.append(
                {
                    "parameter": bp.name,
                    "component": bp.component_id,
                    "quantity": bp.quantity,
                    "lower": bp.lower,
                    "upper": bp.upper,
                    "frac_at_lower": at_lower,
                    "frac_at_upper": at_upper,
                    "flagged": (at_lower + at_upper) > frac_threshold,
                }
            )
        return pd.DataFrame(rows).set_index("parameter") if rows else pd.DataFrame()

    def significance_report(self, *, snr_warranted: float = 3.0) -> Any:
        """Rank each component by how far its flux sits above zero.

        A component whose integrated flux is consistent with zero (low
        signal-to-noise) is not supported by the data -- the canonical test for
        whether a broad or outflow component is real.

        Parameters
        ----------
        snr_warranted : float, default 3.0
            Flux ``median / sd`` at or above which a component is marked
            ``warranted``.

        Returns
        -------
        pandas.DataFrame
            Indexed by component id: flux median, sd, lower 16th percentile, the
            ``snr`` (median / sd), and a ``warranted`` boolean. A fixed-flux
            component has zero spread and infinite ``snr``.
        """
        import pandas as pd

        rows = []
        for comp in self.components:
            flux = self._flat(f"{comp.id}__flux")
            median = float(np.median(flux))
            sd = float(np.std(flux))
            snr = median / sd if sd > 0 else float("inf")
            rows.append(
                {
                    "component": comp.id,
                    "flux_median": median,
                    "flux_sd": sd,
                    "flux_p16": float(np.percentile(flux, 16.0)),
                    "snr": snr,
                    "warranted": snr >= snr_warranted,
                }
            )
        return pd.DataFrame(rows).set_index("component")

    def continuum_curve(self, grid: np.ndarray) -> np.ndarray:
        """Return the posterior-median continuum on ``grid``."""
        grid = np.asarray(grid, dtype=float)
        c0 = float(np.median(self._flat("continuum__c0")))
        if self.continuum_degree >= 1:
            c1 = float(np.median(self._flat("continuum__c1")))
            return c0 + c1 * (grid - self.lambda0)
        return np.full_like(grid, c0)

    def component_curve(
        self, component_id: str, grid: np.ndarray, *, with_continuum: bool = True
    ) -> np.ndarray:
        """Return the posterior-median curve of one component on ``grid``.

        Parameters
        ----------
        component_id : str
            The component id.
        grid : numpy.ndarray
            Wavelengths to evaluate at.
        with_continuum : bool, default True
            Add the median continuum, so the component sits on it (the natural
            decomposition overlay); ``False`` returns the bare line profile.

        Returns
        -------
        numpy.ndarray
        """
        grid = np.asarray(grid, dtype=float)
        sign = {c.id: c.sign for c in self.components}[component_id]
        mu = float(np.median(self._flat(f"{component_id}__mu")))
        sw = float(np.median(self._flat(f"{component_id}__sigma_w")))
        amp = float(np.median(self._flat(f"{component_id}__amp")))
        line = sign * amp * np.exp(-0.5 * ((grid - mu) / sw) ** 2)
        return line + self.continuum_curve(grid) if with_continuum else line

    def plot(
        self,
        *,
        full: bool = False,
        residual: bool = True,
        decompose: bool = True,
        **kwargs: Any,
    ) -> Figure:
        """Plot the data with the posterior model, decomposition, and residuals.

        Parameters
        ----------
        full : bool, default False
            Plot the whole spectrum; otherwise just the fitted window.
        residual : bool, default True
            Add a shared-x panel of standardized residuals ``(data - model) / sigma``.
        decompose : bool, default True
            Overlay each component's median curve (on the continuum) alongside the
            total posterior band.
        **kwargs
            Forwarded to
            :func:`~noobfriend.inference.spectrum._plot.plot_result` (e.g.
            ``size``, ``title``, ``save``).

        Returns
        -------
        matplotlib.figure.Figure
        """
        from noobfriend.inference.spectrum._plot import plot_result

        return plot_result(
            self, full=full, residual=residual, decompose=decompose, **kwargs
        )

    def compare(
        self,
        others: LineFitResult | Sequence[LineFitResult] | Mapping[str, LineFitResult],
        *,
        label: str | None = None,
    ) -> Any:
        """Compare this fit to one or more others by PSIS-LOO.

        Parameters
        ----------
        others : LineFitResult, sequence of LineFitResult, or mapping
            The fit(s) to compare against. Each is named by its ``model_name``
            unless given as a ``name -> result`` mapping. This fit and all of
            ``others`` must carry a ``log_likelihood`` group (the default from
            :meth:`LineFitSetup.run`).
        label : str, optional
            Override the name for *this* fit (defaults to its ``model_name``).

        Returns
        -------
        pandas.DataFrame
            ArviZ's ranked comparison table; the highest expected log predictive
            density (rank 0) is favoured.

        Notes
        -----
        Duplicate names (rare, since the default ``model_name`` encodes the
        component count) get a ``.2`` / ``.3`` suffix with a warning.
        """
        import arviz as az

        pairs: list[tuple[str, Any]] = [
            (label if label is not None else self.model_name, self.idata)
        ]
        if isinstance(others, LineFitResult):
            pairs.append((others.model_name, others.idata))
        elif isinstance(others, Mapping):
            pairs.extend((name, res.idata) for name, res in others.items())
        else:
            pairs.extend((res.model_name, res.idata) for res in others)

        counts: dict[str, int] = {}
        named: dict[str, Any] = {}
        for name, idata in pairs:
            counts[name] = counts.get(name, 0) + 1
            if counts[name] > 1:
                unique = f"{name}.{counts[name]}"
                warnings.warn(
                    f"duplicate model name {name!r}; using {unique!r}.", stacklevel=2
                )
                name = unique
            named[name] = idata
        return az.compare(named)

    def __repr__(self) -> str:
        """Concise representation: model name, component count, draws."""
        n = int(np.asarray(self.idata.posterior["continuum__c0"].values).size)
        return (
            f"LineFitResult({self.model_name!r}, "
            f"{len(self.components)} components, {n} draws)"
        )
