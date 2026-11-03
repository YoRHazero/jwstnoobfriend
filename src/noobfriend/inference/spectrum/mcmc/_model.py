"""NoobSpectrumModel: the NumPyro model of a workspace, and its escape hatch."""

from __future__ import annotations

from collections.abc import Callable
from numbers import Integral, Real
from typing import TYPE_CHECKING

from noobfriend.inference.spectrum.mcmc._build import (
    Reparameterizations,
    build_numpyro_model,
)
from noobfriend.inference.spectrum.mcmc._names import physical_names

if TYPE_CHECKING:
    import xarray as xr

    from noobfriend.inference.spectrum.mcmc._result import NoobMCMCResult
    from noobfriend.inference.spectrum.mcmc._run import SamplingRun
    from noobfriend.inference.spectrum.workspace import NoobFitWorkspace


class NoobSpectrumModel:
    """The sampling model of a workspace (``ws.model()``).

    :attr:`numpyro_model` is a plain NumPyro model taking no arguments. The
    library guarantees two things about it, and about any replacement it will
    sample:

    - every physical parameter is a named ``numpyro.deterministic`` site
      using the workspace's flat names (``"Ha.fwhm"``, ``"continuum.c0"``,
      ``"frame.v1.noise.scale"``, ...);
    - every likelihood term is an observed sample site with a real
      distribution (``frame.<id>.flux``: ``Normal`` for white noise,
      ``MultivariateNormal(scale_tril=...)`` for a fixed correlation, an
      AR(1) Markov normal for AR(1)), never a bare ``factor``.

    To customize anything the library does not offer (a Student-t
    likelihood, an extra prior term), write a new model, typically wrapping
    or copying this one, and hand it back with :meth:`with_numpyro_model`.
    """

    def __init__(
        self,
        workspace: NoobFitWorkspace,
        model: Callable[[], None] | None = None,
        *,
        reparameterizations: Reparameterizations | None = None,
        replaced: bool = False,
    ) -> None:
        if model is None:
            model, reparameterizations = build_numpyro_model(workspace)
        self._workspace = workspace
        self._model = model
        self._reparameterizations = reparameterizations
        self._replaced = replaced

    @property
    def workspace(self) -> NoobFitWorkspace:
        """The workspace this model samples."""
        return self._workspace

    @property
    def numpyro_model(self) -> Callable[[], None]:
        """The zero-argument NumPyro model."""
        return self._model

    def with_numpyro_model(self, model: Callable[[], None]) -> NoobSpectrumModel:
        """Return a copy that samples ``model`` instead.

        Parameters
        ----------
        model
            Any zero-argument NumPyro model honouring the two site contracts.
            Workspace and parameter table stay the same, so results, plots
            and saving work as usual; ``decompose`` still draws the
            library's own mean model.

        Returns
        -------
        NoobSpectrumModel
            The replaced model.
        """
        if not callable(model):
            raise TypeError("with_numpyro_model expects a callable NumPyro model.")
        return NoobSpectrumModel(
            self._workspace,
            model,
            reparameterizations=self._reparameterizations,
            replaced=True,
        )

    def sample(
        self,
        draws: int = 1000,
        tune: int = 2000,
        chains: int = 16,
        target_accept: float = 0.95,
        random_seed: int = 1729,
        progressbar: bool = False,
    ) -> NoobMCMCResult:
        """Run NUTS and return the posterior.

        Dense mass matrix, float64, CPU only. Chains run in groups of as many
        as JAX has CPU devices (``NOOB_MCMC_CPU_CORES``, default every core),
        groups one after another.

        Parameters
        ----------
        draws
            Kept draws per chain.
        tune
            Warm-up iterations per chain.
        chains
            Number of chains; many chains make R-hat sensitive to multimodal
            or ridge-shaped posteriors.
        target_accept
            NUTS target acceptance. Raising it against divergences shrinks
            the step size; check ESS again afterwards.
        random_seed
            Seed of the first chain group.
        progressbar
            Show NumPyro's progress bars.

        Returns
        -------
        NoobMCMCResult
            The posterior. ``result.sampling.verdict`` holds the convergence
            verdict; an :class:`MCMCConvergenceWarning` lists every failed
            criterion.

        Raises
        ------
        ValueError
            If an option is invalid, or the model lacks a physical-parameter
            site or a likelihood site.
        """
        from noobfriend.inference.spectrum.mcmc._diagnostics import (
            SamplingReport,
            diagnose,
            frozen,
            warn_if_unconverged,
        )
        from noobfriend.inference.spectrum.mcmc._result import NoobMCMCResult
        from noobfriend.inference.spectrum.mcmc._run import (
            likelihood_sites,
            log_likelihood_dataset,
            run_chains,
            trace_model,
        )

        for name, value in (("draws", draws), ("tune", tune), ("chains", chains)):
            if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer; got {value!r}.")
        if not isinstance(target_accept, Real) or not 0.0 < target_accept < 1.0:
            raise ValueError(
                f"target_accept must lie in (0, 1); got {target_accept!r}."
            )

        trace = trace_model(self._model)
        names = physical_names(self._workspace)
        present = {
            name for name, site in trace.items() if not site.get("is_observed", False)
        }
        missing = [name for name in names if name not in present]
        if missing:
            raise ValueError(
                "the model lacks physical-parameter sites (numpyro.deterministic "
                f"with the workspace's flat names): {missing}."
            )
        sites = likelihood_sites(trace)
        if not sites:
            raise ValueError("the model has no observed likelihood site.")

        run = run_chains(
            self._model,
            draws=int(draws),
            tune=int(tune),
            chains=int(chains),
            target_accept=float(target_accept),
            random_seed=int(random_seed),
            progressbar=bool(progressbar),
        )
        log_likelihood = log_likelihood_dataset(
            self._workspace, self._model, run.samples, sites, replaced=self._replaced
        )
        idata = _inference_data(run, names, log_likelihood)
        diagnostics, verdict = diagnose(idata)
        warn_if_unconverged(verdict)
        report = SamplingReport(
            options=frozen(
                {
                    "draws": int(draws),
                    "tune": int(tune),
                    "chains": int(chains),
                    "target_accept": float(target_accept),
                    "random_seed": int(random_seed),
                    "groups": list(run.groups),
                }
            ),
            elapsed_seconds=run.elapsed_seconds,
            diagnostics=frozen(diagnostics),
            verdict=verdict,
            reparameterizations=(
                self._reparameterizations.describe()
                if self._reparameterizations
                else ()
            ),
            replaced_model=self._replaced,
        )
        return NoobMCMCResult(self._workspace, idata, report)

    def __repr__(self) -> str:
        """Name the workspace and whether the model was replaced."""
        kind = "replaced" if self._replaced else "library"
        return f"NoobSpectrumModel({self._workspace!r}, {kind})"


def _inference_data(
    run: SamplingRun, names: tuple[str, ...], log_likelihood: xr.Dataset
) -> xr.DataTree:
    """Split physical and internal sites into an ArviZ DataTree."""
    import xarray as xr

    posterior = run.posterior
    physical = [name for name in names if name in posterior.data_vars]
    internal = [name for name in posterior.data_vars if name not in physical]
    return xr.DataTree.from_dict(
        {
            "posterior": posterior[physical],
            "internal": posterior[internal],
            "sample_stats": run.sample_stats,
            "log_likelihood": log_likelihood,
        }
    )
