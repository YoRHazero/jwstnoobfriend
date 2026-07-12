"""MCMC-specific corner and pointwise diagnostic figures."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from noobfriend.inference.spectrum.workspace.mcmc import MCMCFitResult


def plot_mcmc_corner(
    result: MCMCFitResult,
    *,
    variables: Sequence[tuple[str, str]] | None = None,
    max_draws: int | None = 3000,
    random_seed: int = 1729,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Plot posterior pairs addressed by public component/parameter keys."""
    try:
        import arviz as az
    except ImportError as error:  # pragma: no cover - optional environment
        raise ImportError(
            "Spectrum MCMC corner plots require the 'mcmc' optional dependency."
        ) from error

    selected = _corner_variables(result, variables)
    if not selected:
        raise ValueError("plot_corner requires at least one posterior variable.")
    total_draws = next(iter(selected.values())).size
    indices = _draw_indices(
        total_draws,
        max_draws=max_draws,
        random_seed=random_seed,
    )
    posterior = {name: values[indices][None, :] for name, values in selected.items()}
    plot_data = az.from_dict({"posterior": posterior})
    if len(selected) == 1:
        return _plot_single_corner_variable(
            next(iter(selected)),
            next(iter(posterior.values())).reshape(-1),
            figsize=figsize,
        )
    pair_kwargs = {
        "var_names": list(selected),
        "marginal": True,
        "triangle": "lower",
        "backend": "matplotlib",
        "visuals": {"scatter": {"s": 5, "alpha": 0.2}},
    }
    plot_matrix = az.plot_pair(plot_data, **pair_kwargs)
    figure = plot_matrix.viz["figure"].item()
    if figsize is None:
        side = min(24.0, max(6.0, 2.2 * len(selected)))
        figure.set_size_inches(side, side)
    else:
        figure.set_size_inches(*figsize)
    label_size = 9 if len(selected) <= 5 else 7
    for axis in figure.axes:
        axis.tick_params(labelsize=label_size)
        axis.xaxis.label.set_size(label_size)
        axis.yaxis.label.set_size(label_size)
    figure.tight_layout()
    return figure


def plot_mcmc_pareto_k(
    result: MCMCFitResult,
    *,
    size: int = 900,
    title: str | None = None,
) -> Figure:
    """Plot one PSIS Pareto-k value for every valid fitted pixel."""
    import matplotlib.pyplot as plt

    wavelength = result.inputs.data.valid_wavelength
    pareto_k = result.criteria.psis_loo.pareto_k
    if wavelength.shape != pareto_k.shape:
        raise RuntimeError(
            "Pareto-k values no longer align with the valid input wavelength array."
        )
    good_k = result.criteria.psis_loo.good_k
    width = size / 100.0
    figure, axis = plt.subplots(
        figsize=(width, width * 0.45),
        layout="constrained",
    )
    good = pareto_k <= good_k
    axis.plot(wavelength, pareto_k, color="#7F7F7F", linewidth=0.8, alpha=0.7)
    axis.scatter(
        wavelength[good],
        pareto_k[good],
        color="#4C78A8",
        s=18,
        label=f"k ≤ {good_k:.3g}",
        zorder=3,
    )
    if np.any(~good):
        axis.scatter(
            wavelength[~good],
            pareto_k[~good],
            color="#D62728",
            s=24,
            label=f"k > {good_k:.3g}",
            zorder=4,
        )
    axis.axhline(
        good_k,
        color="#D62728",
        linestyle="--",
        linewidth=1.2,
        label="PSIS reliability threshold",
    )
    axis.set_xlabel(f"Observed wavelength [{result.inputs.data.unit}]")
    axis.set_ylabel("Pareto k")
    if title is not None:
        axis.set_title(title)
    axis.legend(frameon=False)
    return figure


def _corner_variables(
    result: MCMCFitResult,
    variables: Sequence[tuple[str, str]] | None,
) -> dict[str, np.ndarray]:
    requested = (
        tuple(
            (component, parameter)
            for component in result.posterior.components
            for parameter in result.posterior[component].parameters
        )
        if variables is None
        else tuple(variables)
    )
    output: dict[str, np.ndarray] = {}
    for item in requested:
        if len(item) != 2:
            raise ValueError("corner variables must be (component, parameter) pairs.")
        component, parameter = item
        label = f"{component}.{parameter}"
        if label in output:
            raise ValueError(f"corner variable {item!r} was requested more than once.")
        output[label] = result.posterior[component][parameter].samples.reshape(-1)
    return output


def _draw_indices(
    total_draws: int,
    *,
    max_draws: int | None,
    random_seed: int,
) -> np.ndarray:
    if max_draws is not None:
        if isinstance(max_draws, bool) or not isinstance(max_draws, int):
            raise TypeError("max_draws must be a positive integer or None.")
        if max_draws <= 0:
            raise ValueError("max_draws must be a positive integer or None.")
    if (
        isinstance(random_seed, bool)
        or not isinstance(random_seed, int)
        or random_seed < 0
    ):
        raise ValueError("random_seed must be a nonnegative integer.")
    if max_draws is None or max_draws >= total_draws:
        return np.arange(total_draws)
    rng = np.random.default_rng(random_seed)
    return np.sort(rng.choice(total_draws, size=max_draws, replace=False))


def _plot_single_corner_variable(
    label: str,
    values: np.ndarray,
    *,
    figsize: tuple[float, float] | None,
) -> Figure:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(
        figsize=(6.0, 4.0) if figsize is None else figsize,
        layout="constrained",
    )
    axis.hist(values, bins="auto", density=True, color="#4C78A8", alpha=0.8)
    axis.axvline(
        float(np.median(values)),
        color="#D62728",
        linewidth=1.5,
        label="median",
    )
    axis.set_xlabel(label)
    axis.set_ylabel("Density")
    axis.legend(frameon=False)
    return figure
