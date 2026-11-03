"""Plain-text summary of an MCMC result."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from noobfriend.inference.spectrum._text import format_rows

if TYPE_CHECKING:
    from noobfriend.inference.spectrum.mcmc._result import NoobMCMCResult

HDI_PROB = 0.94


def result_summary(result: NoobMCMCResult) -> str:
    """Render plan, sampling, verdict, posterior table, LOO and frames."""
    import arviz as az

    report = result.sampling
    options = report.options
    verdict = report.verdict
    groups = "+".join(str(size) for size in options.get("groups", ()))
    lines = [
        result.workspace.summary(),
        "",
        "Sampling",
        f"  {options['draws']} draws x {options['chains']} chains (groups {groups}), "
        f"tune {options['tune']}, target_accept {options['target_accept']}, "
        f"seed {options['random_seed']}, {report.elapsed_seconds:.1f} s",
        "  model: "
        + ("replaced with with_numpyro_model" if report.replaced_model else "library"),
    ]
    lines += [f"  reparameterized: {text}" for text in report.reparameterizations]
    lines.append(
        f"  divergences {verdict.divergences}, max R-hat {verdict.max_rhat:.4f}, "
        f"min ESS bulk {verdict.min_ess_bulk:.0f} / tail {verdict.min_ess_tail:.0f}, "
        f"min E-BFMI {verdict.min_bfmi:.3f}"
    )
    if verdict.converged:
        lines.append("  verdict: CONVERGED")
    else:
        lines.append("  verdict: NOT CONVERGED (do not report these numbers)")
        lines += [f"    - {failure}" for failure in verdict.failures]

    posterior = result.idata["posterior"].dataset
    units = {row.name: row.unit for row in result.workspace.parameters.values()}
    units |= {q.name: q.unit for q in result.workspace.parameters.quantities.values()}
    rows = []
    for name, array in posterior.data_vars.items():
        values = np.asarray(array.values, dtype=float)
        unit = units.get(name, "") or "-"
        diagnostic = report.diagnostics.get(name)
        if diagnostic is None:
            rows.append([name, unit, f"{np.median(values):.6g}", "fixed", "", "", ""])
            continue
        lower, upper = np.asarray(
            az.hdi(values.reshape(-1), prob=HDI_PROB), dtype=float
        )
        rows.append(
            [
                name,
                unit,
                f"{np.median(values):.6g}",
                f"{lower:.6g}",
                f"{upper:.6g}",
                f"{diagnostic.rhat:.3f}",
                f"{diagnostic.ess_bulk:.0f}",
            ]
        )
    lines += ["", f"Posterior (median, {HDI_PROB:.0%} HDI)"]
    lines += format_rows(
        ["name", "unit", "median", "hdi low", "hdi high", "R-hat", "ESS"], rows
    )

    loo = result.loo
    lines += ["", "PSIS-LOO over pixels (exact conditionals for correlated noise)"]
    lines.append(f"  elpd_loo {loo.elpd:.2f} +- {loo.se:.2f}, p_loo {loo.p:.2f}")
    lines.append("  " + _pareto_counts(loo))
    frames = result.frames
    lines += [
        "",
        "Frames" + (" (leave-one-frame-out)" if frames.loo is not None else ""),
    ]
    lines += format_rows(
        ["frame", "chi2", "pixels", "elpd", "pareto k"],
        [
            [
                f.id,
                f"{f.chi2:.1f}",
                str(f.n_pixels),
                f"{f.elpd:.2f}",
                f"{f.pareto_k:.2f}",
            ]
            for f in frames.values()
        ],
    )
    return "\n".join(lines)


def _pareto_counts(loo: object) -> str:
    k = np.asarray(getattr(loo, "pareto_k"), dtype=float).reshape(-1)
    good_k = float(getattr(loo, "good_k", 0.7))
    good = int(np.sum(k <= good_k))
    bad = int(np.sum((k > good_k) & (k <= 1.0)))
    very_bad = int(np.sum(k > 1.0))
    return (
        f"Pareto k: {good} good (<= {good_k:.2f}), {bad} bad (<= 1), "
        f"{very_bad} very bad (> 1) of {k.size}"
    )
