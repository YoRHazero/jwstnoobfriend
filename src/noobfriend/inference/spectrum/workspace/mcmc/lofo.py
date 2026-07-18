r"""Frame-level leave-one-out cross-validation for joint multi-frame fits.

Pixel PSIS-LOO (see :mod:`~.criteria`) drops one pixel; leave-one-frame-out
(LOFO) drops a whole exposure, which is the right unit for asking whether one
frame is consistent with the shared line model. Naive PSIS on a frame's summed
pixel log-likelihood fails silently for a per-frame continuum: the frame's own
continuum coefficients are constrained only by that frame, so importance
reweighting cannot represent the leave-it-out posterior (the coefficients would
revert to their prior). The result is an optimistic bias with, at best, poor
Pareto-k.

The fix is to **analytically marginalize the per-frame continuum** under its
(pooling) prior. For fixed shared parameters the per-frame continuum is a linear
Gaussian model, so the frame's marginal likelihood integrating the coefficients
out is a multivariate normal with a low-rank-plus-diagonal covariance
(:math:`X W X^\top + \Sigma`), evaluated per posterior draw via the Woodbury
identity. PSIS-LOO on that collapsed per-frame log-likelihood needs no refit and
has no continuum leak. The ``elpd_i <= lpd_i`` inequality (leaving data out
cannot beat in-sample) is used as a cheap red flag; a high per-frame Pareto-k
marks a frame influential enough to warrant an exact refit.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from html import escape
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import logsumexp

from noobfriend.inference.spectrum.workspace.mcmc.frames import (
    _draw_indices,
    _line_model_draws,
    _line_parameter_draws,
)

if TYPE_CHECKING:
    from noobfriend.inference.spectrum.workspace.mcmc.result import MCMCFitResult


class FrameLOOWarning(UserWarning):
    """Warning that a frame-level LOO estimate is unreliable for some frame."""


@dataclass(frozen=True, slots=True, repr=False)
class FrameLOOResult:
    """Leave-one-frame-out PSIS cross-validation for a joint fit.

    ``elpd_i`` is the per-frame expected log predictive density; it scales with
    a frame's pixel count, so use ``pareto_k`` and the ``elpd_i <= lpd_i``
    self-check (not raw ``elpd_i``) to judge whether a frame is anomalous.
    ``flagged`` lists frames whose Pareto-k exceeds ``good_k`` or that violate
    the inequality; such frames are influential and deserve an exact refit.
    """

    frame_ids: tuple[str, ...]
    elpd: float
    standard_error: float
    elpd_i: np.ndarray
    pareto_k: np.ndarray
    lpd_i: np.ndarray
    good_k: float
    n_pixels: np.ndarray
    n_draws: int

    @property
    def inequality_gap(self) -> np.ndarray:
        """Per-frame ``elpd_i - lpd_i``; a positive value is a red flag."""
        return self.elpd_i - self.lpd_i

    @property
    def flagged(self) -> tuple[str, ...]:
        """Frame ids with high Pareto-k or a violated inequality."""
        gap = self.inequality_gap
        return tuple(
            frame_id
            for index, frame_id in enumerate(self.frame_ids)
            if self.pareto_k[index] > self.good_k or gap[index] > 0.0
        )

    @property
    def reliable(self) -> bool:
        """Whether every frame's LOO estimate passed both checks."""
        return len(self.flagged) == 0

    def __repr__(self) -> str:
        """Return a compact frame-LOO summary."""
        return (
            f"FrameLOOResult(n_frames={len(self.frame_ids)}, elpd={self.elpd:.6g}, "
            f"max_pareto_k={float(np.max(self.pareto_k)):.3g}, "
            f"reliable={self.reliable})"
        )

    def _repr_html_(self) -> str:
        """Return a notebook table of per-frame LOO values."""
        gap = self.inequality_gap
        rows = "\n".join(
            f"""    <tr>
      <td><code>{escape(frame_id)}</code></td>
      <td>{self.elpd_i[index]:.6g}</td>
      <td>{self.pareto_k[index]:.3g}</td>
      <td>{gap[index]:+.3g}</td>
      <td>{int(self.n_pixels[index])}</td>
    </tr>"""
            for index, frame_id in enumerate(self.frame_ids)
        )
        return f"""<section class="noob-frame-loo">
  <h3>Frame LOO (elpd={self.elpd:.6g} &plusmn; {self.standard_error:.3g})</h3>
  <p><strong>good_k:</strong> {self.good_k:.3g};
     <strong>reliable:</strong> {self.reliable}</p>
  <table>
    <thead><tr><th>Frame</th><th>elpd_i</th><th>pareto_k</th>
      <th>elpd_i-lpd_i</th><th>pixels</th></tr></thead>
    <tbody>
{rows}
    </tbody>
  </table>
</section>"""


def build_frame_loo(
    result: MCMCFitResult,
    *,
    max_draws: int | None = 1000,
    random_seed: int = 1729,
) -> FrameLOOResult:
    """Compute leave-one-frame-out PSIS cross-validation.

    A ``"pooled"`` continuum is integrated out analytically per frame under its
    pooling prior (the collapsed marginal that removes the per-frame leak). A
    ``"shared"`` continuum has no per-frame nuisance to leak, so the frame's
    likelihood is evaluated directly at the posterior draws. ``"independent"``
    continua are not supported.

    Parameters
    ----------
    result
        A sampled joint (multi-frame) MCMC result with a pooled or shared
        continuum.
    max_draws
        Cap on posterior draws used to reconstruct the per-frame line model;
        ``None`` uses every draw. Subsampling keeps the reconstruction fast.
    random_seed
        Seed for draw subsampling.

    Returns
    -------
    FrameLOOResult
        Per-frame LOO elpd, Pareto-k, in-sample lpd, and reliability flags.

    Raises
    ------
    ValueError
        If the fit has fewer than two frames.
    NotImplementedError
        If the continuum is ``"independent"``.

    Warns
    -----
    FrameLOOWarning
        If any frame's Pareto-k exceeds ``good_k`` or violates ``elpd_i <=
        lpd_i``; refit those frames exactly to trust their numbers.
    """
    workspace = result.inputs.workspace
    if workspace.n_frames < 2:
        raise ValueError("frame LOO requires at least two frames.")
    sharing = workspace.continuum.sharing
    if sharing not in ("pooled", "shared"):
        raise NotImplementedError(
            f"frame LOO supports 'pooled' and 'shared' continua; got {sharing!r}."
        )

    indices = _draw_indices(result, max_draws=max_draws, random_seed=random_seed)
    line_draws = _line_parameter_draws(result, indices)
    if sharing == "pooled":
        mu, tau = _continuum_hyperparameter_draws(result, indices)
    else:
        continuum_coefficients = _shared_continuum_draws(result, indices)

    log_likelihood = np.empty((indices.size, workspace.n_frames), dtype=float)
    n_pixels = np.empty(workspace.n_frames, dtype=int)
    for frame_index, frame in enumerate(workspace.spectra):
        wavelength = np.asarray(frame.valid_wavelength, dtype=float)
        data = np.asarray(frame.valid_flux, dtype=float)
        variance = np.asarray(frame.valid_error, dtype=float) ** 2
        design = workspace.continuum.design(wavelength)
        line_model = _line_model_draws(line_draws, wavelength, frame.resolving_power)
        if sharing == "pooled":
            log_likelihood[:, frame_index] = _collapsed_frame_log_likelihood(
                data=data,
                variance=variance,
                design=design,
                line_model=line_model,
                mu=mu,
                tau=tau,
            )
        else:
            log_likelihood[:, frame_index] = _shared_frame_log_likelihood(
                data=data,
                variance=variance,
                design=design,
                line_model=line_model,
                coefficients=continuum_coefficients,
            )
        n_pixels[frame_index] = wavelength.size

    return _summarize_frame_loo(
        result, log_likelihood, n_pixels=n_pixels, n_draws=indices.size
    )


def _continuum_hyperparameter_draws(
    result: MCMCFitResult, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    continuum = result.posterior["continuum"]
    names = result.inputs.workspace.continuum.parameter_names
    mu = np.stack(
        [continuum[f"{name}__mu"].samples.reshape(-1)[indices] for name in names]
    )
    tau = np.stack(
        [continuum[f"{name}__tau"].samples.reshape(-1)[indices] for name in names]
    )
    return mu, tau


def _shared_continuum_draws(result: MCMCFitResult, indices: np.ndarray) -> np.ndarray:
    continuum = result.posterior["continuum"]
    names = result.inputs.workspace.continuum.parameter_names
    return np.stack([continuum[name].samples.reshape(-1)[indices] for name in names])


def _collapsed_frame_log_likelihood(
    *,
    data: np.ndarray,
    variance: np.ndarray,
    design: np.ndarray,
    line_model: np.ndarray,
    mu: np.ndarray,
    tau: np.ndarray,
) -> np.ndarray:
    """Per-draw marginal log-likelihood with the continuum integrated out.

    The frame is ``y ~ N(design @ beta + line_model, diag(variance))`` with the
    pooling prior ``beta ~ N(mu, diag(tau**2))``; integrating ``beta`` gives a
    normal with covariance ``design @ diag(tau**2) @ design.T + diag(variance)``,
    evaluated here via Woodbury without forming the dense covariance.
    """
    n_pixels = data.size
    inverse_variance = 1.0 / variance
    design_gram = design.T @ (inverse_variance[:, None] * design)  # (p, p)

    mean = (design @ mu).T + line_model  # (S, n)
    residual = data[None, :] - mean  # (S, n)

    tau_squared = tau.T**2  # (S, p)
    inverse_prior = 1.0 / tau_squared  # (S, p)
    normal_matrix = design_gram[None] + _diag_batch(inverse_prior)  # (S, p, p)

    design_weighted_residual = (design.T @ (inverse_variance[:, None] * residual.T)).T
    solved = np.linalg.solve(normal_matrix, design_weighted_residual[..., None])[..., 0]
    inverse_cov_residual = inverse_variance[None] * (residual - solved @ design.T)
    quadratic = np.sum(residual * inverse_cov_residual, axis=1)

    log_det = (
        np.sum(np.log(variance))
        + np.sum(np.log(tau_squared), axis=1)
        + np.linalg.slogdet(normal_matrix)[1]
    )
    return -0.5 * (n_pixels * np.log(2.0 * np.pi) + log_det + quadratic)


def _shared_frame_log_likelihood(
    *,
    data: np.ndarray,
    variance: np.ndarray,
    design: np.ndarray,
    line_model: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Per-draw Gaussian log-likelihood at the shared continuum.

    A shared continuum has no per-frame nuisance to integrate out — leaving a
    frame out cannot revert a frame-specific coefficient to its prior — so the
    frame likelihood is evaluated directly at each draw's continuum
    ``design @ coefficients`` plus the shared line model.
    """
    continuum = (design @ coefficients).T  # (S, n)
    residual = data[None, :] - (continuum + line_model)  # (S, n)
    return -0.5 * np.sum(
        np.log(2.0 * np.pi * variance)[None, :] + residual**2 / variance[None, :],
        axis=1,
    )


def _diag_batch(values: np.ndarray) -> np.ndarray:
    batch, size = values.shape
    out = np.zeros((batch, size, size), dtype=float)
    diagonal = np.arange(size)
    out[:, diagonal, diagonal] = values
    return out


def _summarize_frame_loo(
    result: MCMCFitResult,
    log_likelihood: np.ndarray,
    *,
    n_pixels: np.ndarray,
    n_draws: int,
) -> FrameLOOResult:
    try:
        import arviz as az
        import xarray as xr
    except ImportError as error:  # pragma: no cover - depends on optional environment
        raise ImportError(
            "Frame LOO requires the 'mcmc' optional dependency."
        ) from error

    frame_ids = result.inputs.workspace.frame_ids
    n_samples, n_frames = log_likelihood.shape
    array = xr.DataArray(
        log_likelihood.reshape(1, n_samples, n_frames),
        dims=["chain", "draw", "frame"],
        coords={"frame": list(frame_ids)},
    )
    posterior = xr.Dataset({"_": (["chain", "draw"], np.zeros((1, n_samples)))})
    idata = xr.DataTree.from_dict(
        {"posterior": posterior, "log_likelihood": xr.Dataset({"frame": array})}
    )
    loo = az.loo(idata, var_name="frame", pointwise=True)

    elpd_i = np.asarray(loo.elpd_i, dtype=float).reshape(-1)
    lpd_i = logsumexp(log_likelihood, axis=0) - np.log(n_samples)
    outcome = FrameLOOResult(
        frame_ids=frame_ids,
        elpd=float(loo.elpd),
        standard_error=float(loo.se),
        elpd_i=_readonly(elpd_i),
        pareto_k=_readonly(np.asarray(loo.pareto_k, dtype=float).reshape(-1)),
        lpd_i=_readonly(lpd_i),
        good_k=float(loo.good_k),
        n_pixels=_readonly(n_pixels.astype(float)),
        n_draws=n_draws,
    )
    if not outcome.reliable:
        warnings.warn(
            "frame LOO is unreliable for "
            f"{', '.join(outcome.flagged)}: high Pareto-k or a violated "
            "elpd_i <= lpd_i inequality. Refit those frames exactly to trust "
            "their leave-one-out values.",
            FrameLOOWarning,
            stacklevel=3,
        )
    return outcome


def _readonly(value: np.ndarray) -> np.ndarray:
    output = np.array(value, dtype=float, copy=True)
    output.setflags(write=False)
    return output
