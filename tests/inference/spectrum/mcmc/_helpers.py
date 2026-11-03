"""Synthetic MCMC results built without sampling.

``fake_result`` draws every free and pooled parameter around a chosen value,
evaluates the physical quantities and log likelihoods with the workspace's
own NumPy path, and wraps them exactly as ``sample()`` would. Result-level
features (access, decompose, plots, ``as_prior``, save/load) can then be
tested in milliseconds.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from noobfriend.inference.spectrum._xp import numpy_namespace
from noobfriend.inference.spectrum.workspace import NoobFitWorkspace


def _center(row: object) -> float:
    domain = row.domain  # type: ignore[attr-defined]
    if domain.name == "fraction":
        return 0.5
    if domain.name == "correlation":
        return 0.3
    if domain.low == 0.0:
        return row.scale  # type: ignore[attr-defined]
    return 0.0


def fake_result(
    workspace: NoobFitWorkspace,
    truth: Mapping[str, float] | None = None,
    *,
    spread: float = 0.01,
    chains: int = 4,
    draws: int = 1000,
    seed: int = 0,
) -> object:
    """Return a ``NoobMCMCResult`` with Gaussian draws around ``truth``."""
    import xarray as xr

    from noobfriend.inference.spectrum.mcmc import NoobMCMCResult, SamplingReport
    from noobfriend.inference.spectrum.mcmc._diagnostics import diagnose
    from noobfriend.inference.spectrum.mcmc._names import physical_names

    truth = dict(truth or {})
    rng = np.random.default_rng(seed)
    size = chains * draws
    values = {}
    for row in workspace.parameters.values():
        if row.kind == "fixed":
            continue
        default = _center(row)
        if row.kind == "pooled":
            default = truth.get(row.pooled_from, default)
        center = truth.get(row.name, default)
        values[row.name] = center + spread * max(abs(center), row.scale) * rng.normal(
            size=(size, 1)
        )
    physical = workspace.physical(values)
    posterior = {
        name: (
            ("chain", "draw"),
            np.broadcast_to(
                np.asarray(physical[name], dtype=float).reshape(-1), (size,)
            ).reshape(chains, draws),
        )
        for name in physical_names(workspace)
    }
    xp = numpy_namespace()
    pixels, pixel_frames, pixel_index, frames = [], [], [], []
    for frame_id, frame in workspace.spectrum.frames.items():
        mask = workspace.fitted[frame_id]
        resid = (
            np.asarray(frame.flux)[mask]
            - workspace.frame_model(xp, physical, frame_id).total
        )
        scale = np.broadcast_to(physical[f"frame.{frame_id}.noise.scale"], (size, 1))[
            :, 0
        ]
        rho = physical.get(f"frame.{frame_id}.noise.rho")
        rho = None if rho is None else np.asarray(rho).reshape(-1)
        noise = workspace.frame_noise[frame_id]
        pixels.append(noise.pointwise(xp, resid, scale, rho).reshape(chains, draws, -1))
        frames.append(
            noise.log_likelihood(xp, resid, scale, rho).reshape(chains, draws)
        )
        pixel_frames += [frame_id] * noise.size
        pixel_index.append(np.flatnonzero(mask))
    log_likelihood = xr.Dataset(
        {
            "pixels": (("chain", "draw", "pixel"), np.concatenate(pixels, axis=-1)),
            "frames": (("chain", "draw", "frame"), np.stack(frames, axis=-1)),
        },
        coords={
            "frame": list(workspace.frame_ids),
            "pixel_frame": ("pixel", pixel_frames),
            "pixel_index": ("pixel", np.concatenate(pixel_index)),
        },
    )
    stats = xr.Dataset(
        {
            "diverging": (("chain", "draw"), np.zeros((chains, draws), dtype=bool)),
            "energy": (("chain", "draw"), rng.normal(size=(chains, draws))),
        }
    )
    idata = xr.DataTree.from_dict(
        {
            "posterior": xr.Dataset(posterior),
            "internal": xr.Dataset(),
            "sample_stats": stats,
            "log_likelihood": log_likelihood,
        }
    )
    diagnostics, verdict = diagnose(idata)
    report = SamplingReport(
        options={
            "draws": draws,
            "tune": 0,
            "chains": chains,
            "target_accept": 0.95,
            "random_seed": seed,
            "groups": [chains],
        },
        elapsed_seconds=0.0,
        diagnostics=diagnostics,
        verdict=verdict,
        reparameterizations=(),
        replaced_model=False,
    )
    return NoobMCMCResult(workspace, idata, report)
