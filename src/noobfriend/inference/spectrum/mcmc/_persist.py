"""Save an MCMC result as one netCDF file and reattach it to a workspace.

Only the ArviZ InferenceData tree is written: ``posterior`` (physical
parameters, flat names), ``internal`` (sampling coordinates),
``sample_stats`` and ``log_likelihood``, with root attributes for the format
and library versions, the sampling options, the frame ids, whether the model
was replaced, and the workspace fingerprints. The file is plain ArviZ data:
``arviz.from_netcdf`` reads it without this library.
"""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

from noobfriend.inference.spectrum.mcmc._diagnostics import (
    SamplingReport,
    diagnose,
    frozen,
)

if TYPE_CHECKING:
    from noobfriend.inference.spectrum.mcmc._result import NoobMCMCResult
    from noobfriend.inference.spectrum.workspace import NoobFitWorkspace

FORMAT_VERSION = 1


def save(result: NoobMCMCResult, path: str | Path, *, overwrite: bool) -> Path:
    """Write ``result`` to ``path``; see :meth:`NoobMCMCResult.save`."""
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists; pass overwrite=True to replace it.")
    tree = result.idata.copy()
    report = result.sampling
    workspace = result.workspace
    tree.attrs = {
        "noobfriend_format": FORMAT_VERSION,
        "noobfriend_version": _library_version(),
        "sampling_options": json.dumps(dict(report.options)),
        "elapsed_seconds": float(report.elapsed_seconds),
        "frame_ids": json.dumps(list(workspace.frame_ids)),
        "replaced_model": int(report.replaced_model),
        "reparameterizations": json.dumps(list(report.reparameterizations)),
        "data_fingerprint": workspace.data_fingerprint,
        "model_fingerprint": workspace.model_fingerprint,
    }
    tree.to_netcdf(target, engine="h5netcdf")
    return target


def load(workspace: NoobFitWorkspace, path: str | Path) -> NoobMCMCResult:
    """Read a saved result back onto ``workspace``; see ``NoobFitWorkspace.load``."""
    import xarray as xr

    from noobfriend.inference.spectrum.mcmc._result import NoobMCMCResult

    tree = xr.open_datatree(Path(path), engine="h5netcdf")
    try:
        tree.load()
    finally:
        tree.close()
    attrs = dict(tree.attrs)
    if int(attrs.get("noobfriend_format", -1)) != FORMAT_VERSION:
        raise ValueError(
            f"{path} is not a noobfriend spectrum fit (format {FORMAT_VERSION})."
        )
    changed = []
    if attrs["data_fingerprint"] != workspace.data_fingerprint:
        changed.append(
            "the data (frame arrays, masks, fitted pixels, window, noise or resolving power)"
        )
    if attrs["model_fingerprint"] != workspace.model_fingerprint:
        changed.append(
            "the model (parameter names, priors, bindings or continuum form)"
        )
    if changed:
        raise ValueError(
            f"{path} was fitted on a different workspace: "
            + " and ".join(changed)
            + " changed. Rebuild the workspace with the original script."
        )
    tree.attrs = {}
    diagnostics, verdict = diagnose(tree)
    report = SamplingReport(
        options=frozen(json.loads(attrs["sampling_options"])),
        elapsed_seconds=float(attrs["elapsed_seconds"]),
        diagnostics=frozen(diagnostics),
        verdict=verdict,
        reparameterizations=tuple(json.loads(attrs["reparameterizations"])),
        replaced_model=bool(int(attrs["replaced_model"])),
    )
    return NoobMCMCResult(workspace, tree, report)


def _library_version() -> str:
    try:
        return version("noobfriend")
    except PackageNotFoundError:  # pragma: no cover - source checkout
        return "unknown"
