"""Save and load sampled spectrum MCMC results.

A saved result is a directory of three files:

- ``idata.nc`` — the raw chains (posterior, sample stats, pointwise log
  likelihood, observed data) as an arviz-compatible NetCDF tree. The chains
  are the provenance of a run; this file is readable by any arviz/xarray
  installation without noobfriend.
- ``inputs.pkl`` — the pickled fit inputs (workspace and sampling options)
  needed to rebuild physical results.
- ``manifest.json`` — human-readable provenance plus the sampling
  diagnostics, so a saved run's validity can be checked before loading any
  Python objects.

Loading re-derives the posterior summaries, diagnostics, and predictive
criteria from the saved chains through the same builders that run after
sampling (recompiling the PyMC model to recover variable naming), so a
loaded result is always consistent with the chains on disk.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from noobfriend.inference.spectrum.workspace.mcmc.result import MCMCFitResult

_FORMAT_VERSION = 1
_IDATA_FILE = "idata.nc"
_INPUTS_FILE = "inputs.pkl"
_MANIFEST_FILE = "manifest.json"
_FILES = (_IDATA_FILE, _INPUTS_FILE, _MANIFEST_FILE)


def save_mcmc_result(
    result: MCMCFitResult, path: str | Path, *, overwrite: bool = False
) -> Path:
    """Write a sampled result to ``path`` for later :func:`load_mcmc_result`.

    Parameters
    ----------
    result
        The sampled result to persist.
    path
        Target directory, created (with parents) if needed.
    overwrite
        Whether existing persistence files in ``path`` may be replaced.

    Returns
    -------
    pathlib.Path
        The directory the result was written to.

    Raises
    ------
    FileExistsError
        If any persistence file already exists and ``overwrite`` is false.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        existing = [name for name in _FILES if (directory / name).exists()]
        if existing:
            names = ", ".join(existing)
            raise FileExistsError(
                f"refusing to overwrite {names} in {directory}; "
                "pass overwrite=True to replace a saved result."
            )

    result.idata.to_netcdf(directory / _IDATA_FILE, engine="h5netcdf")
    payload = {
        "format_version": _FORMAT_VERSION,
        "workspace": result.inputs.workspace,
        "options": result.inputs.options,
        "elapsed_seconds": result.sampling.elapsed_seconds,
    }
    with (directory / _INPUTS_FILE).open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = _build_manifest(result)
    (directory / _MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return directory


def load_mcmc_result(path: str | Path) -> MCMCFitResult:
    """Rebuild a result saved by :func:`save_mcmc_result`.

    The workspace and options are unpickled, the chains are read back from
    NetCDF, and the posterior, diagnostics, and criteria are re-derived from
    those chains through the regular result builders. Rebuilding recompiles
    the PyMC model to recover variable naming, so loading requires the
    ``mcmc`` optional dependency.

    Parameters
    ----------
    path
        Directory previously written by :func:`save_mcmc_result`.

    Returns
    -------
    MCMCFitResult
        The reconstructed result.

    Raises
    ------
    FileNotFoundError
        If ``path`` is missing any of the persistence files.
    ValueError
        If the saved format version is not supported by this library.
    """
    directory = Path(path)
    missing = [name for name in _FILES if not (directory / name).exists()]
    if missing:
        names = ", ".join(missing)
        raise FileNotFoundError(
            f"not a saved MCMC result: {directory} is missing {names}."
        )

    manifest = json.loads((directory / _MANIFEST_FILE).read_text(encoding="utf-8"))
    _check_format_version(manifest.get("format_version"), source=_MANIFEST_FILE)
    with (directory / _INPUTS_FILE).open("rb") as stream:
        payload = pickle.load(stream)
    _check_format_version(payload.get("format_version"), source=_INPUTS_FILE)

    from noobfriend.inference.spectrum.workspace.mcmc.model import (
        build_workspace_model,
    )
    from noobfriend.inference.spectrum.workspace.mcmc.result import build_mcmc_result

    workspace = payload["workspace"]
    built = build_workspace_model(workspace)
    return build_mcmc_result(
        workspace,
        _read_idata(directory / _IDATA_FILE),
        built.metadata,
        payload["options"],
        elapsed_seconds=payload["elapsed_seconds"],
    )


def _check_format_version(value: Any, *, source: str) -> None:
    if value != _FORMAT_VERSION:
        raise ValueError(
            f"unsupported saved-result format version {value!r} in {source}; "
            f"this noobfriend reads version {_FORMAT_VERSION}."
        )


def _read_idata(path: Path) -> Any:
    """Load the full chain tree into memory and release the file handle."""
    import xarray as xr

    tree = xr.open_datatree(path, engine="h5netcdf")
    try:
        return tree.load()
    finally:
        tree.close()


def _build_manifest(result: MCMCFitResult) -> dict[str, Any]:
    """Summarize provenance, options, and diagnostics as plain JSON data."""
    try:
        library_version = version("noobfriend")
    except PackageNotFoundError:  # pragma: no cover - depends on install mode
        library_version = "unknown"
    criteria = result.criteria
    return {
        "format_version": _FORMAT_VERSION,
        "noobfriend_version": library_version,
        "created_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": result.sampling.elapsed_seconds,
        "options": asdict(result.inputs.options),
        "frame_ids": list(result.inputs.workspace.frame_ids),
        "components": list(result.posterior.components),
        "diagnostics": asdict(result.sampling.diagnostics),
        "criteria": {
            "psis_loo": {
                "elpd": criteria.psis_loo.elpd,
                "standard_error": criteria.psis_loo.standard_error,
                "max_pareto_k": float(max(criteria.psis_loo.pareto_k)),
                "warning": criteria.psis_loo.warning,
            },
            "waic": {
                "elpd": criteria.waic.elpd,
                "standard_error": criteria.waic.standard_error,
                "warning": criteria.waic.warning,
            },
        },
    }
