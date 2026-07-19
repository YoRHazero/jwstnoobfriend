"""Tests for saving and loading sampled MCMC results."""

from __future__ import annotations

import json

import numpy as np
import pytest

from noobfriend.inference.spectrum import (
    NoobLine,
    NoobSpectrum,
    NoobSpectrumSet,
)
from noobfriend.inference.spectrum.workspace.mcmc import MCMCFitResult

from ._helpers import gaussian

pytest.importorskip("pymc")


@pytest.fixture(scope="module")
def sampled_result() -> MCMCFitResult:
    wavelength = np.linspace(4990.0, 5010.0, 41)
    line = (
        NoobLine("line", obs=5000.0)
        .flux(override=8.0)
        .fwhm(override=180.0)
        .center(delta_v_kms=0.0)
    )
    error = np.full_like(wavelength, 0.1)
    data = 0.5 + gaussian(wavelength, center=5000.0, flux=8.0, fwhm_kms=180.0)
    workspace = NoobSpectrum(data, error, obs=wavelength).prepare(
        [line], continuum_order=0
    )
    return workspace.model().sample(
        draws=40, tune=40, chains=2, cores=1, random_seed=7, progressbar=False
    )


def test_save_writes_files_and_manifest(sampled_result, tmp_path) -> None:
    directory = sampled_result.save(tmp_path / "run")

    assert directory == tmp_path / "run"
    assert sorted(entry.name for entry in directory.iterdir()) == [
        "idata.nc",
        "inputs.pkl",
        "manifest.json",
    ]
    manifest = json.loads((directory / "manifest.json").read_text())
    diagnostics = sampled_result.sampling.diagnostics
    assert manifest["format_version"] == 1
    assert manifest["options"]["chains"] == 2
    assert manifest["options"]["draws"] == 40
    assert manifest["frame_ids"] == list(sampled_result.inputs.workspace.frame_ids)
    assert manifest["components"] == ["line", "continuum"]
    assert manifest["diagnostics"]["divergences"] == diagnostics.divergences
    assert manifest["diagnostics"]["max_rhat"] == diagnostics.max_rhat
    assert manifest["criteria"]["psis_loo"]["elpd"] == pytest.approx(
        sampled_result.criteria.psis_loo.elpd
    )


def test_load_roundtrips_posterior_inputs_and_diagnostics(
    sampled_result, tmp_path
) -> None:
    directory = sampled_result.save(tmp_path / "run")

    loaded = MCMCFitResult.load(directory)

    assert loaded.posterior.components == sampled_result.posterior.components
    for parameter in ("flux", "fwhm", "center", "delta_v_kms"):
        assert np.array_equal(
            loaded.posterior["line"][parameter].samples,
            sampled_result.posterior["line"][parameter].samples,
        )
    assert loaded.inputs.options == sampled_result.inputs.options
    assert loaded.inputs.workspace.ids == sampled_result.inputs.workspace.ids
    assert np.array_equal(
        loaded.inputs.workspace.spectrum.flux,
        sampled_result.inputs.workspace.spectrum.flux,
    )
    assert loaded.sampling.elapsed_seconds == sampled_result.sampling.elapsed_seconds
    assert loaded.sampling.diagnostics == sampled_result.sampling.diagnostics
    assert loaded.criteria.psis_loo.elpd == pytest.approx(
        sampled_result.criteria.psis_loo.elpd
    )
    assert np.array_equal(
        loaded.criteria.psis_loo.pointwise, sampled_result.criteria.psis_loo.pointwise
    )
    assert "log_likelihood" in loaded.idata.children


def test_save_refuses_then_allows_overwrite(sampled_result, tmp_path) -> None:
    directory = sampled_result.save(tmp_path / "run")

    with pytest.raises(FileExistsError, match="overwrite=True"):
        sampled_result.save(tmp_path / "run")

    assert sampled_result.save(tmp_path / "run", overwrite=True) == directory


def test_load_reports_missing_files(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="idata.nc"):
        MCMCFitResult.load(tmp_path / "absent")


def test_load_rejects_unknown_format_version(sampled_result, tmp_path) -> None:
    directory = sampled_result.save(tmp_path / "run")
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["format_version"] = 999
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="format version 999"):
        MCMCFitResult.load(directory)


@pytest.mark.integration
def test_joint_fit_roundtrips_frames_and_frame_loo(tmp_path) -> None:
    rest_um = 0.65646
    z = 5.228
    center = rest_um * (1.0 + z)

    def frame(seed: int, offset: float) -> NoobSpectrum:
        rng = np.random.default_rng(seed)
        wavelength = np.linspace(3.95, 4.17, 120)
        flux = (
            offset
            + gaussian(
                wavelength,
                center=center,
                flux=1.5e-3,
                fwhm_kms=350.0,
                resolving_power=1600.0,
            )
            + rng.normal(0.0, 8e-3, wavelength.size)
        )
        return NoobSpectrum(
            flux,
            np.full(wavelength.size, 8e-3),
            obs=wavelength,
            z=z,
            unit="um",
            resolving_power=1600.0,
        )

    joint = NoobSpectrumSet([frame(1, 0.05), frame(2, 0.09)])
    workspace = joint.prepare(
        [NoobLine("Ha", rest=rest_um, z=z, unit="um")], continuum_sharing="pooled"
    )
    result = workspace.model().sample(
        draws=40, tune=40, chains=2, cores=1, random_seed=11, progressbar=False
    )

    loaded = MCMCFitResult.load(result.save(tmp_path / "joint"))

    assert loaded.inputs.workspace.frame_ids == workspace.frame_ids
    assert np.array_equal(
        loaded.posterior["Ha"]["fwhm"].samples, result.posterior["Ha"]["fwhm"].samples
    )
    for key in result.posterior["continuum"].parameters:
        assert np.array_equal(
            loaded.posterior["continuum"][key].samples,
            result.posterior["continuum"][key].samples,
        )
    original_loo = result.frame_loo(max_draws=40)
    loaded_loo = loaded.frame_loo(max_draws=40)
    assert loaded_loo.frame_ids == original_loo.frame_ids
    assert loaded_loo.elpd == pytest.approx(original_loo.elpd)
