"""Tests for spectrum model contracts."""

from __future__ import annotations

import numpy as np
import pytest

from noobfriend.inference.spectrum import NoobLine, NoobSpectrum, NoobSpectrumModel


def _workspace():
    obs = np.linspace(39000.0, 41000.0, 100)
    spectrum = NoobSpectrum(np.ones_like(obs), np.full_like(obs, 0.1), obs=obs, z=7.0)
    base = NoobLine("OIII", rest=5008.24, z=7.0)
    broad = base.derive(component="broad", profile="lorentzian")
    return spectrum.prepare([base, broad], continuum_order=2, continuum_lambda_0=40000.0)


def test_workspace_builds_model_contract() -> None:
    workspace = _workspace()

    model = workspace.model()

    assert isinstance(model, NoobSpectrumModel)
    assert model.workspace is workspace
    assert model.spectrum is workspace.spectrum
    assert model.handles == workspace.handles
    assert model.continuum is workspace.continuum
    assert model.component_names == ("narrow", "broad")
    assert model.contribution_names == ("emission",)
    assert model.profile_names == ("gaussian", "lorentzian")


def test_model_builds_continuum_design_matrix() -> None:
    model = _workspace().model()

    design = model.continuum_design(np.array([39999.0, 40000.0, 40002.0]))

    assert design.flags.writeable is False
    assert model.continuum_parameter_names == ("c", "k1", "k2")
    assert design.tolist() == [
        [1.0, -1.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 2.0, 4.0],
    ]


def test_model_uses_valid_pixels_for_default_continuum_design() -> None:
    obs = np.array([39999.0, 40000.0, 40001.0])
    spectrum = NoobSpectrum(
        [1.0, np.nan, 1.0],
        [0.1, 0.1, 0.0],
        obs=obs,
        mask_excluded=[False, False, True],
    )
    line = NoobLine(obs=40000.0)
    model = spectrum.prepare([line], continuum_lambda_0=40000.0).model()

    assert model.continuum_design().tolist() == [[1.0, -1.0]]


def test_model_fit_backend_is_explicitly_unimplemented() -> None:
    model = _workspace().model()

    with pytest.raises(NotImplementedError, match="not implemented"):
        model.fit()
