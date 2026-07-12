"""Tests for preparing one line combination against a spectrum."""

from __future__ import annotations

import numpy as np
import pytest

from noobfriend.inference.spectrum import NoobLine, NoobSpectrum


def _spectrum() -> NoobSpectrum:
    obs = np.linspace(39000.0, 41000.0, 100)
    return NoobSpectrum(np.ones_like(obs), np.full_like(obs, 0.1), obs=obs, z=7.0)


def test_prepare_sequence_assigns_auto_ids_and_handles() -> None:
    base = NoobLine("OIII", rest=5008.24, z=7.0)
    broad = base.derive(component="broad")
    lorentz = base.derive(component="broad", profile="lorentzian")
    anonymous = NoobLine(obs=40100.0)

    workspace = _spectrum().prepare([base, broad, lorentz, anonymous])

    assert workspace.id_mode == "auto"
    assert workspace.ids == (
        "OIII.narrow",
        "OIII.broad.gaussian",
        "OIII.broad.lorentzian",
        "Line_obs_40100",
    )
    assert workspace.handle_for(base).line is base
    assert workspace.roots[0].line is base
    assert len(workspace.derived) == 2
    assert {
        (handle.component, handle.contribution, handle.profile)
        for handle in workspace.handles
    } == {
        ("narrow", "emission", "gaussian"),
        ("broad", "emission", "gaussian"),
        ("broad", "emission", "lorentzian"),
    }


def test_prepare_mapping_uses_manual_ids() -> None:
    base = NoobLine("OIII", rest=5008.24, z=7.0)
    broad = base.derive(component="broad")

    workspace = _spectrum().prepare({"core": base, "wide": broad})

    assert workspace.id_mode == "manual"
    assert workspace.ids == ("core", "wide")
    assert workspace.handles[1].observed_wavelength == pytest.approx(
        broad.observed_wavelength
    )


def test_prepare_reserves_continuum_component_id() -> None:
    line = NoobLine("line", rest=5008.24, z=7.0)
    auto = NoobLine("continuum", rest=5008.24, z=7.0)

    with pytest.raises(ValueError, match="reserved"):
        _spectrum().prepare({"continuum": line})
    with pytest.raises(ValueError, match="reserved"):
        _spectrum().prepare([auto])


def test_prepare_auto_ids_use_contribution_only_when_needed() -> None:
    emission = NoobLine("Ha", rest=6564.61, z=5.1)
    absorption = NoobLine("Ha", rest=6564.61, z=5.1, contribution="absorption")

    workspace = NoobSpectrum(
        np.ones(100),
        np.full(100, 0.1),
        obs=np.linspace(39000.0, 41000.0, 100),
    ).prepare([emission, absorption])

    assert workspace.ids == (
        "Ha.narrow.gaussian.emission",
        "Ha.narrow.gaussian.absorption",
    )
    assert workspace.handles[1].contribution == "absorption"


def test_prepare_resolves_continuum_options() -> None:
    base = NoobLine("OIII", rest=5008.24, z=7.0)

    default = _spectrum().prepare([base])
    explicit = _spectrum().prepare(
        [base], continuum_order=2, continuum_lambda_0=40000.0
    )

    assert default.continuum.order == 1
    assert default.continuum.lambda_0 == pytest.approx(base.observed_wavelength)
    assert default.continuum.parameter_names == ("c", "k1")
    assert explicit.continuum.order == 2
    assert explicit.continuum.lambda_0 == 40000.0
    assert explicit.continuum.parameter_names == ("c", "k1", "k2")


def test_workspace_summary_renders_prepared_plan_as_html() -> None:
    base = NoobLine("O&III", rest=5008.24, z=7.0).center(delta_v_kms=(-100.0, 100.0))
    broad = base.derive(component="broad").flux(ratio=0.335)

    html = (
        _spectrum()
        .prepare({"core": base, "wide": broad}, continuum_lambda_0=40000.0)
        .summary()
    )

    assert html.startswith("<section")
    assert "NoobFitWorkspace" in html
    assert "O&amp;III" in html
    assert "<code>core</code>" in html
    assert "c + k1 * (lambda - lambda_0)" in html
    assert "lambda_0 = 40000" in html
    assert "<th>Contribution</th>" in html
    assert "<td>emission</td>" in html
    assert "Center locks share velocity offset" in html
    assert "bounded delta_v_kms=[-100, 100]" in html
    assert "bounded [10, 700] km/s" in html
    assert "locked to core" in html
    assert "ratio 0.335 to core" in html


def test_prepare_validates_continuum_options() -> None:
    base = NoobLine("OIII", rest=5008.24, z=7.0)

    with pytest.raises(ValueError, match="nonnegative integer"):
        _spectrum().prepare([base], continuum_order=-1)

    with pytest.raises(ValueError, match="finite"):
        _spectrum().prepare([base], continuum_lambda_0=float("nan"))


def test_prepare_validates_input_mode_and_graph() -> None:
    base = NoobLine("OIII", rest=5008.24, z=7.0)
    broad = base.derive(component="broad")

    with pytest.raises(ValueError, match="cannot appear twice"):
        _spectrum().prepare([base, base])

    with pytest.raises(ValueError, match="base is not"):
        _spectrum().prepare([broad])

    with pytest.raises(ValueError, match="non-empty"):
        _spectrum().prepare({"": base})

    with pytest.raises(TypeError, match="NoobLine"):
        _spectrum().prepare([("core", base)])  # type: ignore[list-item]


def test_prepare_validates_spectrum_line_consistency() -> None:
    outside = NoobLine("outside", obs=50000.0)
    wrong_z = NoobLine("wrong", rest=5008.24, z=6.0)

    with pytest.raises(ValueError, match="outside"):
        _spectrum().prepare([outside])

    with pytest.raises(ValueError, match="inconsistent"):
        _spectrum().prepare([wrong_z])
