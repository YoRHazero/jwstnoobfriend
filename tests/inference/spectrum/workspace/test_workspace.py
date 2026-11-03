"""Tests for prepare and the fit workspace."""

from __future__ import annotations

import numpy as np
import pytest

from noobfriend.inference.spectrum._wavelength import C_KMS
from noobfriend.inference.spectrum._xp import numpy_namespace
from noobfriend.inference.spectrum.continuum import NoobContinuum
from noobfriend.inference.spectrum.data import NoobSpectrum
from noobfriend.inference.spectrum.line import NoobLine
from noobfriend.inference.spectrum.noise import AR1, Noise
from noobfriend.inference.spectrum.priors import (
    NONNEGATIVE,
    POSITIVE,
    JointNormal,
    LogUniform,
    Normal,
    PriorSource,
    TruncatedNormal,
    Uniform,
)
from noobfriend.inference.spectrum.workspace import NoobFitWorkspace

from .._helpers import HALPHA, Z, halpha_spectrum

XP = numpy_namespace()
CONTINUUM = NoobContinuum.polynomial(1, lambda_0=HALPHA * (1 + Z))


def ha() -> NoobLine:
    return NoobLine("Ha", rest=HALPHA, unit="angstrom")


def joined(**noise: Noise) -> NoobSpectrum:
    spectrum = NoobSpectrum.join(
        {"v1": halpha_spectrum(seed=1), "v2": halpha_spectrum(seed=2)}
    )
    return spectrum.with_noise(noise) if noise else spectrum


class TestDefaults:
    """Defaults resolved to explicit priors (T4b.2)."""

    def test_line_defaults(self) -> None:
        ws = halpha_spectrum().prepare([ha()], continuum=CONTINUUM)
        table = ws.parameters
        assert table["Ha.delta_v"].prior == Uniform(-300.0, 300.0)
        assert table["Ha.fwhm"].prior == LogUniform(10.0, 700.0)
        flux = table["Ha.flux"].prior
        assert isinstance(flux, TruncatedNormal)
        assert (flux.mu, flux.low, flux.high) == (0.0, 0.0, np.inf)

    def test_continuum_defaults_scale_with_data(self) -> None:
        ws = halpha_spectrum().prepare([ha()], continuum=CONTINUUM)
        scales = ws.scales
        c1 = ws.parameters["continuum.c1"].prior
        assert c1 == Normal(0.0, 5.0 * scales.flux / scales.wavelength)

    def test_broad_component_width(self) -> None:
        line = NoobLine("Hb", rest=HALPHA, unit="angstrom", component="broad")
        ws = halpha_spectrum().prepare([line], continuum=CONTINUUM)
        assert ws.parameters["Hb.fwhm"].prior == LogUniform(800.0, 5000.0)


class TestRelations:
    """Derive-tree bindings and physical quantities (T4b.4)."""

    def test_physical_quantities_follow_the_base(self) -> None:
        nii = ha().derive("NII", rest=6585.3).flux(ratio_to_base=Uniform(0.0, 1.0))
        ws = halpha_spectrum().prepare([ha(), nii], continuum=CONTINUUM)
        values = {row.name: 0.0 for row in ws.parameters.free()}
        values |= {"Ha.delta_v": 30.0, "Ha.fwhm": 250.0, "Ha.flux": 10.0}
        values |= {"NII.flux_ratio_to_base": 0.3}
        physical = ws.physical(values)
        assert physical["NII.fwhm"] == 250.0
        assert physical["NII.flux"] == pytest.approx(3.0)
        nominal = ws["NII"].nominal
        assert physical["NII.center"] == pytest.approx(nominal * (1 + 30.0 / C_KMS))
        assert "NII.fwhm_ratio_to_base" in ws.parameters
        assert ws.parameters["NII.fwhm_ratio_to_base"].kind == "fixed"

    def test_base_may_follow_derived_line(self) -> None:
        nii = ha().derive("NII", rest=6585.3)
        ws = halpha_spectrum().prepare([nii, ha()], continuum=CONTINUUM)
        assert [line.name for line in ws.lines] == ["NII", "Ha"]

    def test_delta_wavelength_converts_units(self) -> None:
        line = NoobLine("Ha", rest=HALPHA / 10.0, unit="nm").center(
            delta_wavelength=Uniform(-0.5, 0.5)
        )
        ws = halpha_spectrum().prepare([line], continuum=CONTINUUM)
        assert ws["Ha"].nominal == pytest.approx(HALPHA * (1 + Z))
        assert ws.parameters["Ha.delta_wavelength"].prior == Uniform(-5.0, 5.0)

    def test_locked_kernel_reads_base(self) -> None:
        base = ha().convolve("laplace", LogUniform(100.0, 3000.0), fraction=0.5)
        nii = base.derive("NII", rest=6585.3)
        ws = halpha_spectrum().prepare([base, nii], continuum=CONTINUUM)
        assert "NII.laplace.fwhm" not in ws.parameters
        quantity = ws.parameters.quantities["NII.laplace.fwhm"]
        assert quantity.expression.text() == "Ha.laplace.fwhm"


class TestValidation:
    """Errors raised by prepare."""

    def test_rest_line_needs_z(self) -> None:
        spectrum = halpha_spectrum()
        frame = spectrum.frames["frame_0"]
        no_z = NoobSpectrum(
            frame.obs, frame.flux, frame.error, unit="angstrom", resolving_power=1600.0
        )
        with pytest.raises(ValueError, match="no z"):
            no_z.prepare([ha()], continuum=CONTINUUM)

    def test_line_outside_window(self) -> None:
        center = HALPHA * (1 + Z)
        with pytest.raises(ValueError, match="outside every fit window"):
            halpha_spectrum().prepare(
                [ha()], continuum=CONTINUUM, window=(center + 20, center + 80)
            )

    def test_duplicate_names(self) -> None:
        with pytest.raises(ValueError, match="twice"):
            halpha_spectrum().prepare([ha(), ha()], continuum=CONTINUUM)

    def test_missing_base(self) -> None:
        with pytest.raises(ValueError, match="not among"):
            halpha_spectrum().prepare(
                [ha().derive("NII", rest=6585.3)], continuum=CONTINUUM
            )

    def test_lines_must_be_a_sequence(self) -> None:
        with pytest.raises(TypeError, match="sequence"):
            halpha_spectrum().prepare({"Ha": ha()}, continuum=CONTINUUM)  # type: ignore[arg-type]

    def test_continuum_is_required(self) -> None:
        with pytest.raises(TypeError):
            halpha_spectrum().prepare([ha()])  # type: ignore[call-arg]

    def test_continuum_only_fit(self) -> None:
        ws = halpha_spectrum().prepare([], continuum=CONTINUUM)
        assert [row.name for row in ws.parameters.free()] == [
            "continuum.c0",
            "continuum.c1",
        ]


class TestFrames:
    """Per-frame continuum sharing and noise rows."""

    def test_pooled_rows(self) -> None:
        ws = joined().prepare([ha()], continuum=CONTINUUM)
        assert ws.parameters["continuum.c0.spread"].is_free
        pooled = ws.parameters["frame.v2.continuum.c1"]
        assert (pooled.kind, pooled.pooled_from) == ("pooled", "continuum.c1")
        assert ws.continuum.frame_parameters["v1"] == (
            "frame.v1.continuum.c0",
            "frame.v1.continuum.c1",
        )

    def test_independent_and_fixed_coefficients(self) -> None:
        continuum = NoobContinuum.polynomial(
            1, lambda_0=HALPHA * (1 + Z), sharing="independent"
        ).coefficient(1, 0.0)
        ws = joined().prepare([ha()], continuum=continuum)
        assert ws.parameters["frame.v1.continuum.c0"].is_free
        assert ws.parameters["continuum.c1"].kind == "fixed"
        assert ws.continuum.frame_parameters["v2"] == (
            "frame.v2.continuum.c0",
            "continuum.c1",
        )

    def test_noise_rows(self) -> None:
        ws = joined(
            v2=Noise(scale=Uniform(0.5, 2.0), correlation=AR1(Uniform()))
        ).prepare([ha()], continuum=CONTINUUM)
        assert ws.parameters["frame.v1.noise.scale"].value == 1.0
        assert ws.parameters["frame.v2.noise.rho"].prior == Uniform(-1.0, 1.0)
        assert ws.frame_noise["v2"].kind == "ar1"


class TestModel:
    """Single xp implementation of the mean model."""

    def test_total_is_continuum_plus_observed_components(self) -> None:
        ws = halpha_spectrum().prepare(
            [ha().convolve("laplace", 600.0, fraction=0.4)], continuum=CONTINUUM
        )
        values = {row.name: 0.0 for row in ws.parameters.free()}
        values |= {"Ha.fwhm": 300.0, "Ha.flux": 40.0, "continuum.c0": 1.0}
        physical = ws.physical(values)
        observed = ws.frame_model(XP, physical, "frame_0")
        emergent = ws.frame_model(XP, physical, "frame_0", view="emergent")
        np.testing.assert_allclose(
            observed.total, observed.continuum + observed.components["Ha"]
        )
        np.testing.assert_allclose(emergent.total, observed.total)
        assert emergent.components["Ha"].max() > observed.components["Ha"].max()

    def test_draw_axis(self) -> None:
        ws = halpha_spectrum().prepare([ha()], continuum=CONTINUUM)
        values = {row.name: np.zeros((3, 1)) for row in ws.parameters.free()}
        values["Ha.fwhm"] = np.full((3, 1), 300.0)
        values["Ha.flux"] = np.array([[10.0], [20.0], [30.0]])
        total = ws.frame_model(XP, ws.physical(values), "frame_0").total
        assert total.shape == (3, int(ws.fitted["frame_0"].sum()))
        np.testing.assert_allclose(total[2], 3 * total[0])


class TestFingerprints:
    """Data and model fingerprints change independently (T11c.3)."""

    def test_data_and_model_parts(self) -> None:
        base = halpha_spectrum(seed=3).prepare([ha()], continuum=CONTINUUM)
        same = halpha_spectrum(seed=3).prepare([ha()], continuum=CONTINUUM)
        other_data = halpha_spectrum(seed=4).prepare([ha()], continuum=CONTINUUM)
        other_model = halpha_spectrum(seed=3).prepare(
            [ha().fwhm(250.0)], continuum=CONTINUUM
        )
        assert (same.data_fingerprint, same.model_fingerprint) == (
            base.data_fingerprint,
            base.model_fingerprint,
        )
        assert other_data.data_fingerprint != base.data_fingerprint
        assert (
            other_data.model_fingerprint != base.model_fingerprint
        )  # data-derived flux prior
        assert other_model.data_fingerprint == base.data_fingerprint
        assert other_model.model_fingerprint != base.model_fingerprint


class TestJointPriors:
    """Collecting joint priors (T7.3) and the data-reuse check (T7.6)."""

    def joint(self, sources: tuple[PriorSource, ...] = ()) -> JointNormal:
        return JointNormal(
            names=("Ha.fwhm", "Ha.flux", "NII.flux_ratio_to_base"),
            domains=(POSITIVE, NONNEGATIVE, NONNEGATIVE),
            mean=np.array([5.5, 3.0, -1.0]),
            covariance=np.diag([0.1, 0.2, 0.3]),
            sources=sources,
        )

    def test_members_gathered_and_rest_marginalized(self) -> None:
        joint = self.joint()
        line = ha().fwhm(joint.member("Ha.fwhm")).flux(joint.member("Ha.flux"))
        ws = halpha_spectrum().prepare([line], continuum=CONTINUUM)
        (block,) = ws.joint_priors
        assert block.parameters == ("Ha.fwhm", "Ha.flux")
        assert block.marginalized == ("NII.flux_ratio_to_base",)
        np.testing.assert_allclose(block.covariance, np.diag([0.1, 0.2]))
        assert "marginalized out: NII.flux_ratio_to_base" in ws.summary()

    def test_override_leaves_the_joint_prior(self) -> None:
        joint = self.joint()
        line = (
            ha().fwhm(joint.member("Ha.fwhm")).flux(joint.member("Ha.flux")).fwhm(300.0)
        )
        ws = halpha_spectrum().prepare([line], continuum=CONTINUUM)
        assert ws.joint_priors[0].parameters == ("Ha.flux",)

    def test_refuses_reused_pixels(self) -> None:
        spectrum = halpha_spectrum(seed=5)
        first = spectrum.prepare([ha()], continuum=CONTINUUM)
        source = PriorSource(first.frame_digests["frame_0"], first.fitted["frame_0"])
        line = ha().fwhm(self.joint((source,)).member("Ha.fwhm"))
        with pytest.raises(ValueError, match="same pixels"):
            spectrum.prepare([line], continuum=CONTINUUM)


def test_handles_and_summary() -> None:
    ws = halpha_spectrum().prepare([ha()], continuum=CONTINUUM)
    assert isinstance(ws, NoobFitWorkspace)
    assert [handle.name for handle in ws] == ["Ha", "continuum"]
    assert ws["Ha"].parameters == ("Ha.delta_v", "Ha.fwhm", "Ha.flux")
    with pytest.raises(KeyError):
        ws["Hb"]
    text = ws.summary()
    for section in (
        "Frames",
        "Components",
        "Parameters",
        "Ha.flux",
        "frame.frame_0.noise.scale",
    ):
        assert section in text
