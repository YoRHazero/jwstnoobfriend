"""Tests for NoobLine declarations."""

from __future__ import annotations

import pytest

from noobfriend.inference.spectrum.line import LineKernel, LineRule, NoobLine
from noobfriend.inference.spectrum.priors import LogUniform, Normal, Uniform


def halpha(**kwargs: object) -> NoobLine:
    return NoobLine("Ha", rest=6564.6, unit="angstrom", **kwargs)  # type: ignore[arg-type]


class TestDeclaration:
    """Constructor fields and validation (T4a)."""

    @pytest.mark.parametrize("name", ["", "Ha.broad", "continuum", "frame"])
    def test_rejects_bad_names(self, name: str) -> None:
        with pytest.raises(ValueError):
            NoobLine(name, rest=6564.6, unit="angstrom")

    def test_needs_exactly_one_wavelength(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            NoobLine("Ha", rest=6564.6, obs=6600.0, unit="angstrom")

    def test_constructor_takes_declaration_fields_only(self) -> None:
        with pytest.raises(TypeError):
            NoobLine("Hb", rest=4862.7, unit="angstrom", base=halpha())  # type: ignore[call-arg]

    def test_interval_shorthand_is_gone(self) -> None:
        with pytest.raises(TypeError, match="prior"):
            halpha().fwhm((100.0, 300.0))  # type: ignore[arg-type]

    def test_width_rejects_unbounded_normal(self) -> None:
        with pytest.raises(ValueError, match="TruncatedNormal"):
            halpha().fwhm(Normal(300.0, 50.0))

    def test_defaults_are_unresolved(self) -> None:
        line = halpha()
        assert (
            line.center_rule == line.fwhm_rule == line.flux_rule == LineRule("default")
        )


class TestDerive:
    """Relations along the derive tree only (T4b)."""

    def test_same_component_defaults(self) -> None:
        nii = halpha().derive("NII", rest=6585.3)
        assert nii.base is not None and nii.base.name == "Ha"
        assert nii.center_rule == LineRule("delta_v_from_base_kms", 0.0)
        assert nii.fwhm_rule == LineRule("ratio_to_base", 1.0)
        assert nii.flux_rule == LineRule("default")

    def test_other_component_gets_independent_width(self) -> None:
        broad = halpha().derive("Ha_broad", component="broad")
        assert broad.fwhm_rule == LineRule("default")
        assert broad.rest == 6564.6

    def test_relative_rules_need_a_base(self) -> None:
        with pytest.raises(ValueError, match="no base"):
            halpha().flux(ratio_to_base=0.3)
        with pytest.raises(ValueError, match="no base"):
            halpha().center(delta_v_from_base_kms=0.0)

    def test_ordering_prior_on_velocity_difference(self) -> None:
        blue = (
            halpha().derive("Ha_blue").center(delta_v_from_base_kms=Uniform(high=0.0))
        )
        assert blue.center_rule.kind == "delta_v_from_base_kms"
        assert blue.center_rule.value == Uniform(high=0.0)

    def test_unit_without_wavelength_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unit"):
            halpha().derive("x", unit="nm")


class TestKernels:
    """convolve and kernel inheritance (T4c)."""

    def test_same_component_locks_inherited_kernel(self) -> None:
        base = halpha().convolve(
            "laplace", LogUniform(100.0, 2000.0), fraction=Uniform()
        )
        derived = base.derive("NII", rest=6585.3)
        assert derived.kernels == (LineKernel("laplace", None, None),)

    def test_other_component_copies_effective_declaration(self) -> None:
        base = halpha().convolve("laplace", 900.0, fraction=0.5)
        chained = base.derive("NII", rest=6585.3).derive("NII_broad", component="broad")
        assert chained.kernels == (LineKernel("laplace", 900.0, 0.5),)

    def test_convolve_same_kind_replaces(self) -> None:
        base = halpha().convolve("laplace", 900.0)
        derived = base.derive("NII", rest=6585.3).convolve("laplace", 500.0)
        assert derived.kernels == (LineKernel("laplace", 500.0, 1.0),)

    def test_kernel_needs_gaussian_base(self) -> None:
        with pytest.raises(NotImplementedError):
            halpha(profile="lorentzian").convolve("gaussian", 300.0)

    def test_kernel_kind_is_a_string(self) -> None:
        with pytest.raises(TypeError):
            halpha().convolve(len, 300.0)  # type: ignore[arg-type]

    def test_fraction_domain(self) -> None:
        with pytest.raises(ValueError):
            halpha().convolve("laplace", 300.0, fraction=1.5)
