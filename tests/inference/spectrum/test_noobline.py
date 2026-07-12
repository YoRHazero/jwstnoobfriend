"""Tests for the spectrum line contract."""

from __future__ import annotations

import pytest

from noobfriend.inference.spectrum import DEFAULT_FWHM_RANGES, NoobLine


def test_line_converts_rest_wavelength_to_observed_frame() -> None:
    line = NoobLine("OIII5007", rest=5008.24, z=7.0)

    assert line.observed_wavelength == pytest.approx(40065.92)
    assert line.rest == 5008.24
    assert line.z == 7.0
    assert line.component == "narrow"
    assert line.contribution == "emission"
    assert line.center_rule.is_bounded
    assert line.center_rule.bounds == (-300.0, 300.0)
    assert line.center_rule.offset_unit == "km/s"
    assert line.fwhm_rule.is_bounded
    assert line.fwhm_rule.bounds == DEFAULT_FWHM_RANGES["narrow"]
    assert line.flux_rule.is_free


def test_line_accepts_observed_wavelength_without_redshift() -> None:
    line = NoobLine(obs=40065.92)

    assert line.observed_wavelength == 40065.92
    assert line.linename is None
    assert line.z is None


def test_line_rejects_invalid_wavelength_contracts() -> None:
    with pytest.raises(ValueError, match="z must be strictly positive"):
        NoobLine("OIII5007", rest=5008.24, z=0.0)

    with pytest.raises(ValueError, match="z is required"):
        NoobLine("OIII5007", rest=5008.24)

    with pytest.raises(ValueError, match="inconsistent"):
        NoobLine("OIII5007", rest=5008.24, obs=40000.0, z=7.0)


def test_line_validates_component_and_profile() -> None:
    with pytest.raises(ValueError, match="Unsupported component"):
        NoobLine("OIII5007", obs=1.0, component="core")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unsupported component"):
        NoobLine("OIII5007", obs=1.0, component="outflow")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unsupported component"):
        NoobLine("OIII5007", obs=1.0, component="absorption")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unsupported contribution"):
        NoobLine("OIII5007", obs=1.0, contribution="negative")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unsupported profile"):
        NoobLine("OIII5007", obs=1.0, profile="voigt")  # type: ignore[arg-type]


def test_derive_creates_strongly_bound_line() -> None:
    base = NoobLine("OIII5007", rest=5008.24, z=7.0)
    derived = base.derive("OIII4959", rest=4960.30).flux(ratio=0.335)

    assert derived.base is base
    assert derived.z == base.z
    assert derived.unit == base.unit
    assert derived.observed_wavelength == pytest.approx(39682.4)
    assert derived.center_rule.is_locked
    assert derived.center_rule.target is base
    assert derived.fwhm_rule.is_locked
    assert derived.fwhm_rule.target is base
    assert derived.flux_rule.is_ratio
    assert derived.flux_rule.value == 0.335


def test_derive_can_change_component_contribution_and_profile_at_creation() -> None:
    narrow = NoobLine("Ha", rest=6564.61, z=2.0)
    broad = narrow.derive(
        component="broad", contribution="absorption", profile="lorentzian"
    )

    assert broad.linename == "Ha"
    assert broad.component == "broad"
    assert broad.contribution == "absorption"
    assert broad.profile == "lorentzian"
    assert broad.base is narrow
    assert broad.observed_wavelength == narrow.observed_wavelength


def test_default_fwhm_range_tracks_component() -> None:
    narrow = NoobLine("Ha", obs=1.0)
    broad = NoobLine("Ha", obs=1.0, component="broad")

    assert narrow.fwhm_rule.is_bounded
    assert narrow.fwhm_rule.bounds == (10.0, 700.0)
    assert broad.fwhm_rule.is_bounded
    assert broad.fwhm_rule.bounds == (800.0, 5000.0)


def test_center_sets_fixed_or_bounded_offsets() -> None:
    line = NoobLine("OIII5007", obs=40065.92)

    fixed = line.center(delta_v_kms=25.0)
    bounded = line.center(delta_wavelength=(-2.0, 2.0))

    assert fixed.center_rule.is_fixed
    assert fixed.center_rule.value == 25.0
    assert fixed.center_rule.offset_unit == "km/s"
    assert bounded.center_rule.is_bounded
    assert bounded.center_rule.bounds == (-2.0, 2.0)
    assert bounded.center_rule.offset_unit == "wavelength"

    with pytest.raises(ValueError, match="exactly one"):
        line.center()
    with pytest.raises(ValueError, match="exactly one"):
        line.center(delta_v_kms=0.0, delta_wavelength=0.0)


def test_fwhm_override_and_lock_rules() -> None:
    base = NoobLine("OIII5007", obs=40065.92)
    derived = base.derive("OIII4959", obs=39682.4)

    fixed = base.fwhm(override=180.0)
    bounded = base.fwhm(override=(80.0, 400.0))
    locked_to_base = derived.fwhm(locked=True)
    unlocked = derived.fwhm(locked=False)

    assert fixed.fwhm_rule.is_fixed
    assert fixed.fwhm_rule.value == 180.0
    assert bounded.fwhm_rule.is_bounded
    assert bounded.fwhm_rule.bounds == (80.0, 400.0)
    assert locked_to_base.fwhm_rule.is_locked
    assert locked_to_base.fwhm_rule.target is base
    assert unlocked.fwhm_rule.is_bounded
    assert unlocked.fwhm_rule.bounds == DEFAULT_FWHM_RANGES[unlocked.component]

    with pytest.raises(ValueError, match="positive"):
        base.fwhm(override=0.0)
    with pytest.raises(ValueError, match="requires a derived line"):
        base.fwhm(locked=True)


def test_flux_override_and_ratio_rules() -> None:
    base = NoobLine("OIII5007", obs=40065.92)
    derived = base.derive("OIII4959", obs=39682.4)

    fixed = base.flux(override=1.2e-17)
    bounded = base.flux(override=(0.0, 5.0e-17))
    ratio = derived.flux(ratio=0.335)

    assert fixed.flux_rule.is_fixed
    assert fixed.flux_rule.value == 1.2e-17
    assert bounded.flux_rule.is_bounded
    assert bounded.flux_rule.bounds == (0.0, 5.0e-17)
    assert ratio.flux_rule.is_ratio
    assert ratio.flux_rule.value == 0.335
    assert ratio.flux_rule.bounds is None

    with pytest.raises(ValueError, match="nonnegative"):
        base.flux(override=-1.0)
    with pytest.raises(ValueError, match="requires a derived line"):
        base.flux(ratio=1.0)
    with pytest.raises(TypeError, match="finite number"):
        derived.flux(ratio=(0.32, 0.35))  # type: ignore[arg-type]
