"""Tests for the inference.spectrum modelling layer (NoobLine / setup resolution).

These cover the load-bearing logic only: id generation, the root-floats /
derived-locks centre default, the FWHM auto-lock, flux ties, the per-axis
conflict guards, and the 2-D collapse. No real spectra are needed.
"""

import numpy as np
import pytest

from noobfriend.inference.spectrum import NoobLine, NoobSpectrum
from noobfriend.inference.spectrum._setup import DEFAULT_DV_BOUNDS


def _spec() -> NoobSpectrum:
    wl = np.linspace(6400.0, 6700.0, 50)
    return NoobSpectrum.from_1d(
        wl,
        np.ones_like(wl),
        np.full_like(wl, 0.1),
        z=0.0,
        wave_unit="A",
        R=2700.0,
    )


def _by_id(setup) -> dict[str, object]:
    return {c.id: c for c in setup.components}


def test_root_floats_and_derived_locks_co_moving():
    ha = NoobLine("Halpha", rest_wavelength=6562.8, unit="A", component="narrow")
    nii = ha.derive("NII_6583", rest_wavelength=6583.0)
    comps = _by_id(_spec().setup([ha, nii]))

    root = comps["Halpha"].centre
    assert root.kind == "free" and root.base_id is None
    assert root.bounds == DEFAULT_DV_BOUNDS

    child = comps["NII_6583"].centre
    assert child.kind == "fixed" and child.value == 0.0
    assert child.base_id == "Halpha" and "co-moving" in child.render()
    # same component -> width auto-locks; flux free by default.
    assert comps["NII_6583"].width.kind == "tied"
    assert (
        comps["NII_6583"].flux.kind == "free" and comps["NII_6583"].flux.base_id is None
    )


def test_id_generation_minimal_numbered_and_custom():
    ha_n = NoobLine("Halpha", rest_wavelength=6562.8, unit="A", component="narrow")
    ha_b = ha_n.derive(component="broad")
    abs1 = ha_n.derive(component="absorption")
    abs2 = ha_n.derive(component="absorption")  # second auto absorption -> numbered
    custom = ha_n.derive(component="broad", custom_id="Halpha_blue")
    nii = ha_n.derive("NII_6583", rest_wavelength=6583.0)

    comps = _by_id(_spec().setup([ha_n, ha_b, abs1, abs2, custom, nii]))
    # single-component line -> bare linename.
    assert "NII_6583" in comps
    # repeated (linename, component) -> 1-based numeric suffix.
    assert {"Halpha.absorption.1", "Halpha.absorption.2"} <= set(comps)
    # lone auto components stay unsuffixed; the custom_id line is excluded from
    # the count, so it does not bump its broad sibling to "Halpha.broad.1".
    assert {"Halpha.narrow", "Halpha.broad", "Halpha_blue"} <= set(comps)


def test_duplicate_custom_id_raises():
    ha = NoobLine(
        "Halpha",
        rest_wavelength=6562.8,
        unit="A",
        component="narrow",
        custom_id="dup",
    )
    other = ha.derive(component="broad", custom_id="dup")
    with pytest.raises(ValueError, match="duplicate component id"):
        _spec().setup([ha, other])


def test_derive_inherits_identity():
    ha = NoobLine("Halpha", rest_wavelength=6562.8, unit="A", component="narrow")
    child = ha.derive()  # everything inherited
    assert child.linename == "Halpha"
    assert child.rest_wavelength == 6562.8
    assert child.unit == "A"
    assert child.component == "narrow"
    assert child.parent is ha and not child.is_root


def test_centre_fixed_and_bounded_offsets():
    ha = NoobLine("Halpha", rest_wavelength=6562.8, unit="A", component="narrow")
    fixed = ha.derive(component="broad", delta_v_kms=120.0)
    bounded = ha.derive(component="broad", delta_v_kms=(-800.0, 0.0))
    comps = _by_id(_spec().setup([ha, fixed, bounded]))

    f = comps["Halpha.broad.1"].centre
    assert f.kind == "fixed" and f.value == 120.0 and f.base_id == "Halpha.narrow"
    b = comps["Halpha.broad.2"].centre
    assert (
        b.kind == "free" and b.bounds == (-800.0, 0.0) and b.base_id == "Halpha.narrow"
    )


def test_width_auto_free_lock_override_and_abs():
    ha = NoobLine("Halpha", rest_wavelength=6562.8, unit="A", component="narrow")
    # different component -> width free from the broad template.
    broad = ha.derive(component="broad")
    # same component but lock_fwhm=False -> forced free.
    sii = ha.derive("SII", rest_wavelength=6716.0, lock_fwhm=False)
    # outflow pattern: same component, absolute width override.
    outflow = ha.derive(
        delta_v_kms=(-600.0, 0.0), abs_fwhm=(300.0, 1000.0), custom_id="Halpha_outflow"
    )
    comps = _by_id(_spec().setup([ha, broad, sii, outflow]))

    assert comps["Halpha.broad"].width.kind == "free"
    assert comps["Halpha.broad"].width.bounds == (1000.0, 10000.0)
    assert comps["SII"].width.kind == "free"  # lock_fwhm=False broke the auto-lock
    assert comps["Halpha_outflow"].width.kind == "free"
    assert comps["Halpha_outflow"].width.bounds == (300.0, 1000.0)


def test_flux_ratio_tie_and_abs_override():
    nii_b = NoobLine("NII_6583", rest_wavelength=6583.0, unit="A", component="narrow")
    nii_a = nii_b.derive("NII_6548", rest_wavelength=6548.0, flux_ratio=1 / 3)
    pinned = nii_b.derive("OIII", rest_wavelength=5007.0, abs_flux=2.5)
    comps = _by_id(_spec().setup([nii_b, nii_a, pinned]))

    tie = comps["NII_6548"].flux
    assert tie.kind == "fixed" and tie.base_id == "NII_6583" and tie.relation == "ratio"
    assert abs(tie.value - 1 / 3) < 1e-12
    assert comps["OIII"].flux.kind == "fixed" and comps["OIII"].flux.base_id is None


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"delta_v_kms": 1.0, "delta_wavelength": 1.0}, "not both"),
        ({"lock_fwhm": True, "abs_fwhm": 500.0}, "conflicts with abs_fwhm"),
        ({"flux_ratio": 0.5, "abs_flux": 1.0}, "not both"),
    ],
)
def test_per_axis_conflicts_raise(kwargs, match):
    parent = NoobLine("Halpha", rest_wavelength=6562.8, unit="A", component="narrow")
    with pytest.raises(ValueError, match=match):
        parent.derive(component="broad", **kwargs)


def test_root_only_options_raise():
    with pytest.raises(ValueError, match="lock_fwhm=True needs a parent"):
        NoobLine(
            "X",
            rest_wavelength=5000.0,
            unit="A",
            component="narrow",
            lock_fwhm=True,
        )
    with pytest.raises(ValueError, match="flux_ratio needs a parent"):
        NoobLine(
            "X",
            rest_wavelength=5000.0,
            unit="A",
            component="narrow",
            flux_ratio=0.5,
        )


def test_setup_rejects_absent_parent():
    ha = NoobLine("Halpha", rest_wavelength=6562.8, unit="A", component="narrow")
    orphan = ha.derive("NII_6583", rest_wavelength=6583.0)
    with pytest.raises(ValueError, match="not in the lines list"):
        _spec().setup([orphan])  # parent ha omitted


def test_cross_unit_conversion_to_spectrum_frame():
    # micron spectrum, Angstrom line -> rest converts into the spectrum's frame.
    wl_um = np.linspace(0.64, 0.67, 50)
    spec = NoobSpectrum.from_1d(
        wl_um, np.ones_like(wl_um), np.full_like(wl_um, 0.1), z=0.0, wave_unit="um"
    )
    ha = NoobLine("Halpha", rest_wavelength=6562.8, unit="A", component="narrow")
    comp = spec.setup([ha]).components[0]
    assert comp.rest_wavelength == pytest.approx(0.65628)  # 6562.8 A -> 0.65628 um


def test_delta_wavelength_converts_to_spectrum_frame():
    # a +-10 A centre wander on a micron spectrum resolves to +-0.001 um bounds.
    wl_um = np.linspace(0.64, 0.67, 50)
    spec = NoobSpectrum.from_1d(
        wl_um, np.ones_like(wl_um), np.full_like(wl_um, 0.1), z=0.0, wave_unit="um"
    )
    ha = NoobLine("Halpha", rest_wavelength=6562.8, unit="A", component="narrow")
    broad = ha.derive(component="broad", delta_wavelength=(-10.0, 10.0))
    axis = {c.id: c for c in spec.setup([ha, broad]).components}["Halpha.broad"].centre
    assert axis.unit == "um"
    assert axis.bounds == pytest.approx((-0.001, 0.001))


def test_invalid_wave_unit_rejected():
    with pytest.raises(ValueError, match="must be one of"):
        NoobLine("X", rest_wavelength=5000.0, unit="Angstrom", component="narrow")
    with pytest.raises(ValueError, match="must be one of"):
        NoobSpectrum.from_1d(
            np.arange(5.0), np.ones(5), np.ones(5), z=0.0, wave_unit="nanometer"
        )


def test_from_2d_collapse_sums_flux_and_propagates_error():
    wl = np.linspace(6400.0, 6700.0, 20)
    flux2d = np.ones((5, 20))  # 5 cross-dispersion rows, dispersion along axis 1
    err2d = np.full((5, 20), 0.2)
    spec = NoobSpectrum.from_2d(
        wl,
        flux2d,
        err2d,
        collapse_window=(1, 4),
        dispersion="row",
        z=0.0,
        wave_unit="A",
        boost=2.0,
    )
    assert np.allclose(spec.flux, 3.0)  # 3 rows summed
    assert np.allclose(spec.error, np.sqrt(3 * 0.2**2) * 2.0)  # quadrature * boost


def test_from_1d_shape_mismatch_raises():
    with pytest.raises(ValueError, match="equal length"):
        NoobSpectrum.from_1d(
            np.arange(5.0), np.arange(4.0), np.arange(5.0), z=0.0, wave_unit="A"
        )


def test_run_recovers_synthetic_halpha_nii():
    """End-to-end: the PyMC fit recovers a known Hα + tied [NII] doublet.

    The one integration test that exercises the model build, sampler, and
    result. Skipped when the optional ``mcmc`` extra is absent.
    """
    pytest.importorskip("pymc")

    c = 299792.458
    fw2sig = 1.0 / 2.3548200450309493
    rng = np.random.default_rng(0)
    wl = np.linspace(6500.0, 6620.0, 240)

    def gauss(center: float, fwhm_kms: float, flux: float) -> np.ndarray:
        sw = center * (fwhm_kms / c) * fw2sig
        return (
            flux / (sw * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((wl - center) / sw) ** 2)
        )

    truth = (
        1.0
        + gauss(6562.8, 300.0, 20.0)
        + gauss(6583.4, 300.0, 9.0)
        + gauss(6548.0, 300.0, 3.0)
    )
    err = np.full_like(wl, 0.08)
    spec = NoobSpectrum.from_1d(
        wl,
        truth + rng.normal(0, 0.08, wl.size),
        err,
        z=0.0,
        wave_unit="A",
        R=3000.0,
    )

    ha = NoobLine("Halpha", rest_wavelength=6562.8, unit="A", component="narrow")
    nii_b = ha.derive("NII_6583", rest_wavelength=6583.4)
    nii_a = nii_b.derive("NII_6548", rest_wavelength=6548.0, flux_ratio=1 / 3)

    res = spec.setup([ha, nii_b, nii_a]).run(
        draws=300, tune=300, chains=2, random_seed=1, progressbar=False
    )

    assert res.flux("Halpha")[1] == pytest.approx(20.0, rel=0.15)
    assert res.flux("NII_6583")[1] == pytest.approx(9.0, rel=0.2)
    assert res.fwhm_kms("Halpha")[1] == pytest.approx(300.0, rel=0.2)
    # the locked doublet ratio holds exactly in every draw.
    assert res.flux("NII_6548")[1] == pytest.approx(
        res.flux("NII_6583")[1] / 3, rel=1e-6
    )

    diag = res.diagnostics()
    assert diag["divergences"] >= 0 and np.isfinite(diag["max_rhat"])
    low, mid, high = res.model_curve(wl)
    assert low.shape == mid.shape == high.shape == wl.shape
    assert np.all(low <= high)

    # pathology reports: the strong lines are warranted; bounds carry the columns.
    sig = res.significance_report()
    assert bool(sig.loc["Halpha", "warranted"])
    assert sig.loc["Halpha", "snr"] > 5.0
    bnd = res.boundary_report()
    assert {"flagged", "frac_at_lower", "frac_at_upper"} <= set(bnd.columns)

    import matplotlib.pyplot as plt

    assert len(res.plot().axes) == 2  # data + residual panels
    assert len(res.plot(residual=False, decompose=False).axes) == 1
    plt.close("all")


def test_setup_plot_preview_returns_figure():
    """setup.plot() previews window + initial guess without needing PyMC."""
    import matplotlib.pyplot as plt

    ha = NoobLine("Halpha", rest_wavelength=6562.8, unit="A", component="narrow")
    nii = ha.derive("NII_6583", rest_wavelength=6583.0)
    fig = _spec().setup([ha, nii]).plot()
    assert len(fig.axes) == 1
    plt.close(fig)
