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


def test_continuum_boost_recovers_underquoted_error():
    from noobfriend.core.specutils import continuum_boost

    rng = np.random.default_rng(0)
    wl = np.linspace(6500.0, 6620.0, 300)
    flux = 1.0 + rng.normal(0.0, 0.2, wl.size)  # real scatter 0.2 ...
    error = np.full(wl.size, 0.1)  # ... but error quoted as 0.1
    assert continuum_boost(wl, flux, error) == pytest.approx(2.0, rel=0.15)


def test_from_2d_background_inflates_error_and_guard():
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(1)
    corr = gaussian_filter(rng.normal(0.0, 1.0, (40, 60)), sigma=1.0)  # correlated
    err2d = np.full((40, 60), float(corr.std()))
    wl = np.linspace(6500.0, 6620.0, 60)
    common = {
        "collapse_window": (18, 23),
        "dispersion": "row",
        "z": 0.0,
        "wave_unit": "A",
    }

    naive = NoobSpectrum.from_2d(wl, corr, err2d, **common)
    bg = NoobSpectrum.from_2d(
        wl, corr, err2d, correlation_source="background", **common
    )
    assert np.median(bg.error) > np.median(naive.error)  # correlation inflates it

    with pytest.raises(ValueError, match="only used with"):
        NoobSpectrum.from_2d(
            wl,
            corr,
            err2d,
            correlation_source="continuum",
            correlation_correction=1.3,
            **common,
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
        correlation_source="manual",
        correlation_correction=2.0,
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

    # the predictive band adds observation noise: wider than the model band at
    # matching percentiles, reproducible under a fixed seed, and containing most
    # of the data when evaluated at the fitted pixels (exact sigma, grid=None).
    m_lo, _, m_hi = res.model_curve(wl, q=(2.5, 50.0, 97.5))
    p_lo, p_mid, p_hi = res.predictive_curve(wl, q=(2.5, 50.0, 97.5), seed=0)
    assert p_lo.shape == p_mid.shape == p_hi.shape == wl.shape
    assert np.all(p_lo <= p_hi)
    assert np.mean(p_hi - p_lo) > np.mean(m_hi - m_lo)
    assert np.array_equal(
        p_lo, res.predictive_curve(wl, q=(2.5, 50.0, 97.5), seed=0)[0]
    )
    cov_lo, _, cov_hi = res.predictive_curve(seed=0)
    inside = (res.window_flux >= cov_lo) & (res.window_flux <= cov_hi)
    assert inside.mean() >= 0.85
    # past the fitted edges sigma is undefined -> band is NaN there, finite within.
    w_lo, w_hi = float(res.window_wl.min()), float(res.window_wl.max())
    ext = np.linspace(w_lo - 50.0, w_hi + 50.0, 64)
    e_lo, _, e_hi = res.predictive_curve(ext, seed=0)
    outside = (ext < w_lo) | (ext > w_hi)
    assert np.all(np.isnan(e_lo[outside])) and np.all(np.isnan(e_hi[outside]))
    assert np.all(np.isfinite(e_lo[~outside])) and np.all(np.isfinite(e_hi[~outside]))

    # pathology reports: the strong lines are warranted; bounds carry the columns.
    sig = res.significance_report()
    assert bool(sig.loc["Halpha", "warranted"])
    assert sig.loc["Halpha", "snr"] > 5.0
    bnd = res.boundary_report()
    assert {"flagged", "frac_at_lower", "frac_at_upper"} <= set(bnd.columns)

    import matplotlib.pyplot as plt

    assert len(res.plot().axes) == 2  # data + residual panels
    assert len(res.plot(residual=False, decompose=False).axes) == 1
    assert len(res.plot(predictive=True).axes) == 2  # predictive band overlay
    plt.close("all")


def test_setup_plot_preview_returns_figure():
    """setup.plot() previews window + initial guess without needing PyMC."""
    import matplotlib.pyplot as plt

    ha = NoobLine("Halpha", rest_wavelength=6562.8, unit="A", component="narrow")
    nii = ha.derive("NII_6583", rest_wavelength=6583.0)
    fig = _spec().setup([ha, nii]).plot()
    assert len(fig.axes) == 1
    plt.close(fig)


def test_select_window_clamps_bounds_to_valid_pixels():
    """Returned bounds clamp to real data, never past the spectrum's coverage.

    The pre-fit preview draws its initial-guess curve over these bounds, so they
    must track the first/last pixel that actually exists -- not the nominal
    velocity-padded request, which can overrun the data or land on invalid edges.
    """
    from noobfriend.inference.spectrum._window import select_window

    wl = np.linspace(6500.0, 6620.0, 240)
    ha = NoobLine("Halpha", rest_wavelength=6562.8, unit="A", component="narrow")
    comps = (
        NoobSpectrum.from_1d(
            wl, np.ones_like(wl), np.full_like(wl, 0.1), z=0.0, wave_unit="A"
        )
        .setup([ha])
        .components
    )

    # a huge pad makes the nominal window spill past both data edges.
    spec = NoobSpectrum.from_1d(
        wl, np.ones_like(wl), np.full_like(wl, 0.1), z=0.0, wave_unit="A"
    )
    mask, (lo, hi) = select_window(spec, comps, pad_kms=50000.0)
    assert lo == pytest.approx(float(wl[mask].min()))
    assert hi == pytest.approx(float(wl[mask].max()))
    assert lo >= wl.min() and hi <= wl.max()

    # invalid edge pixels (non-finite error) are excluded -> bounds move inward.
    err = np.full_like(wl, 0.1)
    err[:5] = np.nan
    err[-3:] = np.nan
    spec2 = NoobSpectrum.from_1d(wl, np.ones_like(wl), err, z=0.0, wave_unit="A")
    _, (lo2, hi2) = select_window(spec2, comps, pad_kms=50000.0)
    assert lo2 == pytest.approx(wl[5])
    assert hi2 == pytest.approx(wl[-4])


def test_mask_excluded_threads_and_with_mask():
    """Load-time mask_excluded and fit-time with_mask both drop window pixels.

    with_mask is immutable (returns a copy), replace-semantic, and layers over
    the spectrum's data-quality mask by OR.
    """
    from noobfriend.inference.spectrum._window import select_window

    wl = np.linspace(6500.0, 6620.0, 240)
    dq = np.zeros(wl.size, dtype=bool)
    dq[10] = True  # a load-time bad pixel
    spec = NoobSpectrum.from_1d(
        wl,
        np.ones_like(wl),
        np.full_like(wl, 0.1),
        z=0.0,
        wave_unit="A",
        mask_excluded=dq,
    )
    ha = NoobLine("Halpha", rest_wavelength=6562.8, unit="A", component="narrow")
    setup = spec.setup([ha])

    # DQ is honored by default, and select_window drops it.
    assert setup.mask_excluded is not None and setup.mask_excluded[10]
    mask, _ = select_window(spec, setup.components, exclude=setup.mask_excluded)
    assert not mask[10]

    # intervals -> those pixels excluded on top of DQ; immutable (setup unchanged).
    masked = setup.with_mask([[6550.0, 6555.0]])
    assert masked is not setup
    region = (wl >= 6550.0) & (wl <= 6555.0)
    assert masked.mask_excluded[region].all() and masked.mask_excluded[10]
    assert not setup.mask_excluded[region].any()  # original untouched

    # bool-array form.
    arr = np.zeros(wl.size, dtype=bool)
    arr[100:105] = True
    assert setup.with_mask(arr).mask_excluded[100:105].all()

    # None resets the fit-time layer back to DQ only.
    cleared = masked.with_mask(None)
    assert cleared.mask_excluded[10] and not cleared.mask_excluded[region].any()

    # wrong-length bool array errors; auto tuning without 'auto' errors.
    with pytest.raises(ValueError, match="length"):
        setup.with_mask(np.zeros(5, dtype=bool))
    with pytest.raises(ValueError, match="auto"):
        setup.with_mask([[6550.0, 6555.0]], tau=2.0)


def test_compute_auto_mask_flags_outliers():
    """auto-mask flags a cosmic ray and an unrelated line, sparing real signal."""
    from noobfriend.inference.spectrum._mask import compute_auto_mask

    c = 299792.458
    fw2sig = 1.0 / 2.3548200450309493
    rng = np.random.default_rng(0)
    wl = np.linspace(6500.0, 6620.0, 240)

    def gauss(center: float, fwhm_kms: float, flux: float) -> np.ndarray:
        sw = center * (fwhm_kms / c) * fw2sig
        return (
            flux / (sw * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((wl - center) / sw) ** 2)
        )

    flux = 1.0 + gauss(6562.8, 300.0, 20.0)  # target Halpha (modelled)
    flux += gauss(6600.0, 400.0, 6.0)  # unrelated resolved line (not modelled)
    flux = flux + rng.normal(0, 0.05, wl.size)
    cr = int(np.argmin(np.abs(wl - 6515.0)))  # an isolated cosmic ray
    flux[cr] += 2.0

    spec = NoobSpectrum.from_1d(
        wl, flux, np.full_like(wl, 0.05), z=0.0, wave_unit="A", R=3000.0
    )
    setup = spec.setup(
        [NoobLine("Halpha", rest_wavelength=6562.8, unit="A", component="narrow")]
    )

    mask = compute_auto_mask(setup)
    assert mask[cr]  # cosmic ray
    assert mask[(wl >= 6597.0) & (wl <= 6603.0)].sum() >= 3  # unrelated line
    assert not mask[(wl >= 6560.0) & (wl <= 6566.0)].any()  # Halpha core spared
    assert not mask[int(np.argmin(np.abs(wl - 6540.0)))]  # clean continuum spared


# --- observed-frame (blind-search) lines -----------------------------------


def _zless_spec() -> NoobSpectrum:
    wl = np.linspace(6500.0, 6800.0, 60)
    return NoobSpectrum.from_1d(
        wl, np.ones_like(wl), np.full_like(wl, 0.1), wave_unit="A", R=3000.0
    )


def test_observed_line_auto_names_and_centre_without_z():
    # An unnamed observed line gets a centre-derived id, has no rest wavelength,
    # and places its centre directly (no redshift) -- all on a z=None spectrum.
    line = NoobLine.observed(6712.3, unit="A", component="narrow")
    assert line.linename is None and line.custom_id == "Line_obs_6712_narrow"
    assert line.frame == "observed" and line.center == 6712.3

    comp = _zless_spec().setup([line]).components[0]
    assert comp.id == "Line_obs_6712_narrow"
    assert comp.rest_wavelength is None
    assert comp.centre_wavelength == pytest.approx(6712.3)
    # free centre by default (blind-search wants the centre to float).
    assert comp.centre.kind == "free" and comp.centre.base_id is None


def test_observed_line_converts_centre_into_spectrum_frame():
    # micron spectrum, Angstrom observed centre -> centre converts into the frame.
    wl_um = np.linspace(0.64, 0.70, 60)
    spec = NoobSpectrum.from_1d(
        wl_um, np.ones_like(wl_um), np.full_like(wl_um, 0.1), wave_unit="um"
    )
    line = NoobLine.observed(6712.3, unit="A", component="narrow")
    comp = spec.setup([line]).components[0]
    assert comp.centre_wavelength == pytest.approx(0.67123)  # 6712.3 A -> 0.67123 um


def test_observed_lines_tie_like_rest_lines():
    # Two observed lines bind exactly like two rest lines: a derived companion
    # co-moves in velocity (centre tied to the parent) and ties its flux ratio.
    a = NoobLine.observed(6712.0, unit="A", component="narrow", linename="featA")
    b = a.derive(linename="featB", observed_center=6731.0, flux_ratio=(0.3, 3.0))
    comps = _by_id(_zless_spec().setup([a, b]))

    assert comps["featA"].centre_wavelength == pytest.approx(6712.0)
    assert comps["featB"].centre_wavelength == pytest.approx(6731.0)
    # shared velocity tie (co-moving) -- the thing a rest doublet relies on.
    child = comps["featB"].centre
    assert child.kind == "fixed" and child.value == 0.0 and child.base_id == "featA"
    flux = comps["featB"].flux
    assert flux.base_id == "featA" and flux.relation == "ratio" and flux.kind == "free"


def test_rest_line_requires_z():
    spec = _zless_spec()
    ha = NoobLine("Halpha", rest_wavelength=6562.8, unit="A", component="narrow")
    with pytest.raises(ValueError, match="needs the spectrum's redshift"):
        spec.setup([ha])


def test_derive_cannot_cross_frames():
    rest = NoobLine("Halpha", rest_wavelength=6562.8, unit="A", component="narrow")
    obs = NoobLine.observed(6560.0, unit="A", component="narrow")
    with pytest.raises(ValueError, match="across frames"):
        NoobLine(
            "Y",
            frame="observed",
            observed_center=5000.0,
            unit="A",
            component="narrow",
            parent=rest,
        )
    with pytest.raises(ValueError, match="across frames"):
        NoobLine(
            "Z",
            rest_wavelength=5000.0,
            unit="A",
            component="narrow",
            parent=obs,
        )


def test_observed_line_field_validation():
    with pytest.raises(ValueError, match="observed_center must be a positive"):
        NoobLine.observed(-5.0, unit="A", component="narrow")
    with pytest.raises(ValueError, match="rest_wavelength is only for"):
        NoobLine(
            "X",
            frame="observed",
            observed_center=5000.0,
            rest_wavelength=5000.0,
            unit="A",
            component="narrow",
        )
    with pytest.raises(ValueError, match="observed_center is only for"):
        NoobLine(
            "X",
            rest_wavelength=5000.0,
            observed_center=5000.0,
            unit="A",
            component="narrow",
        )


# --- manual initvals (start-only) ------------------------------------------


def test_resolve_initvals_inverts_physical_to_free_rvs():
    from noobfriend.inference.spectrum._initvals import normalize_init, resolve_initvals

    ha = NoobLine("Halpha", rest_wavelength=6562.8, unit="A", component="narrow")
    broad = ha.derive(component="broad")
    comps = _spec().setup([ha, broad]).components  # R=2700
    ids = {c.template.name: c.id for c in comps}
    init = {
        "continuum": {"c0": 1.5, "c1": -0.001},
        ids["narrow"]: {"flux": 20.0, "fwhm_kms": 250.0, "dv_kms": 30.0},
        ids["broad"]: {"flux": 5.0, "fwhm_kms": 2500.0},
    }
    norm = normalize_init(init, [c.id for c in comps])
    out = resolve_initvals(
        comps, norm, fwhm_inst=299792.458 / 2700.0, continuum_degree=1
    )

    # the free RVs *are* the physical quantities -> identity (within clipping).
    assert out["continuum__c0"] == 1.5 and out["continuum__c1"] == -0.001
    assert out[f"{ids['narrow']}__flux_raw"] == 20.0
    assert out[f"{ids['narrow']}__fwhm_free"] == pytest.approx(250.0)
    assert out[f"{ids['narrow']}__dv_off"] == pytest.approx(30.0)
    assert out[f"{ids['broad']}__flux_raw"] == 5.0
    assert out[f"{ids['broad']}__fwhm_free"] == pytest.approx(2500.0)


def test_resolve_initvals_clips_to_lsf_floored_bounds():
    from noobfriend.inference.spectrum._initvals import normalize_init, resolve_initvals

    ha = NoobLine("Halpha", rest_wavelength=6562.8, unit="A", component="narrow")
    comps = _spec().setup([ha]).components  # R=2700 -> LSF FWHM ~111 km/s
    norm = normalize_init({"Halpha": {"fwhm_kms": 5.0}}, [c.id for c in comps])
    out = resolve_initvals(
        comps, norm, fwhm_inst=299792.458 / 2700.0, continuum_degree=1
    )
    # 5 km/s is below the instrumental floor -> clipped up to it, not used raw.
    assert 111.0 < out["Halpha__fwhm_free"] < 112.0


def test_resolve_initvals_skips_non_free_axes_and_degree0_c1():
    from noobfriend.inference.spectrum._initvals import normalize_init, resolve_initvals

    ha = NoobLine("Halpha", rest_wavelength=6562.8, unit="A", component="narrow")
    nii = ha.derive("NII_6583", rest_wavelength=6583.0)  # width tied, centre fixed
    comps = _spec().setup([ha, nii]).components
    ids = [c.id for c in comps]

    with pytest.warns(UserWarning):
        out = resolve_initvals(
            comps,
            normalize_init({"NII_6583": {"fwhm_kms": 300.0, "dv_kms": 10.0}}, ids),
            fwhm_inst=None,
            continuum_degree=1,
        )
    assert "NII_6583__fwhm_free" not in out and "NII_6583__dv_off" not in out

    with pytest.warns(UserWarning, match="degree 0"):
        out2 = resolve_initvals(
            comps,
            normalize_init({"continuum": {"c0": 1.0, "c1": 0.5}}, ids),
            fwhm_inst=None,
            continuum_degree=0,
        )
    assert out2 == {"continuum__c0": 1.0}


def test_normalize_init_rejects_unknown_keys():
    from noobfriend.inference.spectrum._initvals import normalize_init

    ha = NoobLine("Halpha", rest_wavelength=6562.8, unit="A", component="narrow")
    ids = [c.id for c in _spec().setup([ha]).components]
    with pytest.raises(ValueError, match="unknown component id"):
        normalize_init({"Hbeta": {"flux": 1.0}}, ids)
    with pytest.raises(ValueError, match="unknown quantity"):
        normalize_init({"Halpha": {"width": 1.0}}, ids)


def test_run_recovers_observed_free_centre_gaussian():
    """End-to-end: an observed-frame fit recovers a synthetic free-centre line.

    The blind-search path -- no redshift, no line identity -- recovers the
    observed centre, width, and flux of a lone Gaussian. Skipped without ``mcmc``.
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

    center_true, fwhm_true, flux_true = 6562.0, 350.0, 15.0
    truth = 1.0 + gauss(center_true, fwhm_true, flux_true)
    err = np.full_like(wl, 0.08)
    spec = NoobSpectrum.from_1d(  # no z -- a pure observed-frame fit
        wl, truth + rng.normal(0, 0.08, wl.size), err, wave_unit="A", R=3000.0
    )

    line = NoobLine.observed(
        6560.0, unit="A", component="narrow", delta_wavelength=(-15.0, 15.0)
    )
    setup = spec.setup([line])
    cid = setup.components[0].id
    res = setup.run(draws=300, tune=300, chains=2, random_seed=1, progressbar=False)

    assert res.center(cid)[1] == pytest.approx(center_true, abs=2.0)
    assert res.fwhm_kms(cid)[1] == pytest.approx(fwhm_true, rel=0.25)
    assert res.flux(cid)[1] == pytest.approx(flux_true, rel=0.2)


def test_initvals_improve_convergence_under_short_warmup():
    """A good manual start mixes far better than a bad one under short warmup.

    Same model, data, and seed -- only the NUTS start differs (via ``init_guess``,
    a start-only knob that leaves the priors untouched). With deliberately short
    tuning, a start at the truth converges (R-hat near 1) while a start far in the
    tails is left badly mixed: the good start has not just travelled less but
    landed on the answer. Skipped without ``mcmc``.
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
        wl, truth + rng.normal(0, 0.08, wl.size), err, z=0.0, wave_unit="A", R=3000.0
    )
    ha = NoobLine("Halpha", rest_wavelength=6562.8, unit="A", component="narrow")
    nii_b = ha.derive("NII_6583", rest_wavelength=6583.4)
    nii_a = nii_b.derive("NII_6548", rest_wavelength=6548.0, flux_ratio=1 / 3)
    setup = spec.setup([ha, nii_b, nii_a])

    # NII width/velocity are tied to Halpha (not free), so only its flux is seeded.
    good = {
        "Halpha": {"flux": 20.0, "fwhm_kms": 300.0, "dv_kms": 0.0},
        "NII_6583": {"flux": 9.0},
        "continuum": {"c0": 1.0, "c1": 0.0},
    }
    bad = {
        "Halpha": {"flux": 0.5, "fwhm_kms": 950.0, "dv_kms": 250.0},
        "NII_6583": {"flux": 0.5},
    }
    common = {
        "draws": 200,
        "tune": 80,
        "chains": 2,
        "random_seed": 0,
        "progressbar": False,
    }
    diag_bad = setup.run(init_guess=bad, **common).diagnostics()
    res_good = setup.run(init_guess=good, **common)
    diag_good = res_good.diagnostics()

    # The good start has converged; the bad start, on the same budget, has not.
    assert diag_good["max_rhat"] < 1.2
    assert diag_bad["max_rhat"] > diag_good["max_rhat"] + 0.3
    # ... and the good start lands on the truth.
    assert res_good.flux("Halpha")[1] == pytest.approx(20.0, rel=0.2)
    assert res_good.flux("NII_6583")[1] == pytest.approx(9.0, rel=0.3)


# --- per-fit narrow/broad FWHM discriminator --------------------------------


def test_apply_narrow_broad_bound_moves_template_default_only():
    from noobfriend.inference.spectrum._setup import apply_narrow_broad_bound
    from noobfriend.inference.spectrum._template import get_template

    ha = NoobLine("Halpha", rest_wavelength=6562.8, unit="A", component="narrow")
    broad = ha.derive(component="broad")
    pinned = ha.derive(component="broad", abs_fwhm=(1500.0, 4000.0), custom_id="pinned")
    absb = ha.derive(component="absorption")
    comps = _spec().setup([ha, broad, pinned, absb]).components

    # None -> unchanged (the same object, no copy).
    assert apply_narrow_broad_bound(comps, None) is comps

    out = {c.id: c for c in apply_narrow_broad_bound(comps, 600.0)}
    assert out["Halpha.narrow"].width.bounds == (30.0, 600.0)  # narrow upper capped
    assert out["Halpha.broad"].width.bounds == (600.0, 10000.0)  # broad lower raised
    assert out["pinned"].width.bounds == (1500.0, 4000.0)  # explicit abs_fwhm kept
    assert out["Halpha.absorption"].width.bounds == (50.0, 3000.0)  # absorption left

    # the global registry is never mutated.
    assert get_template("narrow").fwhm_kms_bounds == (30.0, 1000.0)
    assert get_template("broad").fwhm_kms_bounds == (1000.0, 10000.0)

    # non-positive and bound-collapsing values raise.
    with pytest.raises(ValueError, match="must be positive"):
        apply_narrow_broad_bound(comps, 0.0)
    with pytest.raises(ValueError, match="collapses"):
        apply_narrow_broad_bound(comps, 20.0)  # below the narrow floor (30)


def test_narrow_broad_kms_threads_into_model_fwhm_bounds():
    """The per-fit discriminator moves the narrow/broad fwhm prior support.

    Checks the model's recorded truncation bounds (which the posterior support
    nests inside) rather than sampling, so it is fast and deterministic. Skipped
    without ``mcmc``.
    """
    pytest.importorskip("pymc")

    from noobfriend.inference.spectrum._pymc_model import build_model

    wl = np.linspace(6500.0, 6620.0, 240)
    spec = NoobSpectrum.from_1d(
        wl, np.ones_like(wl), np.full_like(wl, 0.1), z=0.0, wave_unit="A", R=3000.0
    )
    ha = NoobLine("Halpha", rest_wavelength=6562.8, unit="A", component="narrow")
    broad = ha.derive(component="broad")
    setup = spec.setup([ha, broad])

    def fwhm_bounds(narrow_broad_kms):
        bundle = build_model(setup, narrow_broad_kms=narrow_broad_kms)
        bp = {(b.component_id, b.quantity): b for b in bundle.bounded}
        return bp[("Halpha.narrow", "fwhm")], bp[("Halpha.broad", "fwhm")]

    narrow6, broad6 = fwhm_bounds(600.0)
    assert narrow6.upper == pytest.approx(600.0)  # narrow upper = the discriminator
    assert broad6.lower == pytest.approx(600.0)  # broad lower = the discriminator

    narrow4, broad4 = fwhm_bounds(400.0)
    assert narrow4.upper == pytest.approx(400.0)  # support tracks the value
    assert broad4.lower == pytest.approx(400.0)

    narrow_d, broad_d = fwhm_bounds(None)  # default -> template bounds
    assert narrow_d.upper == pytest.approx(1000.0)
    assert broad_d.lower == pytest.approx(1000.0)
