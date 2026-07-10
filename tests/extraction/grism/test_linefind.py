"""Unit tests for the grism line-finder band-pass SNR core.

These exercise pure array logic on synthetic frames (a Gaussian "line" on a
smooth continuum); they do not touch ``example_data/`` or any real FITS file.
"""

import numpy as np
import pytest

from noobfriend.extraction.grism.linefind import GrismLineFinder
from noobfriend.extraction.grism.linefind._combine import _combine_stack, _union_bbox
from noobfriend.extraction.grism.linefind._detect import detect
from noobfriend.extraction.grism.linefind._filter import BandPass, band_pass_snr
from noobfriend.extraction.grism.linefind._linefind import _velocity_to_sigma


def _gaussian_blob(
    shape: tuple[int, int], y: float, x: float, sigma: float, amplitude: float
) -> np.ndarray:
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    r2 = (yy - y) ** 2 + (xx - x) ** 2
    return amplitude * np.exp(-r2 / (2 * sigma**2))


def _streak(shape: tuple[int, int], y0: float, y_sigma: float) -> np.ndarray:
    """Build a smooth continuum: linear ramp along x, Gaussian profile in y."""
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    profile = np.exp(-((yy - y0) ** 2) / (2 * y_sigma**2))
    ramp = 5.0 + 0.05 * xx
    return profile * ramp


def test_line_on_blank_is_a_high_snr_peak() -> None:
    shape = (81, 81)
    data = _gaussian_blob(shape, 40, 40, sigma=1.0, amplitude=20.0)
    error = np.ones(shape)
    bp = band_pass_snr(
        data,
        error,
        dispersion_axis=1,
        line_scales=[(1.0, 1.0)],
        continuum_sigma_disp=4.0,
    )
    snr = bp.snr[0]
    assert np.unravel_index(np.nanargmax(snr), snr.shape) == (40, 40)
    assert snr[40, 40] > 5.0
    assert abs(snr[5, 5]) < 3.0


def test_smooth_continuum_is_suppressed() -> None:
    shape = (81, 81)
    data = _streak(shape, y0=40, y_sigma=3.0)
    error = np.ones(shape)
    bp = band_pass_snr(
        data,
        error,
        dispersion_axis=1,
        line_scales=[(1.0, 1.0)],
        continuum_sigma_disp=4.0,
    )
    interior = bp.snr[0][20:-20, 20:-20]
    assert np.nanmax(np.abs(interior)) < 3.0


def test_line_on_continuum_still_detected() -> None:
    shape = (81, 81)
    data = _streak(shape, y0=40, y_sigma=3.0) + _gaussian_blob(
        shape, 40, 40, sigma=1.0, amplitude=20.0
    )
    error = np.ones(shape)
    bp = band_pass_snr(
        data,
        error,
        dispersion_axis=1,
        line_scales=[(1.0, 1.0)],
        continuum_sigma_disp=4.0,
    )
    snr = bp.snr[0]
    assert snr[40, 40] > 5.0
    # a continuum-only spot on the same rows, far in x from the line, stays low
    assert abs(snr[40, 12]) < 3.0


def test_snr_scales_inversely_with_error() -> None:
    shape = (81, 81)
    data = _gaussian_blob(shape, 40, 40, sigma=1.0, amplitude=20.0)
    bp1 = band_pass_snr(
        data,
        np.ones(shape),
        dispersion_axis=1,
        line_scales=[(1.0, 1.0)],
        continuum_sigma_disp=4.0,
    )
    bp2 = band_pass_snr(
        data,
        2.0 * np.ones(shape),
        dispersion_axis=1,
        line_scales=[(1.0, 1.0)],
        continuum_sigma_disp=4.0,
    )
    ratio = bp2.snr[0][40, 40] / bp1.snr[0][40, 40]
    assert np.isclose(ratio, 0.5, atol=1e-3)


def test_flat_image_has_near_zero_signal() -> None:
    shape = (81, 81)
    data = np.full(shape, 7.0)
    error = np.ones(shape)
    bp = band_pass_snr(
        data,
        error,
        dispersion_axis=1,
        line_scales=[(1.0, 1.0)],
        continuum_sigma_disp=4.0,
    )
    interior = bp.signal[0][20:-20, 20:-20]
    assert np.nanmax(np.abs(interior)) < 1e-6


def test_masked_pixels_do_not_crash_and_line_survives() -> None:
    shape = (81, 81)
    data = _gaussian_blob(shape, 40, 40, sigma=1.0, amplitude=20.0)
    error = np.ones(shape)
    data[10, 10] = np.nan  # DQ-masked data
    error[60, 60] = 0.0  # invalid error -> treated as missing
    bp = band_pass_snr(
        data,
        error,
        dispersion_axis=1,
        line_scales=[(1.0, 1.0)],
        continuum_sigma_disp=4.0,
    )
    snr = bp.snr[0]
    assert np.isfinite(snr[40, 40])
    assert snr[40, 40] > 5.0


def test_row_and_column_dispersion_are_transposes() -> None:
    shape = (81, 81)
    data = _gaussian_blob(shape, 40, 40, sigma=1.0, amplitude=20.0)
    error = np.ones(shape)
    row = band_pass_snr(
        data,
        error,
        dispersion_axis=1,
        line_scales=[(1.0, 1.0)],
        continuum_sigma_disp=4.0,
    )
    col = band_pass_snr(
        data.T,
        error.T,
        dispersion_axis=0,
        line_scales=[(1.0, 1.0)],
        continuum_sigma_disp=4.0,
    )
    assert np.allclose(row.snr[0], col.snr[0].T, equal_nan=True, atol=1e-6)


def test_continuum_sigma_disp_must_exceed_line_sigma_disp() -> None:
    shape = (41, 41)
    data = np.zeros(shape)
    error = np.ones(shape)
    with pytest.raises(ValueError, match="continuum_sigma_disp"):
        band_pass_snr(
            data,
            error,
            dispersion_axis=1,
            line_scales=[(4.0, 4.0)],
            continuum_sigma_disp=2.0,
        )


def test_low_coverage_borders_are_masked_not_spurious_peaks() -> None:
    rng = np.random.default_rng(0)
    shape = (81, 81)
    data = rng.standard_normal(shape) + _gaussian_blob(
        shape, 40, 40, sigma=1.5, amplitude=20.0
    )
    error = np.ones(shape)
    bp = band_pass_snr(
        data,
        error,
        dispersion_axis=1,
        line_scales=[(1.5, 1.5)],
        continuum_sigma_disp=5.0,
    )
    snr = bp.snr[0]
    # the border ring is masked (low kernel coverage -> NaN), not a huge peak;
    # this is the regression for the spurious edge SNR seen on a real frame.
    assert np.all(np.isnan(snr[0, :]))
    assert np.all(np.isnan(snr[-1, :]))
    assert np.all(np.isnan(snr[:, 0]))
    assert np.all(np.isnan(snr[:, -1]))
    # the global maximum over the whole frame is the injected line, not an edge
    assert np.unravel_index(np.nanargmax(snr), snr.shape) == (40, 40)
    assert np.nanmax(np.abs(snr)) < 500.0


def test_band_pass_snr_anisotropic_matches_broad_line() -> None:
    shape = (81, 81)
    # broad along dispersion (x), narrow across it (y)
    data = _bar(shape, 40, 40, sigma_disp=2.5, sigma_cross=1.0, amplitude=20.0)
    error = np.ones(shape)
    bp = band_pass_snr(
        data,
        error,
        dispersion_axis=1,
        line_scales=[(1.0, 2.5)],
        continuum_sigma_disp=6.0,
    )
    snr = bp.snr[0]
    assert np.unravel_index(np.nanargmax(snr), shape) == (40, 40)
    assert snr[40, 40] > 5.0


def _bandpass(snr2d: np.ndarray, scale: float = 1.5) -> BandPass:
    snr = snr2d[np.newaxis].astype(float)
    return BandPass(snr=snr, signal=snr.copy(), scales=((scale, scale),))


def _bar(
    shape: tuple[int, int],
    y: float,
    x: float,
    sigma_disp: float,
    sigma_cross: float,
    amplitude: float,
) -> np.ndarray:
    """Build an anisotropic Gaussian elongated along the dispersion axis (x)."""
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return amplitude * np.exp(
        -((xx - x) ** 2) / (2 * sigma_disp**2) - ((yy - y) ** 2) / (2 * sigma_cross**2)
    )


def test_detect_two_separated_lines() -> None:
    shape = (81, 81)
    snr = _gaussian_blob(shape, 20, 20, 1.5, 10.0) + _gaussian_blob(
        shape, 60, 60, 1.5, 8.0
    )
    cands = detect(_bandpass(snr), dispersion_axis=1, threshold=5.0)
    assert len(cands) == 2
    assert {(c.y, c.x) for c in cands} == {(20, 20), (60, 60)}
    assert cands[0].snr >= cands[1].snr  # sorted descending


def test_detect_faint_below_threshold_is_empty() -> None:
    shape = (81, 81)
    snr = _gaussian_blob(shape, 40, 40, 1.5, 3.0)
    assert detect(_bandpass(snr), dispersion_axis=1, threshold=5.0) == []


def test_detect_compactness_distinguishes_line_from_streak() -> None:
    shape = (81, 81)
    snr = _gaussian_blob(shape, 20, 40, 1.5, 10.0) + _bar(
        shape, 60, 40, sigma_disp=6.0, sigma_cross=1.5, amplitude=10.0
    )
    cands = detect(_bandpass(snr), dispersion_axis=1, threshold=5.0, grow_threshold=2.0)
    line = next(c for c in cands if abs(c.y - 20) <= 2)
    streak = next(c for c in cands if abs(c.y - 60) <= 2)
    # high recall: both are found; compactness is a feature that separates them
    assert streak.disp_extent > line.disp_extent
    assert streak.disp_extent > streak.cross_extent


def test_detect_deblends_two_close_peaks() -> None:
    shape = (81, 81)
    snr = _gaussian_blob(shape, 40, 39, 1.5, 10.0) + _gaussian_blob(
        shape, 40, 43, 1.5, 9.0
    )
    cands = detect(
        _bandpass(snr),
        dispersion_axis=1,
        threshold=5.0,
        grow_threshold=2.0,
        min_distance=3,
    )
    assert len(cands) == 2


def test_union_bbox_floor_ceil() -> None:
    x_off, y_off, (h, w) = _union_bbox(
        np.array([0.0, 99.4, -3.2, 50.0]), np.array([2.0, -1.1, 200.9, 10.0])
    )
    assert (x_off, y_off) == (-4, -2)
    assert w == 100 - (-4) + 1  # ceil(99.4) = 100
    assert h == 201 - (-2) + 1  # ceil(200.9) = 201


def test_combine_stack_median_rejects_cosmic_ray() -> None:
    images = np.full((4, 5, 5), 5.0)
    images[2, 2, 2] = 500.0  # a cosmic ray in one frame
    data, error, count = _combine_stack(images, np.ones((4, 5, 5)))
    assert np.all(count == 4)
    assert np.isclose(data[2, 2], 5.0)  # median ignores the outlier
    assert np.isclose(error[2, 2], 1.2533 / 2, atol=1e-3)


def test_combine_stack_partial_coverage_and_uncovered() -> None:
    images = np.full((3, 4, 4), 10.0)
    images[1:, 0, 0] = np.nan  # (0,0) covered by 1 frame
    images[:, 1, 1] = np.nan  # (1,1) covered by none
    data, error, count = _combine_stack(images, np.ones((3, 4, 4)))
    assert count[0, 0] == 1
    assert count[1, 1] == 0
    assert np.isclose(data[0, 0], 10.0)
    assert np.isnan(data[1, 1])
    assert np.isnan(error[1, 1])
    assert np.isclose(error[0, 0], 1.2533, atol=1e-3)
    assert np.isclose(error[3, 3], 1.2533 / np.sqrt(3), atol=1e-3)


def test_linefinder_configure_dispersion_axis() -> None:
    assert GrismLineFinder.configure(dispersion="row").dispersion_axis == 1
    assert GrismLineFinder.configure(dispersion="column").dispersion_axis == 0
    with pytest.raises(ValueError, match="dispersion"):
        GrismLineFinder.configure(dispersion="bogus")


def test_linefinder_exposure_heatmap_and_catalog() -> None:
    shape = (81, 81)
    data = _gaussian_blob(shape, 40, 40, 1.0, 20.0)
    error = np.ones(shape)
    finder = GrismLineFinder.configure(
        dispersion="row", line_sigmas=(1.0,), continuum_sigma_disp=4.0, threshold=5.0
    )
    hm = finder.exposure_heatmap(data, error)
    assert hm.shape == shape
    assert np.unravel_index(np.nanargmax(hm), shape) == (40, 40)
    cands = finder.catalog(hm)
    assert any((c.y, c.x) == (40, 40) for c in cands)


def test_linefinder_isotropic_scales_are_pairs() -> None:
    finder = GrismLineFinder.configure(dispersion="row", line_sigmas=(1.2, 2.0))
    assert finder.line_scales == ((1.2, 1.2), (2.0, 2.0))


def test_linefinder_velocity_config_is_anisotropic() -> None:
    finder = GrismLineFinder.configure(
        dispersion="row",
        line_sigmas=(0.9,),
        line_velocities_kms=(200.0,),
        reference_wavelength=4.4,
        dispersion_per_pixel=1e-3,
    )
    cross, disp = finder.line_scales[0]
    s_spec = _velocity_to_sigma(200.0, 4.4, 1e-3)
    assert cross == 0.9
    assert np.isclose(disp, np.hypot(0.9, s_spec), atol=1e-6)
    assert disp > cross  # along-dispersion exceeds the spatial sigma


def test_linefinder_velocity_cartesian_product() -> None:
    finder = GrismLineFinder.configure(
        dispersion="row",
        line_sigmas=(0.9, 1.5),
        line_velocities_kms=(150.0, 400.0),
        reference_wavelength=4.4,
        dispersion_per_pixel=1e-3,
        continuum_sigma_disp=12.0,
    )
    assert len(finder.line_scales) == 4  # 2 spatial x 2 velocities
    assert {c for c, _ in finder.line_scales} == {0.9, 1.5}


def test_linefinder_continuum_velocity_overrides() -> None:
    finder = GrismLineFinder.configure(
        dispersion="row",
        line_sigmas=(1.0,),
        continuum_velocity_kms=2000.0,
        reference_wavelength=4.4,
        dispersion_per_pixel=1e-3,
    )
    expected = _velocity_to_sigma(2000.0, 4.4, 1e-3)
    assert np.isclose(finder.continuum_sigma_disp, expected)


def test_linefinder_config_validation() -> None:
    # a km/s input without the conversion inputs
    with pytest.raises(ValueError, match="reference_wavelength"):
        GrismLineFinder.configure(dispersion="row", line_velocities_kms=(200.0,))
    # continuum disp must exceed every line disp
    with pytest.raises(ValueError, match="continuum_sigma_disp"):
        GrismLineFinder.configure(
            dispersion="row", line_sigmas=(5.0,), continuum_sigma_disp=3.0
        )
