"""Tests for the core image utilities: segmentation and correlated-noise stats."""

import numpy as np
import pytest

from noobfriend.core.imgutils import (
    build_catalog,
    correlated_sum_variance,
    noise_autocovariance,
    segment,
)
from noobfriend.core.imgutils._noise import _lagged_autocorr


def _blob(image: np.ndarray, top: int, left: int, size: int, value: float) -> None:
    """Set a square block of ``image`` to ``value`` in place."""
    image[top : top + size, left : left + size] = value


class TestSegment:
    """Behaviour of the segmentation wrapper on synthetic images."""

    def test_two_separated_blobs_get_distinct_labels(self) -> None:
        img = np.zeros((20, 20))
        _blob(img, 2, 2, 4, 10.0)
        _blob(img, 12, 12, 4, 10.0)

        labels = segment(img, threshold=1.0, deblend=False)

        ids = set(np.unique(labels)) - {0}
        assert len(ids) == 2
        assert labels[3, 3] != 0
        assert labels[13, 13] != 0
        assert labels[3, 3] != labels[13, 13]

    def test_threshold_excludes_faint_blob(self) -> None:
        img = np.zeros((20, 20))
        _blob(img, 2, 2, 4, 0.4)  # below threshold
        _blob(img, 12, 12, 4, 10.0)  # above threshold

        labels = segment(img, threshold=1.0, deblend=False)

        assert labels[3, 3] == 0
        assert labels[13, 13] != 0
        assert set(np.unique(labels)) - {0} == {labels[13, 13]}

    def test_min_size_removes_specks(self) -> None:
        img = np.zeros((10, 10))
        img[5, 5] = 10.0

        assert segment(img, threshold=1.0, min_size=5).max() == 0
        assert segment(img, threshold=1.0, min_size=1)[5, 5] != 0

    def test_deblend_splits_touching_blobs(self) -> None:
        yy, xx = np.mgrid[0:9, 0:15]
        img = 10.0 * np.exp(-(((xx - 4) ** 2 + (yy - 4) ** 2) / 4.0))
        img += 10.0 * np.exp(-(((xx - 10) ** 2 + (yy - 4) ** 2) / 4.0))

        merged = segment(img, threshold=1.0, deblend=False)
        split = segment(img, threshold=1.0, deblend=True)

        assert len(set(np.unique(merged)) - {0}) == 1
        assert len(set(np.unique(split)) - {0}) == 2

    def test_nan_is_background(self) -> None:
        img = np.full((10, 10), np.nan)
        _blob(img, 2, 2, 4, 10.0)

        labels = segment(img, threshold=1.0, deblend=False)

        assert labels[0, 0] == 0
        assert labels[3, 3] != 0
        assert segment(np.full((5, 5), np.nan)).max() == 0

    def test_auto_threshold_from_noise(self) -> None:
        # Deterministic background: median 0, MAD 1 -> sigma ~1.48,
        # so a 2-sigma threshold sits near 2.97; noise (max 2) stays background.
        img = np.tile(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]), (20, 4))
        _blob(img, 8, 8, 4, 20.0)

        labels = segment(img, nsigma=2.0, deblend=False)

        assert labels[9, 9] != 0
        assert labels[0, 0] == 0

    def test_connectivity_distinguishes_diagonal_neighbours(self) -> None:
        img = np.zeros((6, 6))
        img[1, 1] = 10.0
        img[2, 2] = 10.0  # diagonally adjacent to (1, 1)

        eight = segment(img, threshold=1.0, min_size=1, connectivity=8, deblend=False)
        four = segment(img, threshold=1.0, min_size=1, connectivity=4, deblend=False)

        assert len(set(np.unique(eight)) - {0}) == 1
        assert eight[1, 1] == eight[2, 2]
        assert len(set(np.unique(four)) - {0}) == 2
        assert four[1, 1] != four[2, 2]

    def test_invalid_connectivity_raises(self) -> None:
        with pytest.raises(ValueError, match="connectivity must be 4 or 8"):
            segment(np.zeros((4, 4)), connectivity=6)  # type: ignore[arg-type]

    def test_non_2d_raises(self) -> None:
        with pytest.raises(ValueError, match="must be 2-D"):
            segment(np.zeros((3, 3, 3)))


def _brute_autocorr(a: np.ndarray, lag: int) -> np.ndarray:
    """Linear autocorrelation by an explicit pixel-pair loop (test oracle)."""
    n_rows, n_cols = a.shape
    out = np.zeros((2 * lag + 1, 2 * lag + 1))
    for i, dy in enumerate(range(-lag, lag + 1)):
        for j, dx in enumerate(range(-lag, lag + 1)):
            total = 0.0
            for r in range(n_rows):
                for c in range(n_cols):
                    rr, cc = r + dy, c + dx
                    if 0 <= rr < n_rows and 0 <= cc < n_cols:
                        total += a[r, c] * a[rr, cc]
            out[i, j] = total
    return out


def _brute_variance(weights: np.ndarray, autocov: object) -> float:
    """Double sum ``Σ_ij w_i w_j C(r_i - r_j)`` truncated at ``max_lag``."""
    lag = autocov.max_lag  # type: ignore[attr-defined]
    cov = autocov.cov  # type: ignore[attr-defined]
    pix = list(zip(*np.where(weights != 0.0)))
    total = 0.0
    for r1, c1 in pix:
        for r2, c2 in pix:
            dy, dx = r1 - r2, c1 - c2
            if abs(dy) <= lag and abs(dx) <= lag:
                total += weights[r1, c1] * weights[r2, c2] * cov[dy + lag, dx + lag]
    return total


def _add_gaussian(
    image: np.ndarray,
    yc: float,
    xc: float,
    peak: float,
    sigma_y: float,
    sigma_x: float = None,  # type: ignore[assignment]
) -> None:
    """Add an (optionally elongated, axis-aligned) 2-D Gaussian in place."""
    if sigma_x is None:
        sigma_x = sigma_y
    yy, xx = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    image += peak * np.exp(
        -0.5 * (((yy - yc) / sigma_y) ** 2 + ((xx - xc) / sigma_x) ** 2)
    )


class TestBuildCatalog:
    """Matched-filter detection and weighted-moment shape measurement."""

    def test_recovers_known_positions(self) -> None:
        rng = np.random.default_rng(0)
        img = rng.normal(0.0, 0.5, size=(80, 80))
        _add_gaussian(img, 25.3, 40.6, 200.0, sigma_y=1.5)
        _add_gaussian(img, 55.0, 18.0, 120.0, sigma_y=1.5)

        cat = build_catalog(img, fwhm=3.5, nsigma=5.0)

        assert len(cat) == 2
        assert cat.y[0] == pytest.approx(25.3, abs=0.3)
        assert cat.x[0] == pytest.approx(40.6, abs=0.3)
        assert cat.peak[0] > cat.peak[1]  # ordered by descending peak

    def test_round_vs_elongated_ellipticity(self) -> None:
        img = np.zeros((60, 120))
        _add_gaussian(img, 30.0, 30.0, 100.0, sigma_y=2.0, sigma_x=2.0)
        _add_gaussian(img, 30.0, 90.0, 100.0, sigma_y=2.0, sigma_x=5.0)

        cat = build_catalog(img, fwhm=4.0, nsigma=3.0)
        by_x = np.argsort(cat.x)

        assert cat.ellipticity[by_x][0] < 0.1
        assert cat.ellipticity[by_x][1] > 0.3

    def test_measured_fwhm_matches_input(self) -> None:
        img = np.zeros((60, 60))
        sigma = 2.0
        _add_gaussian(img, 30.0, 30.0, 500.0, sigma_y=sigma)

        cat = build_catalog(img, fwhm=2.35 * sigma, nsigma=3.0)

        assert len(cat) == 1
        assert cat.fwhm[0] == pytest.approx(2.3548 * sigma, rel=0.15)

    def test_pixel_to_world_fills_ra_dec(self) -> None:
        img = np.zeros((60, 60))
        _add_gaussian(img, 30.0, 30.0, 200.0, sigma_y=2.0)

        def p2w(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return 53.0 + 0.01 * x, -27.0 + 0.01 * y

        cat = build_catalog(img, fwhm=4.0, nsigma=3.0, pixel_to_world=p2w)

        assert cat.ra[0] == pytest.approx(53.0 + 0.01 * cat.x[0])
        assert cat.dec[0] == pytest.approx(-27.0 + 0.01 * cat.y[0])

    def test_no_wcs_gives_nan_sky(self) -> None:
        img = np.zeros((40, 40))
        _add_gaussian(img, 20.0, 20.0, 200.0, sigma_y=2.0)
        cat = build_catalog(img, fwhm=4.0, nsigma=3.0)
        assert np.isnan(cat.ra[0]) and np.isnan(cat.dec[0])

    def test_mask_excludes_source(self) -> None:
        rng = np.random.default_rng(11)
        img = rng.normal(0.0, 1.0, size=(60, 120))
        _add_gaussian(img, 30.0, 30.0, 200.0, sigma_y=2.0)  # masked away
        _add_gaussian(img, 30.0, 90.0, 200.0, sigma_y=2.0)  # survives
        mask = np.zeros_like(img, dtype=bool)
        mask[15:46, 15:46] = True  # over the first source and its wings

        cat = build_catalog(img, fwhm=4.0, nsigma=5.0, mask=mask)

        assert len(cat) == 1
        assert cat.x[0] == pytest.approx(90.0, abs=0.5)

    def test_negative_error_excludes_source(self) -> None:
        rng = np.random.default_rng(12)
        img = rng.normal(0.0, 1.0, size=(60, 120))
        _add_gaussian(img, 30.0, 30.0, 200.0, sigma_y=2.0)  # excluded by bad err
        _add_gaussian(img, 30.0, 90.0, 200.0, sigma_y=2.0)  # survives
        err = np.ones_like(img)
        err[15:46, 15:46] = -1.0  # non-physical error over the first source

        cat = build_catalog(img, fwhm=4.0, nsigma=5.0, err=err)

        assert len(cat) == 1
        assert cat.x[0] == pytest.approx(90.0, abs=0.5)

    def test_snr_from_error_map(self) -> None:
        img = np.zeros((60, 60))
        _add_gaussian(img, 30.0, 30.0, 500.0, sigma_y=2.0)
        err = np.ones_like(img)

        cat = build_catalog(img, fwhm=4.0, nsigma=3.0, err=err)

        assert len(cat) == 1
        assert np.isfinite(cat.snr[0]) and cat.snr[0] > 50.0

    def test_nearest_neighbour_distance(self) -> None:
        img = np.zeros((60, 60))
        _add_gaussian(img, 20.0, 20.0, 100.0, sigma_y=1.5)
        _add_gaussian(img, 20.0, 40.0, 100.0, sigma_y=1.5)

        cat = build_catalog(img, fwhm=3.0, nsigma=3.0)

        assert len(cat) == 2
        np.testing.assert_allclose(cat.nn_dist, [20.0, 20.0], atol=0.5)

    def test_hot_pixel_is_sharper_than_source(self) -> None:
        img = np.zeros((60, 120))
        _add_gaussian(img, 30.0, 30.0, 100.0, sigma_y=2.0)
        img[30, 90] = 100.0  # single hot pixel

        cat = build_catalog(img, fwhm=4.0, nsigma=3.0)
        by_x = np.argsort(cat.x)

        assert cat.sharpness[by_x][1] > cat.sharpness[by_x][0]

    def test_boolean_subset(self) -> None:
        img = np.zeros((60, 120))
        _add_gaussian(img, 30.0, 30.0, 100.0, sigma_y=2.0, sigma_x=2.0)
        _add_gaussian(img, 30.0, 90.0, 100.0, sigma_y=2.0, sigma_x=6.0)

        cat = build_catalog(img, fwhm=4.0, nsigma=3.0)
        compact = cat[cat.ellipticity < 0.1]

        assert len(compact) == 1
        assert compact.x[0] == pytest.approx(30.0, abs=0.5)

    def test_empty_when_nothing_above_threshold(self) -> None:
        rng = np.random.default_rng(3)
        img = rng.normal(0.0, 1.0, size=(40, 40))
        assert len(build_catalog(img, fwhm=3.0, nsigma=50.0)) == 0

    def test_to_table_columns(self) -> None:
        img = np.zeros((40, 40))
        _add_gaussian(img, 20.0, 20.0, 100.0, sigma_y=2.0)
        table = build_catalog(img, fwhm=4.0, nsigma=3.0).to_table()
        assert set(table.colnames) == {
            "x",
            "y",
            "ra",
            "dec",
            "fwhm",
            "ellipticity",
            "flux",
            "peak",
            "snr",
            "sharpness",
            "nn_dist",
        }

    def test_non_2d_raises(self) -> None:
        with pytest.raises(ValueError, match="must be 2-D"):
            build_catalog(np.zeros((3, 3, 3)), fwhm=3.0)

    def test_non_positive_fwhm_raises(self) -> None:
        with pytest.raises(ValueError, match="fwhm must be positive"):
            build_catalog(np.zeros((10, 10)), fwhm=0.0)

    def test_err_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="err shape"):
            build_catalog(np.zeros((10, 10)), fwhm=3.0, err=np.zeros((5, 5)))

    def test_mask_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="mask shape"):
            build_catalog(
                np.zeros((10, 10)), fwhm=3.0, mask=np.zeros((5, 5), dtype=bool)
            )


class TestLaggedAutocorr:
    """The FFT autocorrelation engine matches an explicit pair loop."""

    def test_matches_bruteforce(self) -> None:
        a = np.random.default_rng(0).normal(size=(7, 6))
        np.testing.assert_allclose(
            _lagged_autocorr(a, 2), _brute_autocorr(a, 2), atol=1e-10
        )


class TestNoiseAutocovariance:
    """Estimating C(d) from masked sky."""

    def test_zero_lag_is_sky_variance(self) -> None:
        img = np.random.default_rng(1).normal(0, 3.0, size=(40, 40))
        ac = noise_autocovariance(img, mask=np.ones_like(img, dtype=bool), max_lag=4)
        assert ac.variance == pytest.approx(img.var(), rel=1e-6)
        assert ac.sigma == pytest.approx(img.std(), rel=1e-6)
        assert ac.n_pairs[4, 4] == pytest.approx(img.size)
        assert ac.autocorr[4, 4] == pytest.approx(1.0)

    def test_mask_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="mask shape"):
            noise_autocovariance(
                np.zeros((20, 20)), mask=np.ones((10, 10), dtype=bool), max_lag=2
            )

    def test_max_lag_window_must_fit(self) -> None:
        img = np.zeros((10, 10))
        with pytest.raises(ValueError, match="does not fit"):
            noise_autocovariance(img, mask=np.ones_like(img, dtype=bool), max_lag=6)

    def test_too_few_sky_pixels_raises(self) -> None:
        mask = np.zeros((20, 20), dtype=bool)
        mask[:3, :3] = True  # 9 sky pixels < (2*2+1)**2 = 25
        with pytest.raises(ValueError, match="too few sky"):
            noise_autocovariance(np.zeros((20, 20)), mask=mask, max_lag=2)


class TestCorrelatedSumVariance:
    """Folding C(d) with the aperture weight autocorrelation."""

    def test_matches_bruteforce_fold(self) -> None:
        img = np.random.default_rng(2).normal(0, 2.0, size=(24, 24))
        ac = noise_autocovariance(img, mask=np.ones_like(img, dtype=bool), max_lag=3)
        w = np.zeros((24, 24))
        w[10:14, 11:15] = 1.0
        w[10, 11] = 0.5  # a partially-covered boundary pixel
        assert correlated_sum_variance(w, ac) == pytest.approx(
            _brute_variance(w, ac), rel=1e-6
        )

    def test_white_noise_boost_near_one(self) -> None:
        img = np.random.default_rng(3).normal(0, 1.0, size=(128, 128))
        ac = noise_autocovariance(img, mask=np.ones_like(img, dtype=bool), max_lag=5)
        w = np.zeros((128, 128))
        w[60:68, 60:68] = 1.0
        formal = ac.variance * float(np.sum(w**2))
        assert correlated_sum_variance(w, ac) == pytest.approx(formal, rel=0.2)

    def test_hybrid_reduces_to_stationary_for_uniform_error(self) -> None:
        img = np.random.default_rng(4).normal(0, 2.0, size=(40, 40))
        err = np.full_like(img, 1.7)
        ac = noise_autocovariance(
            img, mask=np.ones_like(img, dtype=bool), max_lag=4, error=err
        )
        w = np.zeros((40, 40))
        w[18:23, 18:23] = 1.0
        assert correlated_sum_variance(w, ac, error=err) == pytest.approx(
            correlated_sum_variance(w, ac), rel=1e-6
        )

    def test_hybrid_needs_error_aware_autocov(self) -> None:
        ac = noise_autocovariance(
            np.ones((30, 30)), mask=np.ones((30, 30), dtype=bool), max_lag=3
        )
        with pytest.raises(ValueError, match="hybrid variance needs"):
            correlated_sum_variance(np.ones((30, 30)), ac, error=np.ones((30, 30)))

    def test_weights_must_be_2d(self) -> None:
        ac = noise_autocovariance(
            np.ones((30, 30)), mask=np.ones((30, 30), dtype=bool), max_lag=3
        )
        with pytest.raises(ValueError, match="weights must be 2-D"):
            correlated_sum_variance(np.ones(5), ac)
