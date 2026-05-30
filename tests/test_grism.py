"""Tests for the grism rectify/combine/collapse core logic.

These cover only the pure-numpy maths of :class:`GrismSpectrum.collapse` and
:meth:`GrismExtractor.combine`. They build :class:`GrismSpectrum` directly with
synthetic arrays and never touch ``gwcs`` or ``example_data`` (so the
WCS-dependent ``rectify`` / ``from_world`` paths are exercised only by the
throwaway end-to-end smoke check, not here).
"""

import numpy as np
import pytest
from noobase.axis import Grid

from noobfriend.extraction.grism import GrismExtractor, GrismSpectrum


def _spectrum(
    flux: np.ndarray,
    error: np.ndarray,
    *,
    weight: np.ndarray | None = None,
    group=None,
    contam: np.ndarray | None = None,
) -> GrismSpectrum:
    """Build a synthetic GrismSpectrum with grids inferred from ``flux`` shape."""
    flux = np.asarray(flux, dtype=float)
    error = np.asarray(error, dtype=float)
    n_spatial, n_wave = flux.shape
    if weight is None:
        weight = np.ones_like(flux)
    # Grids must be >= 2 long; combine/collapse never index the grid by the
    # array's own length, and only the rebin test (a wide array) actually uses
    # the grid values, so a min-2 placeholder is safe for the 1-wide cases.
    spatial = Grid.linspace(-1.0, 1.0, max(2, n_spatial))
    wavelength = Grid.linspace(1.0, 2.0, max(2, n_wave))
    return GrismSpectrum(
        flux_2d=flux,
        error_2d=error,
        weight_2d=np.asarray(weight, dtype=float),
        wavelength=wavelength,
        spatial_offset=spatial,
        n_frames=1,
        group=group,
        contam_2d=None if contam is None else np.asarray(contam, dtype=float),
    )


def _extractor(spec: GrismSpectrum) -> GrismExtractor:
    """Make a bare extractor sharing a spectrum's grids (combine ignores coverage)."""
    return GrismExtractor(
        ra=0.0,
        dec=0.0,
        wavelength=spec.wavelength,
        spatial_offset=spec.spatial_offset,
        coverage=(),
    )


# --------------------------------------------------------------------------- #
# combine: inverse-variance math
# --------------------------------------------------------------------------- #


def test_combine_inverse_variance_equal_values():
    a = _spectrum(np.array([[10.0, 10.0]]), np.array([[1.0, 1.0]]))
    b = _spectrum(np.array([[10.0, 20.0]]), np.array([[1.0, 1.0]]))
    ext = _extractor(a)
    (prod,) = ext.combine({"a": a, "b": b})

    # cell 0: 10 & 10 -> 10; cell 1: 10 & 20 -> 15. Both err -> 1/sqrt(2).
    np.testing.assert_allclose(prod.flux_2d, [[10.0, 15.0]])
    np.testing.assert_allclose(prod.error_2d, [[1.0 / np.sqrt(2), 1.0 / np.sqrt(2)]])
    assert prod.n_frames == 2


def test_combine_inverse_variance_unequal_errors():
    # Weighted mean of 10 (err 1) and 16 (err 2): w = 1, 0.25.
    a = _spectrum(np.array([[10.0]]), np.array([[1.0]]))
    b = _spectrum(np.array([[16.0]]), np.array([[2.0]]))
    ext = _extractor(a)
    (prod,) = ext.combine({"a": a, "b": b})

    expected_flux = (10.0 * 1.0 + 16.0 * 0.25) / (1.0 + 0.25)
    expected_err = np.sqrt(1.0 / (1.0 + 0.25))
    np.testing.assert_allclose(prod.flux_2d, [[expected_flux]])
    np.testing.assert_allclose(prod.error_2d, [[expected_err]])


# --------------------------------------------------------------------------- #
# combine: grouping
# --------------------------------------------------------------------------- #


def test_combine_grouping_partitions_by_group():
    f = np.array([[5.0, 5.0]])
    e = np.array([[1.0, 1.0]])
    a1 = _spectrum(f, e, group="A")
    a2 = _spectrum(f, e, group="A")
    b1 = _spectrum(f, e, group="B")
    ext = _extractor(a1)
    prods = ext.combine({"a1": a1, "a2": a2, "b1": b1})

    assert len(prods) == 2
    by_group = {p.group: p for p in prods}
    assert set(by_group) == {"A", "B"}
    assert by_group["A"].n_frames == 2
    assert by_group["B"].n_frames == 1


def test_combine_grouping_preserves_first_seen_order():
    f = np.array([[1.0]])
    e = np.array([[1.0]])
    specs = {
        "b": _spectrum(f, e, group="B"),
        "a": _spectrum(f, e, group="A"),
    }
    ext = _extractor(specs["b"])
    prods = ext.combine(specs)
    assert [p.group for p in prods] == ["B", "A"]


# --------------------------------------------------------------------------- #
# combine: NaN / invalid handling
# --------------------------------------------------------------------------- #


def test_combine_falls_back_to_valid_frame():
    # cell 0: a is NaN -> use b; cell 1: b has err 0 -> use a.
    a = _spectrum(np.array([[np.nan, 7.0]]), np.array([[1.0, 2.0]]))
    b = _spectrum(np.array([[3.0, 99.0]]), np.array([[5.0, 0.0]]))
    ext = _extractor(a)
    (prod,) = ext.combine({"a": a, "b": b})

    np.testing.assert_allclose(prod.flux_2d, [[3.0, 7.0]])
    np.testing.assert_allclose(prod.error_2d, [[5.0, 2.0]])


def test_combine_all_invalid_cell_is_nan():
    a = _spectrum(np.array([[np.nan]]), np.array([[1.0]]))
    b = _spectrum(np.array([[5.0]]), np.array([[0.0]]))
    ext = _extractor(a)
    (prod,) = ext.combine({"a": a, "b": b})
    assert np.isnan(prod.flux_2d[0, 0])
    assert np.isnan(prod.error_2d[0, 0])


# --------------------------------------------------------------------------- #
# collapse: boxcar sum + error in quadrature
# --------------------------------------------------------------------------- #


def test_collapse_sums_valid_rows():
    flux = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    error = np.array([[1.0, 1.0], [2.0, 2.0], [2.0, 2.0]])
    spec = _spectrum(flux, error)
    flux_1d, error_1d = spec.collapse()

    np.testing.assert_allclose(flux_1d, [9.0, 12.0])
    np.testing.assert_allclose(
        error_1d,
        [np.sqrt(1 + 4 + 4), np.sqrt(1 + 4 + 4)],
    )
    # Properties match collapse().
    np.testing.assert_allclose(spec.flux_1d, flux_1d)
    np.testing.assert_allclose(spec.error_1d, error_1d)


def test_collapse_column_all_invalid_is_nan():
    flux = np.array([[1.0, np.nan], [2.0, np.nan]])
    error = np.array([[1.0, 1.0], [1.0, 1.0]])
    spec = _spectrum(flux, error)
    flux_1d, error_1d = spec.collapse()

    np.testing.assert_allclose(flux_1d[0], 3.0)
    assert np.isnan(flux_1d[1])
    assert np.isnan(error_1d[1])


def test_collapse_excludes_zero_weight_rows():
    flux = np.array([[10.0], [100.0], [10.0]])
    error = np.array([[1.0], [1.0], [1.0]])
    weight = np.array([[1.0], [0.0], [1.0]])  # middle row excluded
    spec = _spectrum(flux, error, weight=weight)
    flux_1d, error_1d = spec.collapse()

    np.testing.assert_allclose(flux_1d, [20.0])
    np.testing.assert_allclose(error_1d, [np.sqrt(2.0)])


def test_collapse_subtracts_contamination():
    flux = np.array([[10.0], [10.0]])
    error = np.array([[1.0], [1.0]])
    contam = np.array([[2.0], [3.0]])
    spec = _spectrum(flux, error, contam=contam)
    flux_1d, _ = spec.collapse()

    np.testing.assert_allclose(flux_1d, [(10 - 2) + (10 - 3)])


# --------------------------------------------------------------------------- #
# collapse: rebin onto a target grid
# --------------------------------------------------------------------------- #


def test_collapse_rebin_returns_target_length():
    flux = np.ones((3, 10))
    error = np.ones((3, 10))
    spec = _spectrum(flux, error)
    target = Grid.linspace(1.0, 2.0, 4)
    flux_1d, error_1d = spec.collapse(wavelength=target)

    assert flux_1d.shape == (4,)
    assert error_1d.shape == (4,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
