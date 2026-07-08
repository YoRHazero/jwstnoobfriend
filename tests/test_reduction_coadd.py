"""Tests for the inverse-variance tile coadd (synthetic frames only)."""

import numpy as np
from astropy.wcs import WCS

from noobfriend.core.imgutils import NoiseAutocovariance
from noobfriend.reduction.mosaic import (
    TileCoadd,
    TileSpec,
    field_grid,
    noise_kernel,
)
from noobfriend.core.wcs import from_fits_wcs

RA0, DEC0 = 150.0, 2.0
SCALE = 0.5  # arcsec/pix


def _tan(ra0, dec0, shape):
    """Return a plain TAN astropy WCS centred at (ra0, dec0)."""
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra0, dec0]
    w.wcs.crpix = [shape[1] / 2 + 0.5, shape[0] / 2 + 0.5]
    w.wcs.cd = (SCALE / 3600.0) * np.array([[-1.0, 0.0], [0.0, 1.0]])
    return w


def _corners(w, shape):
    ny, nx = shape
    px = [(0, 0), (nx, 0), (nx, ny), (0, ny)]
    return np.array([w.wcs_pix2world([[x, y]], 0)[0] for x, y in px], dtype=float)


def _setup(shape=(100, 100)):
    """Return a field, a whole-field tile and a frame WCS, all co-aligned."""
    w = _tan(RA0, DEC0, shape)
    field = field_grid([_corners(w, shape)], SCALE, rotation=0.0)
    tile = TileSpec(0, 0, 0, 0, field.shape[1], field.shape[0])
    return field, tile, w, shape


def _covered_median(plane):
    return float(np.nanmedian(plane[np.isfinite(plane)]))


def test_coadd_conserves_surface_brightness():
    field, tile, w, shape = _setup()
    co = TileCoadd(field, tile)
    for _ in range(3):  # three identical constant frames
        co.add(
            np.full(shape, 8.0),
            np.full(shape, 1.0),
            np.zeros(shape, np.int32),
            from_fits_wcs(w),
        )
    out = co.result()
    assert abs(_covered_median(out.sci) - 8.0) < 0.05


def test_coadd_inverse_variance_weights_values():
    field, tile, w, shape = _setup()
    co = TileCoadd(field, tile)
    # value 10 at err 1 (ivm 1), value 20 at err 2 (ivm 1/4) -> (10 + 5) / 1.25 = 12
    co.add(
        np.full(shape, 10.0),
        np.full(shape, 1.0),
        np.zeros(shape, np.int32),
        from_fits_wcs(w),
    )
    co.add(
        np.full(shape, 20.0),
        np.full(shape, 2.0),
        np.zeros(shape, np.int32),
        from_fits_wcs(w),
    )
    out = co.result()
    assert abs(_covered_median(out.sci) - 12.0) < 0.1


def test_coadd_err_is_inverse_sqrt_weight():
    field, tile, w, shape = _setup()
    co = TileCoadd(field, tile)
    # N identical frames at err E -> combined err ~ E / sqrt(N)
    for _ in range(4):
        co.add(
            np.full(shape, 5.0),
            np.full(shape, 2.0),
            np.zeros(shape, np.int32),
            from_fits_wcs(w),
        )
    out = co.result()
    assert np.allclose(
        out.err[np.isfinite(out.err)],
        1.0 / np.sqrt(out.wht[out.wht > 0]),
        equal_nan=True,
    )
    assert abs(_covered_median(out.err) - 2.0 / np.sqrt(4)) < 0.05


def test_coadd_subtracts_sky():
    field, tile, w, shape = _setup()
    co = TileCoadd(field, tile)
    co.add(
        np.full(shape, 30.0),
        np.full(shape, 1.0),
        np.zeros(shape, np.int32),
        from_fits_wcs(w),
        sky=25.0,
    )
    out = co.result()
    assert abs(_covered_median(out.sci) - 5.0) < 0.05


def test_coadd_masks_bad_dq():
    field, tile, w, shape = _setup()
    co = TileCoadd(field, tile)
    data = np.full(shape, 4.0)
    dq = np.zeros(shape, np.int32)
    dq[:50, :] = 1  # half the frame is DO_NOT_USE
    co.add(data, np.full(shape, 1.0), dq, from_fits_wcs(w))
    out = co.result()
    # the masked half has no coverage; the good half coadds to 4.0
    assert abs(_covered_median(out.sci) - 4.0) < 0.05
    assert np.isnan(out.sci).sum() > 0


def test_empty_coadd_is_all_nan():
    field, tile, _, _ = _setup()
    out = TileCoadd(field, tile).result()
    assert np.isnan(out.sci).all()
    assert np.all(out.wht == 0)


def test_noise_kernel_measures_correlated_noise():
    field, tile, w, shape = _setup()
    rng = np.random.default_rng(0)
    co = TileCoadd(field, tile)
    for _ in range(5):
        data = 6.0 + rng.normal(0, 1.0, shape)
        co.add(data, np.full(shape, 1.0), np.zeros(shape, np.int32), from_fits_wcs(w))
    out = co.result()
    ac = noise_kernel(out.sci, out.err, max_lag=8)
    assert isinstance(ac, NoiseAutocovariance)
    # C(0) (per-pixel variance) is positive and near the measured background var
    measured = np.nanvar(out.sci[np.isfinite(out.sci)])
    assert ac.variance > 0
    assert 0.3 < ac.variance / measured < 3.0
    assert ac.error_var is not None  # error-aware -> hybrid-capable
