"""Tests for the pure 1/f destriping step ``reduction.subtract_oneoverf``.

All fixtures are synthetic arrays: a flat sky with a bright source, onto which
known per-row *and* per-column stripe patterns plus a per-amplifier pedestal are
injected. The step must remove the injected additive pattern while preserving
the source flux, and pass ``err`` / ``dq`` through untouched.
"""

import numpy as np

from noobfriend.reduction.detector import subtract_oneoverf


def _frame(
    height: int = 64,
    width: int = 64,
    n_amps: int = 4,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(clean, dirty, source_mask)`` synthetic frames.

    ``clean`` is sky noise plus a bright compact source; ``dirty`` adds a random
    per-row offset and per-column offset (the two 1/f stripe directions) and a
    per-amplifier pedestal.
    """
    rng = np.random.default_rng(seed)
    clean = rng.normal(0.0, 1.0, size=(height, width))
    source = np.zeros((height, width), dtype=bool)
    source[28:36, 28:36] = True
    clean[source] += 500.0  # a bright, easily detected source

    row_stripe = rng.normal(0.0, 20.0, size=height)[:, None]
    col_stripe = rng.normal(0.0, 20.0, size=width)[None, :]
    pedestal = np.repeat(rng.normal(0.0, 15.0, size=n_amps), width // n_amps)[None, :]
    dirty = clean + row_stripe + col_stripe + pedestal
    return clean, dirty, source


def test_both_removes_row_and_column_stripes_preserving_source() -> None:
    clean, dirty, source = _frame()
    err = np.full_like(dirty, 1.0)
    dq = np.zeros(dirty.shape, dtype=np.int32)

    out, out_err, out_dq = subtract_oneoverf(dirty, err, dq)  # default by="both"

    background = ~source
    assert np.std(dirty[background]) > 15.0
    assert np.std(out[background]) < 2.0
    assert abs(np.mean(out[background])) < 0.5

    # Source flux (above background) is preserved, not eaten by the correction.
    src_flux_clean = clean[source].sum() - np.median(clean[~source]) * source.sum()
    src_flux_out = out[source].sum() - np.median(out[~source]) * source.sum()
    assert np.isclose(src_flux_out, src_flux_clean, rtol=0.02)

    # err / dq pass through unchanged (same object, untouched).
    assert out_err is err
    assert out_dq is dq


def test_single_direction_leaves_orthogonal_residual() -> None:
    # With stripes in both directions, a single-axis pass cannot clean the frame;
    # this is why "both" is the default.
    _, dirty, source = _frame()
    err = np.ones_like(dirty)
    dq = np.zeros(dirty.shape, dtype=np.int32)

    row_only, _, _ = subtract_oneoverf(dirty, err, dq, by="row")
    both, _, _ = subtract_oneoverf(dirty, err, dq, by="both")

    background = ~source
    assert np.std(row_only[background]) > np.std(both[background])
    assert np.std(both[background]) < 2.0


def test_external_mask_and_dq_exclusion() -> None:
    _, dirty, source = _frame(seed=3)
    err = np.ones_like(dirty)
    dq = np.zeros(dirty.shape, dtype=np.int32)
    dq[10, :8] = 1  # flag pixels DO_NOT_USE; must be ignored in the estimate

    out_auto, _, _ = subtract_oneoverf(dirty, err, dq)
    out_mask, _, _ = subtract_oneoverf(dirty, err, dq, mask=source)

    for out in (out_auto, out_mask):
        assert np.std(out[~source]) < 2.0


def test_rejects_non_2d_and_bad_by() -> None:
    err = np.ones((4, 4))
    dq = np.zeros((4, 4), dtype=np.int32)
    cube = np.zeros((2, 4, 4))
    try:
        subtract_oneoverf(cube, err, dq)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("3-D data should raise ValueError")

    flat = np.zeros((4, 4))
    try:
        subtract_oneoverf(flat, err, dq, by="diagonal")  # type: ignore[arg-type]
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("invalid `by` should raise ValueError")
