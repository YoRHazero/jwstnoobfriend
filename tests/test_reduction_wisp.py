"""Tests for the wisp-subtraction step ``reduction.subtract_wisp``.

Synthetic frames: sky noise plus compact sources, with a known wisp template
added at a known amplitude. The step must recover the amplitude (so the wisp is
removed) while preserving source flux, and pass ``err`` / ``dq`` through.
"""

import numpy as np

from noobfriend.reduction import subtract_wisp

_SIZE = 128


def _template() -> np.ndarray:
    """Return a smooth, localized wisp-like template (a broad off-centre blob)."""
    yy, xx = np.mgrid[0:_SIZE, 0:_SIZE]
    return np.exp(-(((xx - 40) ** 2 + (yy - 80) ** 2) / (2 * 25.0**2)))


def _frame(amp: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(clean, dirty, source_mask)``; dirty = clean + amp * template."""
    rng = np.random.default_rng(seed)
    clean = rng.normal(0.0, 1.0, size=(_SIZE, _SIZE)) + 5.0  # sky + noise
    source = np.zeros((_SIZE, _SIZE), dtype=bool)
    source[100:106, 20:26] = True
    source[30:36, 96:102] = True
    clean[source] += 300.0
    dirty = clean + amp * _template()
    return clean, dirty, source


def test_recovers_amplitude_and_preserves_sources() -> None:
    amp = 40.0
    clean, dirty, source = _frame(amp)
    err = np.ones_like(dirty)
    dq = np.zeros(dirty.shape, dtype=np.int32)

    out, out_err, out_dq = subtract_wisp(dirty, err, dq, template=_template())

    # The wisp is gone: out matches the wisp-free clean frame (background).
    bg = ~source
    assert np.std((out - clean)[bg]) < 0.5
    assert np.std(out[bg]) < 2.0

    # Source flux preserved.
    flux_clean = clean[source].sum() - np.median(clean[~source]) * source.sum()
    flux_out = out[source].sum() - np.median(out[~source]) * source.sum()
    assert np.isclose(flux_out, flux_clean, rtol=0.02)

    assert out_err is err
    assert out_dq is dq


def test_explicit_scale_and_nonnegative_clip() -> None:
    _, dirty, _ = _frame(30.0, seed=1)
    err = np.ones_like(dirty)
    dq = np.zeros(dirty.shape, dtype=np.int32)
    template = _template()

    # Explicit scale subtracts exactly that much.
    out, _, _ = subtract_wisp(dirty, err, dq, template=template, scale=10.0)
    np.testing.assert_allclose(out, dirty - 10.0 * template)

    # A frame with no wisp (and an anti-correlated template) clips the fit to 0.
    flat = np.full((_SIZE, _SIZE), 5.0)
    out2, _, _ = subtract_wisp(flat, err, dq, template=-template)
    np.testing.assert_allclose(out2, flat)


def test_rejects_shape_mismatch() -> None:
    err = np.ones((8, 8))
    dq = np.zeros((8, 8), dtype=np.int32)
    try:
        subtract_wisp(np.zeros((8, 8)), err, dq, template=np.zeros((4, 4)))
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("template shape mismatch should raise ValueError")
