"""Tests for the per-detector fixed-pattern step ``reduction._fixed_pattern``.

Synthetic arrays only: a known zero-mean crosshatch is injected on a smooth sky.
The build helpers must recover the fixed pattern from a stack of dithered
(differently source-masked) frames, the amplitude fit must recover a known
per-frame scale robustly to the sky level, and the subtract step must remove the
pattern while passing ``err`` / ``dq`` through untouched.
"""

import numpy as np

from noobfriend.reduction import (
    combine_fixed_pattern,
    fit_pattern_amplitude,
    fit_pattern_amplitude_offset,
    fixed_pattern_residual,
    grism_fixed_pattern_frame,
    subtract_fixed_pattern,
    subtract_grism_fixed_pattern,
)


def _pattern(height: int = 64, width: int = 64) -> np.ndarray:
    """Return a deterministic zero-mean diagonal crosshatch (period 8 px)."""
    y, x = np.mgrid[0:height, 0:width]
    p = np.sin(2 * np.pi * (x + y) / 8.0) + np.sin(2 * np.pi * (x - y) / 8.0)
    return p - p.mean()


def test_fit_amplitude_recovers_scale_robust_to_sky_level() -> None:
    rng = np.random.default_rng(0)
    template = _pattern()
    dq = np.zeros(template.shape, dtype=np.int32)
    gradient = 0.002 * np.arange(template.shape[1])[None, :]
    for sky in (5.0, 1000.0):  # mean-subtraction makes the fit level-independent
        data = sky + gradient + 0.7 * template + rng.normal(0, 0.02, template.shape)
        assert np.isclose(fit_pattern_amplitude(data, dq, template), 0.7, atol=0.03)


def test_fit_amplitude_zero_when_template_has_no_power() -> None:
    template = np.zeros((32, 32))
    data = np.ones((32, 32))
    dq = np.zeros((32, 32), dtype=np.int32)
    assert fit_pattern_amplitude(data, dq, template) == 0.0


def test_combine_recovers_fixed_pattern_and_zeros_low_coverage() -> None:
    rng = np.random.default_rng(1)
    truth = _pattern()
    residuals = []
    for i in range(10):
        r = truth + rng.normal(0, 0.3, truth.shape)
        r[i * 4 : i * 4 + 6, 10:20] = (
            np.nan
        )  # a dithered source, masked this frame only
        r[:, 0] = np.nan  # a column masked in EVERY frame -> zero coverage
        residuals.append(r)
    template = combine_fixed_pattern(residuals)

    corr = np.corrcoef(template[:, 1:].ravel(), truth[:, 1:].ravel())[0, 1]
    assert corr > 0.9
    assert abs(np.median(template[template != 0])) < 0.05  # zero-mean over covered
    assert np.all(template[:, 0] == 0.0)  # always-masked column -> no correction


def test_combine_empty_raises() -> None:
    try:
        combine_fixed_pattern([])
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("empty residuals should raise ValueError")


def test_subtract_removes_pattern_and_passes_err_dq() -> None:
    rng = np.random.default_rng(2)
    template = _pattern()
    err = np.ones_like(template)
    dq = np.zeros(template.shape, dtype=np.int32)
    data = 10.0 + 0.8 * template + rng.normal(0, 0.02, template.shape)

    out, out_err, out_dq = subtract_fixed_pattern(data, err, dq, template)

    assert np.std(out - np.median(out)) < 0.3 * np.std(data - 10.0)
    assert out_err is err
    assert out_dq is dq


def test_subtract_fixed_alpha_override() -> None:
    template = _pattern()
    err = np.ones_like(template)
    dq = np.zeros(template.shape, dtype=np.int32)
    out, _, _ = subtract_fixed_pattern(template.copy(), err, dq, template, alpha=1.0)
    assert np.allclose(out, 0.0, atol=1e-9)


def test_subtract_rejects_bad_shapes() -> None:
    err = np.ones((8, 8))
    dq = np.zeros((8, 8), dtype=np.int32)
    template = np.zeros((8, 8))
    try:
        subtract_fixed_pattern(np.zeros((2, 8, 8)), err, dq, template)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("3-D data should raise ValueError")
    try:
        subtract_fixed_pattern(np.zeros((8, 8)), err, dq, np.zeros((8, 4)))
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("mismatched template shape should raise ValueError")


def test_residual_masks_sources_and_keeps_pattern() -> None:
    rng = np.random.default_rng(3)
    truth = _pattern()
    gradient = 0.05 * np.arange(truth.shape[0])[:, None]
    data = 5.0 + gradient + truth + rng.normal(0, 0.02, truth.shape)
    data[30:34, 30:34] += 500.0  # a bright source
    dq = np.zeros(truth.shape, dtype=np.int32)
    dq[0, 0] = 1  # DO_NOT_USE

    res = fixed_pattern_residual(data, dq, box_size=16)

    assert np.isnan(res[31, 31])  # source masked out
    assert np.isnan(res[0, 0])  # flagged pixel masked out
    # the smooth gradient is detrended away; the fixed crosshatch survives
    good = np.isfinite(res)
    good[28:36, 28:36] = False  # avoid the dilated source region
    assert np.corrcoef(res[good], truth[good])[0, 1] > 0.8


def test_grism_fixed_pattern_frame_keeps_gradient_and_masks_traces() -> None:
    truth = _pattern()
    gradient = 0.03 * np.arange(truth.shape[1])[None, :]
    data = 20.0 + gradient + 0.4 * truth
    dq = np.zeros(truth.shape, dtype=np.int32)
    dq[0, 0] = 1
    mask = np.zeros(truth.shape, dtype=bool)
    mask[30:34, :] = True  # dispersed trace

    frame = grism_fixed_pattern_frame(data, dq, mask=mask)

    assert np.isnan(frame[31, 10])
    assert np.isnan(frame[0, 0])
    assert abs(np.nanmedian(frame)) < 1e-9
    expected = gradient + 0.4 * truth
    expected = expected - np.median(expected[~mask & (dq == 0)])
    good = np.isfinite(frame)
    assert np.corrcoef(frame[good].ravel(), expected[good].ravel())[0, 1] > 0.99


def test_grism_fit_uses_offset_and_subtracts_only_template() -> None:
    template = 0.1 * np.arange(64)[None, :] + _pattern()
    template = template - np.median(template)
    dq = np.zeros(template.shape, dtype=np.int32)
    err = np.ones_like(template)
    data = 100.0 + 0.35 * template
    mask = np.zeros(template.shape, dtype=bool)
    mask[:, 0:4] = True

    alpha, beta = fit_pattern_amplitude_offset(data, dq, template, mask=mask)
    out, out_err, out_dq = subtract_grism_fixed_pattern(
        data, err, dq, template, mask=mask
    )

    assert np.isclose(alpha, 0.35, atol=1e-9)
    assert np.isclose(beta, 100.0, atol=1e-9)
    assert np.allclose(out, 100.0, atol=1e-9)
    assert out_err is err
    assert out_dq is dq
