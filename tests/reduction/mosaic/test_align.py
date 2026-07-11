"""Tests for the deep-self-catalog stage-3 alignment (synthetic sources only).

No real FITS: synthetic sky sources are perturbed by known per-frame offsets and
the alignment is checked to recover them (relative scatter collapses; the GAIA
tie pins the absolute frame).
"""

import numpy as np
import pytest

from noobfriend.reduction.mosaic import FrameSources, align_group
from noobfriend.reduction.mosaic._align import _fit_affine

RA0, DEC0 = 150.0, 2.0
_COS = np.cos(np.radians(DEC0))


def _field(rng, n, radius_arcsec=60.0):
    """Return ``n`` true source (ra, dec) in a box about (RA0, DEC0)."""
    r = radius_arcsec / 3600.0
    ra = RA0 + rng.uniform(-r, r, n) / _COS
    dec = DEC0 + rng.uniform(-r, r, n)
    return ra, dec


def _apply(corr, ra, dec):
    """Apply a TangentCorrection to (ra, dec), returning corrected (ra, dec)."""
    out = corr.to_models()(np.asarray(ra), np.asarray(dec))
    return np.asarray(out[0]), np.asarray(out[1])


def _scatter_mas(per_source):
    """RMS scatter (mas) of each source's corrected positions across frames."""
    resid = []
    for pts in per_source.values():
        if len(pts) >= 2:
            a = np.array(pts)
            c = a.mean(axis=0)
            d = a - c
            d[:, 0] *= _COS  # ra -> great-circle
            resid.append(d)
    r = np.vstack(resid)
    return float(np.sqrt(np.mean(np.sum(r**2, axis=1)))) * 3.6e6


def _build(rng, n_frames, n_sources, seen, shift_mas, noise_mas=0.0):
    """Overlapping frames of a shared field, each with an injected shift.

    Returns the frames, the injected per-frame (dx, dy) in degrees, and the true
    (ra, dec) with the per-frame source indices, for recovery checks.
    """
    true_ra, true_dec = _field(rng, n_sources)
    shifts = rng.uniform(-shift_mas, shift_mas, (n_frames, 2)) / 3.6e6
    frames, membership = [], []
    for k in range(n_frames):
        idx = rng.choice(n_sources, seen, replace=False)
        ra = true_ra[idx] + shifts[k, 0] / _COS
        dec = true_dec[idx] + shifts[k, 1]
        if noise_mas:
            ra = ra + rng.normal(0, noise_mas / 3.6e6, seen) / _COS
            dec = dec + rng.normal(0, noise_mas / 3.6e6, seen)
        frames.append(FrameSources(f"f{k}", ra, dec, np.ones(seen)))
        membership.append(idx)
    return frames, shifts, (true_ra, true_dec), membership


def test_recovers_injected_shifts_relative():
    rng = np.random.default_rng(0)
    frames, _, _, membership = _build(rng, 6, 60, 45, shift_mas=50.0)

    corr = align_group(frames, n_iter=2)
    assert set(corr) == {f.frame_id for f in frames}

    per_source: dict[int, list] = {}
    for frame, idx in zip(frames, membership):
        cra, cdec = _apply(corr[frame.frame_id], frame.ra, frame.dec)
        for j, s in enumerate(idx):
            per_source.setdefault(int(s), []).append((cra[j], cdec[j]))
    # noiseless translations -> aligned frames agree to ~machine precision
    assert _scatter_mas(per_source) < 0.5


def test_noise_floor_not_overcorrected():
    rng = np.random.default_rng(1)
    frames, _, _, membership = _build(rng, 8, 80, 55, shift_mas=40.0, noise_mas=5.0)

    corr = align_group(frames, n_iter=2)
    per_source: dict[int, list] = {}
    for frame, idx in zip(frames, membership):
        cra, cdec = _apply(corr[frame.frame_id], frame.ra, frame.dec)
        for j, s in enumerate(idx):
            per_source.setdefault(int(s), []).append((cra[j], cdec[j]))
    # residual collapses to ~the injected centroid noise, not below and not the shift
    assert _scatter_mas(per_source) < 12.0


def test_gaia_tie_pins_absolute_frame():
    rng = np.random.default_rng(2)
    frames, _, (true_ra, true_dec), membership = _build(rng, 6, 60, 45, shift_mas=50.0)
    # a stellar subset with known absolute (true) positions is the GAIA reference
    stars = rng.choice(60, 12, replace=False)
    gaia = (true_ra[stars], true_dec[stars])

    corr = align_group(frames, gaia=gaia, n_iter=2)

    # corrected positions of GAIA stars land on their true absolute positions
    errs = []
    star_set = set(int(s) for s in stars)
    for frame, idx in zip(frames, membership):
        cra, cdec = _apply(corr[frame.frame_id], frame.ra, frame.dec)
        for j, s in enumerate(idx):
            if int(s) in star_set:
                d = np.hypot((cra[j] - true_ra[s]) * _COS, cdec[j] - true_dec[s])
                errs.append(d * 3.6e6)
    assert np.median(errs) < 1.0  # mas, noiseless absolute tie


def test_gaia_tie_survives_noise_with_few_anchors():
    # Regression for the real-data tie collapse: with centroid noise and only
    # ~20 anchors, the old clip criterion (bare std of the positive residual
    # norms, no center term) repeatedly clipped good anchors until the affine
    # fit returned a wrong-signed offset. The tie must recover a coherent
    # absolute offset to ~the noise level.
    rng = np.random.default_rng(3)
    frames, _, (true_ra, true_dec), membership = _build(
        rng, 8, 80, 55, shift_mas=30.0, noise_mas=8.0
    )
    common = np.array([20.0, -12.0]) / 3.6e6  # raw-WCS-vs-GAIA style offset
    frames = [
        FrameSources(f.frame_id, f.ra + common[0] / _COS, f.dec + common[1], f.weight)
        for f in frames
    ]
    stars = rng.choice(80, 20, replace=False)
    gaia = (true_ra[stars], true_dec[stars])

    corr = align_group(frames, gaia=gaia, n_iter=2)

    errs = []
    star_set = set(int(s) for s in stars)
    for frame, idx in zip(frames, membership):
        cra, cdec = _apply(corr[frame.frame_id], frame.ra, frame.dec)
        for j, s in enumerate(idx):
            if int(s) in star_set:
                d = np.hypot((cra[j] - true_ra[s]) * _COS, cdec[j] - true_dec[s])
                errs.append(d * 3.6e6)
    # noise is 8 mas per detection; the tie must remove the 23-mas common
    # offset, leaving residuals at the noise level (old clip left ~20 mas)
    assert np.median(errs) < 12.0


def test_fit_affine_survives_tight_core_anchor_geometry():
    # REAL FRESCO GOODS-S GAIA-tie anchors (tangent-plane deg, embedded as
    # literals): a few ultra-tight high-membership centroids + a loose tail
    # with field-gradient structure. On these numbers the old clip criterion
    # (bare std of the positive residual norms, no center term) death-spiraled
    # 20 -> 18 -> 12 -> 4 anchors and returned a wrong-signed offset
    # (-2.5, -5.2) mas; the true tie is ~(+12.5, +6.6) mas.
    cat = np.array([
        [0.0055238236, 0.0592342248], [0.0021460183, 0.0524140888],
        [-0.0368678793, 0.0428699627], [-0.0281249297, 0.0665272418],
        [0.0511112598, 0.0340783331], [0.0297047996, 0.0073650315],
        [0.0293793800, 0.0330371395], [0.0066169081, 0.0194336920],
        [-0.0233098275, 0.0199114159], [-0.0446404564, 0.0278676692],
        [-0.0059950551, 0.0002244442], [-0.0280791630, -0.0215122804],
        [-0.0048723014, -0.0391523024], [0.0026872388, -0.0502389595],
        [-0.0262897247, -0.0458720053], [-0.0480499318, -0.0399020748],
        [-0.0104202901, -0.0447308217], [-0.0512011957, -0.0676788888],
        [0.0538309455, -0.0479988211], [0.0519043345, -0.0044954173],
    ])
    gaia = np.array([
        [0.0055258910, 0.0592343789], [0.0021492790, 0.0524131537],
        [-0.0368665630, 0.0428712676], [-0.0281208774, 0.0665258816],
        [0.0511167613, 0.0340781317], [0.0297095428, 0.0073672789],
        [0.0293852385, 0.0330358174], [0.0066211315, 0.0194380177],
        [-0.0233051644, 0.0199170605], [-0.0446361361, 0.0278733739],
        [-0.0059901751, 0.0002263390], [-0.0280766767, -0.0215093808],
        [-0.0048742802, -0.0391472463], [0.0026896610, -0.0502369274],
        [-0.0262858540, -0.0458674450], [-0.0480423997, -0.0399035311],
        [-0.0104189538, -0.0447274679], [-0.0511938126, -0.0676739818],
        [0.0538324773, -0.0479981482], [0.0519091772, -0.0044990479],
    ])
    weight = np.array([6.0, 6.0, 3.0, 3.0, 3.0, 5.0, 5.0, 6.0, 4.0, 3.0,
                       4.0, 3.0, 5.0, 6.0, 4.0, 3.0, 4.0, 2.0, 3.0, 3.0])

    _, offset = _fit_affine(cat, gaia, weight, 3, 3.0)

    s_mas = offset * 3.6e6
    assert np.hypot(s_mas[0] - 12.5, s_mas[1] - 6.6) < 4.0


def test_empty_frames_raises():
    with pytest.raises(ValueError, match="at least one frame"):
        align_group([])


def test_disjoint_frames_get_identity():
    # frames sharing no sources cannot be aligned -> identity corrections
    rng = np.random.default_rng(3)
    frames = []
    for k in range(3):
        ra, dec = _field(rng, 10)
        frames.append(FrameSources(f"f{k}", ra, dec, np.ones(10)))
    corr = align_group(frames)
    for c in corr.values():
        assert c.matrix == ((1.0, 0.0), (0.0, 1.0))
        assert c.offset == (0.0, 0.0)


def test_starved_frame_stays_identity_others_align():
    rng = np.random.default_rng(4)
    frames, _, _, _ = _build(rng, 5, 60, 45, shift_mas=30.0)
    # add a frame whose sources overlap nothing (its own far-away field)
    far_ra = RA0 + 1.0 + rng.uniform(0, 0.01, 20)
    far_dec = DEC0 + 1.0 + rng.uniform(0, 0.01, 20)
    frames.append(FrameSources("starved", far_ra, far_dec, np.ones(20)))

    corr = align_group(frames)
    assert corr["starved"].offset == (0.0, 0.0)
    assert corr["starved"].matrix == ((1.0, 0.0), (0.0, 1.0))
