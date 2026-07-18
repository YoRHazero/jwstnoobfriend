"""Tests for frame-level leave-one-out cross-validation (collapsed continuum)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from noobfriend.inference.spectrum import NoobLine, NoobSpectrum, NoobSpectrumSet
from noobfriend.inference.spectrum.workspace.mcmc import (
    FrameLOOResult,
    FrameLOOWarning,
)

from ._helpers import gaussian

pytest.importorskip("pymc")

_Z, _R, _REST = 5.228, 1600.0, 0.65646
_CENTER = _REST * (1.0 + _Z)


def _frame(seed: int, continuum_offset: float, line_scale: float = 1.0) -> NoobSpectrum:
    rng = np.random.default_rng(seed)
    wavelength = np.linspace(3.95, 4.17, 140)
    line = line_scale * (
        gaussian(
            wavelength, center=_CENTER, flux=6.0e-4, fwhm_kms=250.0, resolving_power=_R
        )
        + gaussian(
            wavelength, center=_CENTER, flux=1.5e-3, fwhm_kms=2400.0, resolving_power=_R
        )
    )
    flux = continuum_offset + 0.3 * (wavelength - _CENTER) + line
    flux = flux + rng.normal(0.0, 8.0e-3, wavelength.size)
    return NoobSpectrum(
        flux,
        np.full(wavelength.size, 8.0e-3),
        obs=wavelength,
        z=_Z,
        unit="um",
        resolving_power=_R,
    )


def _lines() -> list[NoobLine]:
    narrow = NoobLine("Ha", rest=_REST, z=_Z, unit="um", component="narrow")
    broad = narrow.derive(component="broad").center(delta_v_kms=(-1000.0, 1000.0))
    return [narrow, broad]


@pytest.fixture(scope="module")
def bad_frame_loo():
    # "bad" carries a discrepant line (x2 flux) the shared model cannot fit.
    frames = {f"f{i}": _frame(i, 0.05 + 0.01 * i) for i in range(4)}
    frames["bad"] = _frame(99, 0.07, line_scale=2.0)
    result = (
        NoobSpectrumSet(frames)
        .prepare(_lines(), continuum_sharing="pooled")
        .model()
        .sample(
            draws=300,
            tune=600,
            chains=2,
            cores=1,
            target_accept=0.95,
            random_seed=17,
            progressbar=False,
        )
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loo = result.frame_loo(max_draws=500, random_seed=1)
    return result, loo, caught


def test_frame_loo_isolates_the_injected_bad_frame(bad_frame_loo) -> None:
    result, loo, caught = bad_frame_loo

    assert isinstance(loo, FrameLOOResult)
    assert loo.frame_ids == ("f0", "f1", "f2", "f3", "bad")
    # The injected frame is the most influential (highest Pareto-k) and flagged.
    assert loo.frame_ids[int(np.argmax(loo.pareto_k))] == "bad"
    assert "bad" in loo.flagged
    assert not loo.reliable
    assert any(w.category is FrameLOOWarning for w in caught)
    # Per-frame chi-square corroborates: the bad frame fits far worse.
    chi = {f.frame_id: f.chi_square_per_pixel for f in result.frame_fits()}
    assert chi["bad"] > 2.0 * max(chi[f] for f in ("f0", "f1", "f2", "f3"))


def test_frame_loo_shapes_and_inequality(bad_frame_loo) -> None:
    _, loo, _ = bad_frame_loo
    n = len(loo.frame_ids)
    assert loo.elpd_i.shape == (n,)
    assert loo.pareto_k.shape == (n,)
    assert loo.n_pixels.tolist() == [140, 140, 140, 140, 140]
    assert loo.n_draws == 500
    # The clean frames satisfy the leave-one-out inequality elpd_i <= lpd_i.
    gap = loo.inequality_gap
    for index, frame_id in enumerate(loo.frame_ids):
        if frame_id != "bad":
            assert gap[index] <= 0.5


def test_frame_loo_requires_multiple_frames() -> None:
    wavelength = np.linspace(3.95, 4.17, 100)
    rng = np.random.default_rng(0)
    single = NoobSpectrum(
        0.05 + rng.normal(0.0, 0.01, wavelength.size),
        np.full(wavelength.size, 0.01),
        obs=wavelength,
        z=_Z,
        unit="um",
        resolving_power=_R,
    )
    result = (
        single.prepare([NoobLine("Ha", rest=_REST, z=_Z, unit="um")])
        .model()
        .sample(
            draws=100, tune=200, chains=2, cores=1, progressbar=False, random_seed=1
        )
    )
    with pytest.raises(ValueError, match="two frames"):
        result.frame_loo()


def test_frame_loo_rejects_non_pooled_continuum() -> None:
    frames = NoobSpectrumSet([_frame(1, 0.05), _frame(2, 0.09)])
    result = (
        frames.prepare(_lines(), continuum_sharing="shared")
        .model()
        .sample(
            draws=100, tune=200, chains=2, cores=1, progressbar=False, random_seed=1
        )
    )
    with pytest.raises(NotImplementedError, match="pooled continuum"):
        result.frame_loo()
