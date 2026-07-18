"""Tests for joining several frames into one prepared workspace."""

from __future__ import annotations

import numpy as np
import pytest

from noobfriend.inference.spectrum import (
    NoobLine,
    NoobSpectrum,
    NoobSpectrumSet,
)


def _frame(
    *,
    lo: float = 39000.0,
    hi: float = 41000.0,
    n: int = 100,
    z: float | None = 7.0,
    unit: str = "angstrom",
    resolving_power: float | None = None,
) -> NoobSpectrum:
    obs = np.linspace(lo, hi, n)
    return NoobSpectrum(
        np.ones_like(obs),
        np.full_like(obs, 0.1),
        obs=obs,
        z=z,
        unit=unit,
        resolving_power=resolving_power,
    )


def test_sequence_assigns_positional_frame_ids() -> None:
    frames = NoobSpectrumSet([_frame(), _frame(), _frame()])

    assert frames.n_frames == 3
    assert frames.frame_ids == ("frame_0", "frame_1", "frame_2")
    assert frames.unit == "angstrom"
    assert frames.z == pytest.approx(7.0)
    assert len(frames) == 3
    assert list(frames) == list(frames.frames)


def test_mapping_uses_exposure_names_as_frame_ids() -> None:
    frames = NoobSpectrumSet({"exp_a": _frame(), "exp_b": _frame()})

    assert frames.frame_ids == ("exp_a", "exp_b")
    assert frames.resolving_powers == (None, None)


def test_resolving_powers_reports_the_shared_value() -> None:
    frames = NoobSpectrumSet(
        [_frame(resolving_power=1600.0), _frame(resolving_power=1600.0)]
    )

    assert frames.resolving_powers == (1600.0, 1600.0)


def test_rejects_empty_and_non_container_input() -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        NoobSpectrumSet([])

    with pytest.raises(TypeError, match="sequence of NoobSpectrum"):
        NoobSpectrumSet(_frame())  # type: ignore[arg-type]


def test_rejects_non_spectrum_entries_and_bad_ids() -> None:
    with pytest.raises(TypeError, match="NoobSpectrum instances"):
        NoobSpectrumSet([_frame(), object()])  # type: ignore[list-item]

    with pytest.raises(ValueError, match="non-empty strings"):
        NoobSpectrumSet({"": _frame()})


def test_rejects_inconsistent_unit_redshift_and_resolution() -> None:
    with pytest.raises(ValueError, match="one wavelength unit"):
        NoobSpectrumSet([_frame(unit="angstrom"), _frame(unit="nm")])

    with pytest.raises(ValueError, match="all set or all unset"):
        NoobSpectrumSet([_frame(z=7.0), _frame(z=None)])

    with pytest.raises(ValueError, match="one source redshift"):
        NoobSpectrumSet([_frame(z=7.0), _frame(z=6.0)])

    # A joint fit is one source at a single spectral resolution: frames must
    # share one resolving power (a different value, or a set/unset mix, is out).
    with pytest.raises(ValueError, match="one resolving_power"):
        NoobSpectrumSet(
            [_frame(resolving_power=1600.0), _frame(resolving_power=2700.0)]
        )
    with pytest.raises(ValueError, match="one resolving_power"):
        NoobSpectrumSet([_frame(resolving_power=1600.0), _frame(resolving_power=None)])


def test_prepare_shares_lines_and_keeps_frames() -> None:
    base = NoobLine("OIII", rest=5008.24, z=7.0)
    broad = base.derive(component="broad")
    frames = NoobSpectrumSet(
        [_frame(lo=39000.0, hi=41000.0), _frame(lo=39500.0, hi=41500.0, n=90)]
    )

    workspace = frames.prepare([base, broad], continuum_sharing="pooled")

    assert workspace.n_frames == 2
    assert workspace.frame_ids == ("frame_0", "frame_1")
    assert workspace.ids == ("OIII.narrow", "OIII.broad")
    assert workspace.continuum.sharing == "pooled"
    assert workspace.spectra[1].obs.size == 90


def test_prepare_line_needs_only_union_coverage() -> None:
    # Line center 40100 is inside frame A only; the union still covers it.
    covered_in_a_only = NoobLine("mid", obs=40100.0)
    frames = NoobSpectrumSet(
        [_frame(lo=39000.0, hi=40500.0), _frame(lo=41000.0, hi=42000.0, z=7.0)]
    )

    workspace = frames.prepare([covered_in_a_only])
    assert workspace.handles[0].observed_wavelength == pytest.approx(40100.0)


def test_prepare_rejects_line_outside_every_frame() -> None:
    outside = NoobLine("outside", obs=50000.0)
    frames = NoobSpectrumSet([_frame(), _frame(lo=39500.0, hi=41500.0)])

    with pytest.raises(ValueError, match="outside every frame"):
        frames.prepare([outside])


def test_default_continuum_sharing_is_pooled() -> None:
    base = NoobLine("OIII", rest=5008.24, z=7.0)
    workspace = NoobSpectrumSet([_frame(), _frame()]).prepare([base])

    assert workspace.continuum.sharing == "pooled"


def test_multi_frame_mle_and_summary_defer_cleanly() -> None:
    # The joint MCMC model is available (see test_mcmc_joint). MLE stays
    # single-frame by design (coadd, or use .model()); the HTML summary is
    # single-frame only for now.
    base = NoobLine("OIII", rest=5008.24, z=7.0)
    workspace = NoobSpectrumSet([_frame(), _frame()]).prepare([base])

    with pytest.raises(NotImplementedError, match="single-frame method"):
        workspace.mle()
    with pytest.raises(NotImplementedError, match="multi-frame"):
        workspace.summary()


def test_multi_frame_spectrum_accessor_raises() -> None:
    base = NoobLine("OIII", rest=5008.24, z=7.0)
    workspace = NoobSpectrumSet([_frame(), _frame()]).prepare([base])

    with pytest.raises(RuntimeError, match="holds 2 frames"):
        _ = workspace.spectrum


def test_single_frame_set_matches_single_spectrum() -> None:
    base = NoobLine("OIII", rest=5008.24, z=7.0)
    spectrum = _frame()

    direct = spectrum.prepare([base])
    via_set = NoobSpectrumSet([spectrum]).prepare([base])

    assert via_set.n_frames == 1
    assert via_set.ids == direct.ids
    assert via_set.spectrum is spectrum
    assert via_set.continuum.lambda_0 == pytest.approx(direct.continuum.lambda_0)
