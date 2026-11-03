"""Sampling tests with known ground truth at R = 1600 (slow: they run NUTS).

Run with ``uv run --extra mcmc pytest -m slow tests/inference/spectrum/mcmc``.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numpyro")

from scipy import stats  # noqa: E402

from noobfriend.inference.spectrum import (  # noqa: E402
    AR1,
    Noise,
    NoobContinuum,
    NoobLine,
    NoobSpectrum,
    TruncatedNormal,
    Uniform,
)
from noobfriend.inference.spectrum._xp import numpy_namespace  # noqa: E402

from .._helpers import HALPHA, Z, halpha_spectrum  # noqa: E402

pytestmark = pytest.mark.slow

LAMBDA_0 = HALPHA * (1 + Z)
CONTINUUM = NoobContinuum.polynomial(0, lambda_0=LAMBDA_0)


def test_recovers_a_known_line() -> None:
    spectrum = halpha_spectrum(seed=11, flux=40.0, fwhm_kms=300.0, delta_v_kms=50.0)
    workspace = spectrum.prepare(
        [NoobLine("Ha", rest=HALPHA, unit="angstrom")], continuum=CONTINUUM
    )
    result = workspace.model().sample(draws=500, tune=500, chains=8)
    assert result.sampling.verdict.converged
    for name, truth in (("fwhm", 300.0), ("flux", 40.0), ("delta_v", 50.0)):
        lower, upper = getattr(result["Ha"], name).hdi(0.99)
        assert lower < truth < upper, name


def test_reparameterizations_reproduce_their_priors() -> None:
    """Under a flat likelihood the posterior must equal the declared prior.

    Both the emission/absorption pair (net/cancellation coordinates) and the
    effective-width coordinates carry a change of variables; any Jacobian
    error shows up as a distorted prior here. Flux priors are declared at
    O(1) against errors of 1e6 so the data carry no information.
    """
    flat = halpha_spectrum(seed=12, noise=1.0e6)
    flux = TruncatedNormal(0.0, 1.0, low=0.0)
    emission = NoobLine("Ha", rest=HALPHA, unit="angstrom").flux(flux)
    absorption = NoobLine(
        "Ha_abs", rest=HALPHA, unit="angstrom", contribution="absorption"
    ).flux(flux)
    continuum = NoobContinuum.polynomial(0, lambda_0=LAMBDA_0).coefficient(0, 0.0)
    workspace = flat.prepare([emission, absorption], continuum=continuum)
    result = workspace.model().sample(draws=1000, tune=1000, chains=8)
    reparameterized = " ".join(result.sampling.reparameterizations)
    assert "flux pair Ha / Ha_abs" in reparameterized
    assert "Ha.fwhm: log effective width" in reparameterized
    assert result.sampling.verdict.converged

    width_prior = stats.uniform(np.log(10.0), np.log(700.0) - np.log(10.0))
    for line in ("Ha", "Ha_abs"):
        fluxes = result[line].flux.samples.reshape(-1)[::8]
        assert stats.kstest(fluxes, stats.halfnorm(scale=1.0).cdf).pvalue > 1e-3
        widths = np.log(result[line].fwhm.samples.reshape(-1)[::8])
        assert stats.kstest(widths, width_prior.cdf).pvalue > 1e-3


def test_correlated_noise_loo_terms() -> None:
    joined = NoobSpectrum.join(
        {"w": halpha_spectrum(seed=13), "r": halpha_spectrum(seed=14)}
    )
    spectrum = joined.with_noise({"r": Noise(correlation=AR1(Uniform()))})
    workspace = spectrum.prepare(
        [NoobLine("Ha", rest=HALPHA, unit="angstrom")], continuum=CONTINUUM
    )
    result = workspace.model().sample(draws=200, tune=300, chains=4)
    log_likelihood = result.idata["log_likelihood"].dataset
    posterior = result.idata["posterior"].dataset
    on_r = log_likelihood["pixel_frame"].values == "r"
    pointwise = log_likelihood["pixels"].values[0, 0, on_r]
    values = {
        name: np.asarray(array.values[0, 0])[None, None]
        for name, array in posterior.data_vars.items()
    }
    xp = numpy_namespace()
    frame = spectrum.frames["r"]
    mask = workspace.fitted["r"]
    resid = (
        np.asarray(frame.flux)[mask] - workspace.frame_model(xp, values, "r").total[0]
    )
    noise = workspace.frame_noise["r"]
    scale = float(posterior["frame.r.noise.scale"].values[0, 0])
    rho = float(posterior["frame.r.noise.rho"].values[0, 0])
    np.testing.assert_allclose(
        pointwise, noise.pointwise(xp, resid, scale, rho), rtol=1e-8
    )
    joint = log_likelihood["frames"].sel(frame="r").values[0, 0]
    assert joint == pytest.approx(float(noise.log_likelihood(xp, resid, scale, rho)))
    assert result.frames.loo is not None
