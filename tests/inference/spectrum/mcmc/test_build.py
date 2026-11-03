"""Tests for the NumPyro model contract and reparameterization choices."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numpyro")

from noobfriend.inference.spectrum import (  # noqa: E402
    AR1,
    Autocorrelation,
    LogUniform,
    Noise,
    NoobContinuum,
    NoobLine,
    NoobSpectrum,
    Uniform,
)
from noobfriend.inference.spectrum.mcmc._build import build_numpyro_model  # noqa: E402
from noobfriend.inference.spectrum.mcmc._names import physical_names  # noqa: E402
from noobfriend.inference.spectrum.mcmc._run import (  # noqa: E402
    likelihood_sites,
    trace_model,
)

from .._helpers import HALPHA, Z, halpha_spectrum  # noqa: E402

CONTINUUM = NoobContinuum.polynomial(0, lambda_0=HALPHA * (1 + Z))


def ha(**kwargs: object) -> NoobLine:
    return NoobLine("Ha", rest=HALPHA, unit="angstrom", **kwargs)  # type: ignore[arg-type]


def test_sites_follow_the_contract() -> None:
    spectrum = NoobSpectrum.join(
        {
            "w": halpha_spectrum(seed=1),
            "a": halpha_spectrum(seed=2),
            "f": halpha_spectrum(seed=3),
        }
    ).with_noise(
        {
            "a": Noise(scale=Uniform(0.5, 2.0), correlation=AR1(Uniform())),
            "f": Noise(correlation=Autocorrelation((0.3,))),
        }
    )
    nii = ha().derive("NII", rest=6585.3).flux(ratio_to_base=0.3)
    workspace = spectrum.prepare([ha(), nii], continuum=CONTINUUM)
    model, _ = build_numpyro_model(workspace)
    trace = trace_model(model)
    deterministic = {
        name for name, site in trace.items() if site["type"] == "deterministic"
    }
    assert set(physical_names(workspace)) <= deterministic
    assert "NII.flux_ratio_to_base" not in physical_names(workspace)
    assert "NII.flux" in deterministic
    sites = likelihood_sites(trace)
    assert sites == ("frame.w.flux", "frame.a.flux", "frame.f.flux")
    kinds = {name: type(trace[name]["fn"]).__name__ for name in sites}
    assert kinds == {
        "frame.w.flux": "Normal",
        "frame.a.flux": "AR1Normal",
        "frame.f.flux": "MultivariateNormal",
    }
    internal = [
        name
        for name, site in trace.items()
        if site["type"] == "sample" and not site["is_observed"]
    ]
    assert all(name.endswith(".latent") for name in internal)


def test_effective_width_needs_lsf_and_free_width() -> None:
    lines = [ha(), ha().derive("Hb", rest=4862.7).fwhm(200.0)]
    with_lsf = halpha_spectrum().prepare(lines[:1], continuum=CONTINUUM)
    without = halpha_spectrum(resolving_power=None).prepare(
        lines[:1], continuum=CONTINUUM
    )
    assert dict(build_numpyro_model(with_lsf)[1].effective_fwhm) == pytest.approx(
        {"Ha": 299792.458 / 1600.0}
    )
    assert not build_numpyro_model(without)[1].effective_fwhm


class TestFluxPairs:
    """Pair condition of T9.4."""

    def pairs(self, absorption: NoobLine) -> tuple:
        workspace = halpha_spectrum().prepare([ha(), absorption], continuum=CONTINUUM)
        return build_numpyro_model(workspace)[1].flux_pairs

    def test_matching_pair(self) -> None:
        absorption = NoobLine(
            "Ha_abs", rest=HALPHA, unit="angstrom", contribution="absorption"
        )
        assert self.pairs(absorption) == (("Ha", "Ha_abs"),)

    @pytest.mark.parametrize(
        "absorption",
        [
            NoobLine(
                "Ha_abs",
                rest=HALPHA,
                unit="angstrom",
                contribution="absorption",
                component="broad",
            ),
            NoobLine(
                "Ha_abs",
                rest=HALPHA,
                unit="angstrom",
                contribution="absorption",
                profile="lorentzian",
            ),
            NoobLine(
                "Ha_abs", rest=HALPHA + 1, unit="angstrom", contribution="absorption"
            ),
            NoobLine(
                "Ha_abs", rest=HALPHA, unit="angstrom", contribution="absorption"
            ).flux(5.0),
            NoobLine(
                "Ha_abs", rest=HALPHA, unit="angstrom", contribution="absorption"
            ).flux(Uniform(0.0, 100.0)),
        ],
    )
    def test_no_pair(self, absorption: NoobLine) -> None:
        assert self.pairs(absorption) == ()


def test_missing_physical_site_is_reported_before_sampling() -> None:
    import numpyro
    import numpyro.distributions as dist

    workspace = halpha_spectrum().prepare([ha()], continuum=CONTINUUM)

    def incomplete() -> None:
        numpyro.sample("frame.frame_0.flux", dist.Normal(0.0, 1.0), obs=np.zeros(3))

    with pytest.raises(ValueError, match="lacks physical-parameter sites"):
        workspace.model().with_numpyro_model(incomplete).sample(
            draws=10, tune=10, chains=1
        )


def test_sample_validates_options() -> None:
    workspace = halpha_spectrum().prepare(
        [ha().fwhm(LogUniform(10.0, 700.0))], continuum=CONTINUUM
    )
    with pytest.raises(ValueError, match="target_accept"):
        workspace.model().sample(target_accept=1.0)
    with pytest.raises(ValueError, match="chains"):
        workspace.model().sample(chains=0)
