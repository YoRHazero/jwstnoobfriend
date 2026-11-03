"""Tests for prior value objects, domains and joint priors."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate, stats

from noobfriend.inference.spectrum._xp import numpy_namespace
from noobfriend.inference.spectrum.priors import (
    FRACTION,
    NONNEGATIVE,
    POSITIVE,
    REAL,
    JointNormal,
    LogUniform,
    Normal,
    TruncatedNormal,
    Uniform,
    check_value,
)

XP = numpy_namespace()


class TestResolve:
    """Omitted bounds take the parameter's domain edge (T9.10)."""

    def test_uniform_without_bounds_is_improper_on_positive(self) -> None:
        prior = Uniform().resolve(POSITIVE)
        assert prior.bounds() == (0.0, np.inf)
        assert not prior.is_proper

    def test_half_normal_from_truncated_normal(self) -> None:
        prior = TruncatedNormal(0.0, 2.0, low=0.0).resolve(NONNEGATIVE)
        assert prior.bounds() == (0.0, np.inf)
        x = np.linspace(0.1, 5.0, 7)
        expected = stats.halfnorm(scale=2.0).logpdf(x)
        np.testing.assert_allclose(prior.log_density(XP, x), expected)

    def test_truncated_normal_far_tail_normalizes(self) -> None:
        prior = TruncatedNormal(0.0, 1.0, low=8.0, high=12.0).resolve(REAL)
        total, _ = integrate.quad(lambda v: np.exp(prior.log_density(XP, v)), 8.0, 12.0)
        assert total == pytest.approx(1.0, rel=1e-8)

    def test_log_uniform_normalizes_when_proper(self) -> None:
        prior = LogUniform(10.0, 700.0).resolve(POSITIVE)
        total, _ = integrate.quad(
            lambda v: np.exp(prior.log_density(XP, v)), 10.0, 700.0
        )
        assert total == pytest.approx(1.0)

    @pytest.mark.parametrize(
        ("prior", "domain"),
        [
            (Normal(0.0, 1.0), POSITIVE),
            (Uniform(-5.0, 10.0), POSITIVE),
            (LogUniform(), REAL),
            (Uniform(0.0, 2.0), FRACTION),
        ],
    )
    def test_rejects_mass_outside_domain(self, prior: object, domain: object) -> None:
        with pytest.raises(ValueError):
            prior.resolve(domain)  # type: ignore[attr-defined]


class TestCheckValue:
    """The ``float | Prior`` contract (T4b.1)."""

    def test_fixed_value_coerced(self) -> None:
        assert check_value(3, POSITIVE, "fwhm") == 3.0

    @pytest.mark.parametrize("value", [True, (1.0, 2.0), "1"])
    def test_rejects_non_values(self, value: object) -> None:
        with pytest.raises(TypeError):
            check_value(value, REAL, "x")

    def test_rejects_fixed_value_outside_domain(self) -> None:
        with pytest.raises(ValueError, match="fraction"):
            check_value(0.0, FRACTION, "fraction")


class TestDomainTransforms:
    """Unconstrained coordinates used by joint priors (T7.4)."""

    @pytest.mark.parametrize(
        ("domain", "values"),
        [
            (POSITIVE, [1e-3, 2.0, 5e4]),
            (FRACTION, [0.01, 0.5, 0.999]),
            (REAL, [-3.0, 4.0]),
        ],
    )
    def test_round_trip(self, domain: object, values: list[float]) -> None:
        array = np.asarray(values)
        unconstrained = domain.to_unconstrained(XP, array)  # type: ignore[attr-defined]
        restored = domain.from_unconstrained(XP, unconstrained)  # type: ignore[attr-defined]
        np.testing.assert_allclose(restored, array)


class TestJointNormal:
    """Marginalization and member checks."""

    def make(self) -> JointNormal:
        covariance = np.array([[1.0, 0.3, 0.1], [0.3, 2.0, 0.4], [0.1, 0.4, 3.0]])
        return JointNormal(
            names=("a.flux", "a.fwhm", "a.delta_v"),
            domains=(NONNEGATIVE, POSITIVE, REAL),
            mean=np.array([1.0, 2.0, 3.0]),
            covariance=covariance,
        )

    def test_marginal_is_sub_block(self) -> None:
        joint = self.make()
        mean, covariance = joint.marginal(["a.delta_v", "a.flux"])
        np.testing.assert_allclose(mean, [3.0, 1.0])
        np.testing.assert_allclose(covariance, [[3.0, 0.1], [0.1, 1.0]])

    def test_member_rejects_other_domain(self) -> None:
        member = self.make().member("a.fwhm")
        assert member.resolve(POSITIVE) is member
        with pytest.raises(ValueError, match="positive"):
            member.resolve(REAL)

    def test_rejects_indefinite_covariance(self) -> None:
        with pytest.raises(ValueError, match="positive definite"):
            JointNormal(
                ("x", "y"),
                (REAL, REAL),
                np.zeros(2),
                np.array([[1.0, 2.0], [2.0, 1.0]]),
            )
