"""Tests for the morphology parameter algebra (priors, Param, expressions).

Pure structure: prior validation, spec canonicalisation, identity-based linking,
and the constant / add / scale expression algebra used for ordering constraints.
"""

from __future__ import annotations

import math

import pytest

from noobfriend.inference.morphology import (
    LogUniform,
    Normal,
    Param,
    TruncNormal,
    Uniform,
)
from noobfriend.inference.morphology._param import (
    Add,
    Constant,
    collect_params,
    parse_spec,
)


class TestPriors:
    """Validation of the declarative prior value objects."""

    def test_uniform_requires_low_below_high(self) -> None:
        with pytest.raises(ValueError, match="low < high"):
            Uniform(5.0, 1.0)

    def test_uniform_requires_finite(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            Uniform(1.0, math.inf)

    def test_normal_requires_positive_scale(self) -> None:
        with pytest.raises(ValueError, match="scale > 0"):
            Normal(0.0, 0.0)

    def test_truncnormal_requires_low_below_high(self) -> None:
        with pytest.raises(ValueError, match="low < high"):
            TruncNormal(1.5, 1.0, 8.0, 0.3)

    def test_loguniform_requires_positive_low(self) -> None:
        with pytest.raises(ValueError, match="0 < low < high"):
            LogUniform(0.0, 1.0)

    def test_repr_is_compact(self) -> None:
        assert repr(Uniform(1.0, 6.0)) == "Uniform(1, 6)"


class TestParseSpec:
    """Canonicalisation of a user spec into a parameter node."""

    def test_none_is_anonymous_default(self) -> None:
        node = parse_spec(None, where="x")
        assert isinstance(node, Param)
        assert node.name is None and node.prior is None

    def test_number_is_fixed_constant(self) -> None:
        node = parse_spec(4, where="x")
        assert isinstance(node, Constant) and node.value == 4.0

    def test_interval_is_uniform(self) -> None:
        node = parse_spec((1.0, 6.0), where="x")
        assert isinstance(node, Param) and node.prior == Uniform(1.0, 6.0)

    def test_bool_rejected(self) -> None:
        with pytest.raises(ValueError, match="bool"):
            parse_spec(True, where="x")

    def test_bad_interval_rejected(self) -> None:
        with pytest.raises(ValueError, match="expected"):
            parse_spec((1.0, 2.0, 3.0), where="x")

    def test_prior_becomes_anonymous_param(self) -> None:
        node = parse_spec(Normal(0.0, 1.0), where="x")
        assert isinstance(node, Param) and node.prior == Normal(0.0, 1.0)

    def test_param_passes_through(self) -> None:
        p = Param("n", Uniform(1.0, 6.0))
        assert parse_spec(p, where="x") is p

    def test_two_intervals_are_independent(self) -> None:
        a = parse_spec((1.0, 6.0), where="a")
        b = parse_spec((1.0, 6.0), where="b")
        assert a is not b  # structural equality never links


class TestAlgebra:
    """The add / subtract / scale expression algebra."""

    def test_sum_collects_both_params_in_order(self) -> None:
        r_b = Param("r_b", Uniform(0.02, 0.3))
        gap = Param("gap", LogUniform(0.01, 0.5))
        expr = r_b + gap
        assert isinstance(expr, Add)
        assert collect_params(expr) == (r_b, gap)
        assert expr.evaluate({r_b: 0.1, gap: 0.05}) == pytest.approx(0.15)

    def test_scale_by_constant(self) -> None:
        r_b = Param("r_b", Uniform(0.02, 0.3))
        assert (r_b * 1.5).evaluate({r_b: 0.1}) == pytest.approx(0.15)
        assert (2.0 * r_b).evaluate({r_b: 0.1}) == pytest.approx(0.2)

    def test_subtract_and_negate(self) -> None:
        a = Param("a", Uniform(0.0, 1.0))
        b = Param("b", Uniform(0.0, 1.0))
        assert (a - b).evaluate({a: 0.3, b: 0.1}) == pytest.approx(0.2)
        assert (-a).evaluate({a: 0.3}) == pytest.approx(-0.3)

    def test_add_a_number(self) -> None:
        a = Param("a", Uniform(0.0, 1.0))
        assert (a + 0.1).evaluate({a: 0.3}) == pytest.approx(0.4)

    def test_param_times_param_rejected(self) -> None:
        a = Param("a", Uniform(0.0, 1.0))
        b = Param("b", Uniform(0.0, 1.0))
        with pytest.raises(TypeError):
            _ = a * b

    def test_repeated_param_deduped(self) -> None:
        a = Param("a", Uniform(0.0, 1.0))
        expr = a + a
        assert collect_params(expr) == (a,)
        assert expr.evaluate({a: 0.1}) == pytest.approx(0.2)

    def test_evaluate_missing_value_raises(self) -> None:
        a = Param("a", Uniform(0.0, 1.0))
        with pytest.raises(KeyError):
            a.evaluate({})
