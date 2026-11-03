"""Tests for ParameterPosterior arithmetic and the convergence verdict."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("arviz")

from noobfriend.inference.spectrum.mcmc import ParameterPosterior  # noqa: E402
from noobfriend.inference.spectrum.mcmc._decompose import select_draws  # noqa: E402
from noobfriend.inference.spectrum.mcmc._diagnostics import diagnose  # noqa: E402

RNG = np.random.default_rng(0)


class TestParameterPosterior:
    """Draw-aligned arithmetic (T11a.5)."""

    def test_operations_are_drawwise(self) -> None:
        a = ParameterPosterior("a", RNG.normal(3.0, 0.1, size=(4, 50)))
        b = ParameterPosterior("b", RNG.normal(2.0, 0.1, size=(4, 50)))
        ratio = a / b
        np.testing.assert_allclose(ratio.samples, a.samples / b.samples)
        assert ratio.name == "(a / b)"
        np.testing.assert_allclose((2 * a - 1).samples, 2 * a.samples - 1)
        np.testing.assert_allclose((a**2).samples, a.samples**2)
        np.testing.assert_allclose(a.map(np.log10).samples, np.log10(a.samples))

    def test_misaligned_draws_are_rejected(self) -> None:
        a = ParameterPosterior("a", np.ones((4, 50)))
        b = ParameterPosterior("b", np.ones((2, 50)))
        with pytest.raises(ValueError, match="align"):
            _ = a + b

    def test_hdi_brackets_the_median(self) -> None:
        a = ParameterPosterior("a", RNG.normal(0.0, 1.0, size=(4, 2000)))
        lower, upper = a.hdi(0.94)
        assert lower < a.median < upper
        assert upper - lower == pytest.approx(2 * 1.88, rel=0.1)


def test_select_draws_is_even_and_deterministic() -> None:
    chain, draw = select_draws(4, 1000, 100)
    assert chain.size == draw.size == 100
    assert set(np.unique(chain)) == {0, 1, 2, 3}
    np.testing.assert_array_equal(
        draw[:25], np.linspace(0, 999, 25).round().astype(int)
    )
    all_chain, _ = select_draws(2, 10, 10_000)
    assert all_chain.size == 20


class TestVerdict:
    """Fixed convergence thresholds (T9.5)."""

    def tree(self, posterior: dict[str, np.ndarray], diverging: int = 0) -> object:
        import xarray as xr

        chains, draws = next(iter(posterior.values())).shape
        flags = np.zeros((chains, draws), dtype=bool)
        flags.flat[:diverging] = True
        return xr.DataTree.from_dict(
            {
                "posterior": xr.Dataset(
                    {k: (("chain", "draw"), v) for k, v in posterior.items()}
                ),
                "sample_stats": xr.Dataset(
                    {
                        "diverging": (("chain", "draw"), flags),
                        "energy": (("chain", "draw"), RNG.normal(size=(chains, draws))),
                    }
                ),
            }
        )

    def test_well_mixed_chains_converge(self) -> None:
        _, verdict = diagnose(
            self.tree({"x": RNG.normal(size=(8, 1000)), "c": np.ones((8, 1000))})
        )
        assert verdict.converged
        assert verdict.failures == ()

    def test_every_failure_is_listed(self) -> None:
        stuck = RNG.normal(size=(8, 200)) + np.arange(8)[:, None]
        _, verdict = diagnose(self.tree({"x": stuck}, diverging=3))
        assert not verdict.converged
        text = " ".join(verdict.failures)
        for fragment in ("3 divergent", "R-hat", "bulk ESS", "tail ESS"):
            assert fragment in text
