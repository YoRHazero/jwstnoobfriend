"""Tests for the morphology profile declarations.

Pure structure: axis canonicalisation, typed signatures, identity-based linking
across components, ordering expressions as geometry, and flux normalisation.
"""

from __future__ import annotations

import pytest

from noobfriend.inference.morphology import (
    Exponential,
    LogUniform,
    Param,
    PointSource,
    Sersic,
    Uniform,
)
from noobfriend.inference.morphology._param import Add, Constant
from noobfriend.inference.morphology._profile import FluxSpec


class TestConstruction:
    """Axis sets, defaults, fixing, and typed signatures."""

    def test_sersic_exposes_all_axes(self) -> None:
        bulge = Sersic("bulge")
        assert set(bulge.params) == {"x0", "y0", "r_eff", "n", "q", "theta"}

    def test_omitted_axis_is_anonymous_default(self) -> None:
        node = Sersic("bulge").params["x0"]
        assert isinstance(node, Param) and node.name is None and node.prior is None

    def test_number_fixes_an_axis(self) -> None:
        node = Sersic("bulge", n=4).params["n"]
        assert isinstance(node, Constant) and node.value == 4.0

    def test_interval_axis_is_uniform(self) -> None:
        node = Sersic("bulge", r_eff=(0.02, 0.3)).params["r_eff"]
        assert isinstance(node, Param) and node.prior == Uniform(0.02, 0.3)

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Sersic("")

    def test_pointsource_has_only_centre(self) -> None:
        assert set(PointSource("agn").params) == {"x0", "y0"}

    def test_unknown_axis_is_a_type_error(self) -> None:
        with pytest.raises(TypeError):
            PointSource("agn", n=4)  # type: ignore[call-arg]


class TestFreeParams:
    """Free-parameter collection and identity-based linking."""

    def test_bare_sersic_has_six_free_axes(self) -> None:
        params = Sersic("bulge").free_geometry_params()
        assert len(params) == 6
        assert len(set(id(p) for p in params)) == 6

    def test_fixing_drops_a_free_param(self) -> None:
        assert len(Sersic("bulge", n=4).free_geometry_params()) == 5

    def test_shared_param_links_components(self) -> None:
        x0 = Param("x0", Uniform(-0.2, 0.2))
        y0 = Param("y0", Uniform(-0.2, 0.2))
        bulge = Sersic("bulge", x0=x0, y0=y0)
        disk = Exponential("disk", x0=x0, y0=y0)
        assert x0 in bulge.free_geometry_params()
        assert x0 in disk.free_geometry_params()  # same object -> linked

    def test_ordering_expression_as_geometry(self) -> None:
        r_b = Param("r_b", Uniform(0.02, 0.3))
        gap = Param("gap", LogUniform(0.01, 0.5))
        disk = Exponential("disk", r_eff=r_b + gap)
        assert isinstance(disk.params["r_eff"], Add)
        free = disk.free_geometry_params()
        assert r_b in free and gap in free


class TestFlux:
    """Per-band flux normalisation."""

    def test_default_flux(self) -> None:
        flux = Sersic("bulge").flux
        assert flux == FluxSpec(template=None, per_band=None)

    def test_template_flux(self) -> None:
        flux = Sersic("bulge", flux=(1e-20, 1e-18)).flux
        assert flux.template == (1e-20, 1e-18) and flux.per_band is None

    def test_per_band_flux(self) -> None:
        flux = Sersic("bulge", flux={"F150W": (1e-20, 1e-18), "F444W": None}).flux
        assert flux.template is None
        assert dict(flux.per_band) == {"F150W": (1e-20, 1e-18), "F444W": None}

    def test_empty_flux_mapping_rejected(self) -> None:
        with pytest.raises(ValueError, match="flux mapping is empty"):
            Sersic("bulge", flux={})

    def test_malformed_flux_spec_rejected(self) -> None:
        with pytest.raises(ValueError, match="flux"):
            Sersic("bulge", flux="bright")  # type: ignore[arg-type]
