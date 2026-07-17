"""Tests for compiling prepared line workspaces."""

from __future__ import annotations

import numpy as np
import pytest

from noobfriend.inference.spectrum import NoobLine, NoobSpectrum
from noobfriend.inference.spectrum.workspace.compiler import (
    BASE_SHAPE,
    compile_line_graph,
    evaluate_expression,
    flux_bounds,
    line_center,
    line_fwhm_kms,
    pack_parameter_specs,
    profile_template,
)


def _spectrum() -> NoobSpectrum:
    obs = np.linspace(39000.0, 41000.0, 100)
    return NoobSpectrum(np.ones_like(obs), np.full_like(obs, 0.1), obs=obs, z=7.0)


def _local_spectrum() -> NoobSpectrum:
    obs = np.linspace(4900.0, 5100.0, 100)
    return NoobSpectrum(np.ones_like(obs), np.full_like(obs, 0.1), obs=obs)


def test_compile_line_graph_resolves_flux_ratio_and_locked_shape_rules() -> None:
    base = NoobLine("OIII5007", rest=5008.24, z=7.0).center(delta_v_kms=(-100.0, 100.0))
    derived = base.derive("OIII4959", rest=4960.30).flux(ratio=0.335)
    workspace = _spectrum().prepare({"base": base, "derived": derived})

    compiled = compile_line_graph(workspace)

    base_id = id(base)
    base_flux, derived_flux = compiled.flux_expressions
    assert base_flux.terms == {base_id: 1.0}
    assert set(derived_flux.terms) == {base_id}
    assert derived_flux.terms[base_id] == pytest.approx(0.335)
    assert flux_bounds(base) == (0.0, float("inf"))
    assert evaluate_expression(derived_flux, {base_id: 10.0}) == pytest.approx(3.35)

    base_center, derived_center = compiled.center_expressions
    assert base_center.terms == {base_id: 1.0}
    assert derived_center.terms == {base_id: 1.0}
    assert compiled.center_sources[base_id].lower == pytest.approx(-100.0)
    assert compiled.center_sources[base_id].upper == pytest.approx(100.0)

    base_fwhm, derived_fwhm = compiled.fwhm_expressions
    assert base_fwhm.terms == {base_id: 1.0}
    assert derived_fwhm.terms == {base_id: 1.0}
    assert compiled.fwhm_sources[base_id].lower == pytest.approx(10.0)
    assert compiled.fwhm_sources[base_id].upper == pytest.approx(700.0)


def test_compile_line_graph_uses_only_noobline_center_rules() -> None:
    emission = NoobLine("OIII5007", rest=5008.24, z=7.0)
    absorption = NoobLine(
        "OIII5007", rest=5008.24, z=7.0, contribution="absorption"
    ).center(delta_v_kms=(-250.0, 250.0))
    workspace = _spectrum().prepare({"em": emission, "abs": absorption})

    compiled = compile_line_graph(workspace)

    assert compiled.center_expressions[0].terms == {id(emission): 1.0}
    assert compiled.center_sources[id(emission)].lower == pytest.approx(-300.0)
    assert compiled.center_sources[id(emission)].upper == pytest.approx(300.0)
    assert compiled.center_expressions[1].terms == {id(absorption): 1.0}
    assert compiled.center_sources[id(absorption)].lower == pytest.approx(-250.0)
    assert compiled.center_sources[id(absorption)].upper == pytest.approx(250.0)


def test_compiler_fixed_template_helpers_follow_rules() -> None:
    base = (
        NoobLine("OIII5007", rest=5008.24, z=7.0)
        .center(delta_v_kms=50.0)
        .fwhm(override=320.0)
    )
    derived = base.derive("OIII4959", rest=4960.30)
    workspace = _spectrum().prepare({"base": base, "derived": derived})
    compiled = compile_line_graph(workspace)

    base_handle = workspace.handle_for(base)
    derived_handle = workspace.handle_for(derived)

    assert line_center(base_handle, compiled.handle_by_line) == pytest.approx(
        base_handle.observed_wavelength * (1.0 + 50.0 / 299792.458)
    )
    assert line_center(derived_handle, compiled.handle_by_line) == pytest.approx(
        derived_handle.observed_wavelength * (1.0 + 50.0 / 299792.458)
    )
    assert line_fwhm_kms(base_handle, compiled.handle_by_line) == pytest.approx(320.0)
    assert line_fwhm_kms(derived_handle, compiled.handle_by_line) == pytest.approx(
        320.0
    )


def test_pack_parameter_specs_returns_optimizer_arrays() -> None:
    line = NoobLine("OIII5007", rest=5008.24, z=7.0, component="broad")
    workspace = _spectrum().prepare([line])
    compiled = compile_line_graph(workspace)

    lower, upper, initial, scale = pack_parameter_specs(
        tuple(compiled.center_sources.values()), tuple(compiled.fwhm_sources.values())
    )

    assert lower.tolist() == pytest.approx([-300.0, 800.0])
    assert upper.tolist() == pytest.approx([300.0, 5000.0])
    assert initial.tolist() == pytest.approx([0.0, 1200.0])
    assert np.all(scale > 0.0)


def test_profile_template_supports_declared_profiles() -> None:
    wavelength = np.linspace(4500.0, 5500.0, 10001)
    center = 5000.0

    for profile in ("gaussian", "lorentzian", "exponential"):
        handle = (
            _local_spectrum()
            .prepare([NoobLine("line", obs=center, profile=profile)])
            .handles[0]
        )
        template = profile_template(
            wavelength,
            handle=handle,
            center=center,
            fwhm_kms=200.0,
            resolving_power=None,
        )

        assert np.all(np.isfinite(template))
        assert np.all(template >= 0.0)
        assert wavelength[np.argmax(template)] == pytest.approx(center)
        assert np.trapezoid(template, wavelength) == pytest.approx(1.0, rel=1e-2)


def test_profile_template_rejects_non_gaussian_lsf_convolution() -> None:
    wavelength = np.linspace(4900.0, 5100.0, 4001)
    handle = (
        _local_spectrum()
        .prepare([NoobLine("line", obs=5000.0, profile="lorentzian")])
        .handles[0]
    )

    with pytest.raises(NotImplementedError, match="gaussian"):
        profile_template(
            wavelength,
            handle=handle,
            center=5000.0,
            fwhm_kms=200.0,
            resolving_power=3000.0,
        )


def test_compile_line_graph_groups_shape_parameters_by_name() -> None:
    base = NoobLine("OIII5007", rest=5008.24, z=7.0)
    derived = base.derive("OIII4959", rest=4960.30).flux(ratio=0.335)
    workspace = _spectrum().prepare({"base": base, "derived": derived})

    compiled = compile_line_graph(workspace)

    assert tuple(compiled.shape_sources) == (BASE_SHAPE,)
    assert all(tuple(shapes) == (BASE_SHAPE,) for shapes in compiled.shape_expressions)
    base_shapes, derived_shapes = compiled.shape_expressions
    assert base_shapes[BASE_SHAPE] is compiled.fwhm_expressions[0]
    assert derived_shapes[BASE_SHAPE] is compiled.fwhm_expressions[1]
    assert compiled.shape_sources[BASE_SHAPE] is compiled.fwhm_sources


def test_kernel_shapes_compile_and_emg_matches_numerical_convolution() -> None:
    from scipy.signal import fftconvolve

    from noobfriend.inference.spectrum.line import kernels
    from noobfriend.inference.spectrum.workspace.compiler.expressions import (
        line_kernels_kms,
    )

    wavelength = np.linspace(4900.0, 5100.0, 8001)
    line = (
        NoobLine("b", obs=5000.0, component="broad")
        .fwhm(override=(800.0, 5000.0))
        .convolve(kernels.laplace, fwhm=(200.0, 8000.0))
    )
    spectrum = NoobSpectrum(
        np.ones_like(wavelength), np.full_like(wavelength, 0.1), obs=wavelength
    )
    compiled = compile_line_graph(spectrum.prepare([line]))

    shapes = compiled.shape_expressions[0]
    assert tuple(shapes) == (BASE_SHAPE, "laplace__fwhm", "laplace__fraction")
    spec = next(iter(shapes["laplace__fwhm"].specs.values()))
    assert spec.initial == pytest.approx(np.sqrt(200.0 * 8000.0))

    handle = compiled.workspace.handles[0]
    resolved = line_kernels_kms(handle, shapes)
    assert resolved == (("laplace", pytest.approx(np.sqrt(200.0 * 8000.0)), 1.0),)

    emg = profile_template(
        wavelength,
        handle=handle,
        center=5000.0,
        fwhm_kms=1500.0,
        resolving_power=None,
        kernels=(("laplace", 1000.0, 1.0),),
    )
    gauss = profile_template(
        wavelength, handle=handle, center=5000.0, fwhm_kms=1500.0, resolving_power=None
    )
    dv = float(np.mean(np.diff(wavelength))) / 5000.0 * 299792.458
    vgrid = np.arange(-12000, 12001) * dv
    kernel_values = kernels.laplace(vgrid, 1000.0) * dv
    brute = fftconvolve(gauss, kernel_values, mode="same")
    core = np.abs(wavelength - 5000.0) < 60.0
    assert np.max(np.abs(emg[core] - brute[core])) / np.max(brute) < 2e-3

    folded = profile_template(
        wavelength,
        handle=handle,
        center=5000.0,
        fwhm_kms=1500.0,
        resolving_power=None,
        kernels=(("gaussian", 900.0, 1.0),),
    )
    direct = profile_template(
        wavelength,
        handle=handle,
        center=5000.0,
        fwhm_kms=float(np.hypot(1500.0, 900.0)),
        resolving_power=None,
    )
    assert np.allclose(folded, direct)


def test_fraction_mixes_convolved_and_bare_branches() -> None:
    from noobfriend.inference.spectrum.line import kernels as _k

    wavelength = np.linspace(4900.0, 5100.0, 2001)
    line = NoobLine("b", obs=5000.0, component="broad").convolve(
        _k.laplace, fwhm=1000.0
    )
    handle = compile_line_graph(
        NoobSpectrum(
            np.ones_like(wavelength), np.full_like(wavelength, 0.1), obs=wavelength
        ).prepare([line])
    ).workspace.handles[0]
    common = dict(handle=handle, center=5000.0, fwhm_kms=1500.0, resolving_power=None)
    bare = profile_template(wavelength, **common)
    full = profile_template(wavelength, kernels=(("laplace", 1000.0, 1.0),), **common)
    mixed = profile_template(wavelength, kernels=(("laplace", 1000.0, 0.3),), **common)
    assert np.allclose(mixed, 0.7 * bare + 0.3 * full)
