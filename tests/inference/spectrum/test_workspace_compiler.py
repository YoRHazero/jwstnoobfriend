"""Tests for compiling prepared line workspaces."""

from __future__ import annotations

import numpy as np
import pytest

from noobfriend.inference.spectrum import NoobLine, NoobSpectrum
from noobfriend.inference.spectrum.workspace.compiler import (
    BASE_SHAPE,
    C_KMS,
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


def test_delta_wavelength_center_rules_convert_in_the_line_unit() -> None:
    # The line declares its delta_wavelength offset in its own unit (um); the
    # spectrum is in angstrom. The compiled velocity must equal delta / center
    # with both in the line's unit, independent of the spectrum's unit.
    fixed = NoobLine("line", obs=0.5, unit="um").center(delta_wavelength=1e-4)
    bounded = NoobLine("line", obs=0.5, unit="um").center(
        delta_wavelength=(-2e-4, 1e-4)
    )
    workspace = _local_spectrum().prepare({"fixed": fixed, "bounded": bounded})

    compiled = compile_line_graph(workspace)

    expected = 1e-4 / 0.5 * C_KMS
    assert compiled.center_expressions[0].fixed == pytest.approx(expected)
    spec = compiled.center_sources[id(bounded)]
    assert spec.lower == pytest.approx(-2.0 * expected)
    assert spec.upper == pytest.approx(expected)

    fixed_handle = workspace.handle_for(fixed)
    assert line_center(fixed_handle, compiled.handle_by_line) == pytest.approx(
        fixed_handle.observed_wavelength * (1.0 + expected / C_KMS)
    )


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


def test_lorentzian_profile_with_lsf_matches_numerical_convolution() -> None:
    """Regression: lorentzian + resolving_power must be the exact Voigt profile.

    A lorentzian line seen through a gaussian LSF is a Voigt profile; pin the
    library's width bookkeeping (line FWHM -> lorentzian HWHM, instrumental
    FWHM -> gaussian sigma, both converted at the line centre) against a
    brute-force numerical convolution, and keep the quadrature shortcut
    visibly wrong.
    """
    from scipy.signal import fftconvolve

    from noobfriend.inference.spectrum.line import kernels

    wavelength = np.linspace(4900.0, 5100.0, 8001)
    center = 5000.0
    fwhm_kms = 300.0
    resolving_power = 1000.0
    instrumental = C_KMS / resolving_power
    handle = (
        _local_spectrum()
        .prepare([NoobLine("line", obs=center, profile="lorentzian")])
        .handles[0]
    )

    exact = profile_template(
        wavelength,
        handle=handle,
        center=center,
        fwhm_kms=fwhm_kms,
        resolving_power=resolving_power,
    )
    intrinsic = profile_template(
        wavelength,
        handle=handle,
        center=center,
        fwhm_kms=fwhm_kms,
        resolving_power=None,
    )
    dv = float(np.mean(np.diff(wavelength))) / center * C_KMS
    vgrid = np.arange(-12000, 12001) * dv
    lsf = kernels.gaussian(vgrid, instrumental) * dv
    brute = fftconvolve(intrinsic, lsf, mode="same")
    core = np.abs(wavelength - center) < 60.0

    assert np.all(np.isfinite(exact))
    assert np.max(np.abs(exact[core] - brute[core])) / np.max(brute) < 1e-3

    quadrature = profile_template(
        wavelength,
        handle=handle,
        center=center,
        fwhm_kms=float(np.hypot(fwhm_kms, instrumental)),
        resolving_power=None,
    )
    assert np.max(np.abs(quadrature[core] - brute[core])) / np.max(brute) > 0.05


def test_exponential_profile_with_lsf_matches_numerical_convolution() -> None:
    """Regression: exponential + resolving_power must be the exact Normal–Laplace.

    A Laplace line seen through a gaussian LSF is a Normal–Laplace, not a
    quadrature-widened Laplace; a silent quadrature fold here once forced
    consumers to hand-build the line as a gaussian base with a fully-convolved
    laplace kernel. Pin the closed form against a brute-force numerical
    convolution and against that equivalent kernel spelling, and keep the
    quadrature shortcut visibly wrong.
    """
    from scipy.signal import fftconvolve

    from noobfriend.inference.spectrum.line import kernels

    wavelength = np.linspace(4900.0, 5100.0, 8001)
    center = 5000.0
    fwhm_kms = 300.0
    resolving_power = 1000.0
    instrumental = C_KMS / resolving_power
    handle = (
        _local_spectrum()
        .prepare([NoobLine("line", obs=center, profile="exponential")])
        .handles[0]
    )

    exact = profile_template(
        wavelength,
        handle=handle,
        center=center,
        fwhm_kms=fwhm_kms,
        resolving_power=resolving_power,
    )
    intrinsic = profile_template(
        wavelength,
        handle=handle,
        center=center,
        fwhm_kms=fwhm_kms,
        resolving_power=None,
    )
    dv = float(np.mean(np.diff(wavelength))) / center * C_KMS
    vgrid = np.arange(-12000, 12001) * dv
    lsf = kernels.gaussian(vgrid, instrumental) * dv
    brute = fftconvolve(intrinsic, lsf, mode="same")
    core = np.abs(wavelength - center) < 60.0

    assert np.all(np.isfinite(exact))
    assert np.max(np.abs(exact[core] - brute[core])) / np.max(brute) < 1e-3
    assert np.trapezoid(exact, wavelength) == pytest.approx(1.0, rel=1e-3)

    # The equivalent hand-built spelling (gaussian LSF base, fully-convolved
    # laplace kernel) must be the identical closed form.
    gaussian_handle = (
        _local_spectrum().prepare([NoobLine("line", obs=center)]).handles[0]
    )
    spelled_out = profile_template(
        wavelength,
        handle=gaussian_handle,
        center=center,
        fwhm_kms=instrumental,
        resolving_power=None,
        kernels=(("laplace", fwhm_kms, 1.0),),
    )
    np.testing.assert_allclose(exact, spelled_out, rtol=1e-12, atol=1e-15)

    # The old quadrature fold is measurably wrong (>13% at the peak here);
    # a regression to it must fail loudly.
    quadrature = profile_template(
        wavelength,
        handle=handle,
        center=center,
        fwhm_kms=float(np.hypot(fwhm_kms, instrumental)),
        resolving_power=None,
    )
    assert np.max(np.abs(quadrature[core] - brute[core])) / np.max(brute) > 0.05


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


def test_derived_kernel_shapes_share_the_base_source() -> None:
    base = NoobLine("b", obs=5000.0, component="broad").convolve(
        "laplace", fwhm=(200.0, 8000.0)
    )
    derived = base.derive("b2")
    workspace = _local_spectrum().prepare({"base": base, "derived": derived})

    compiled = compile_line_graph(workspace)

    base_shapes, derived_shapes = compiled.shape_expressions
    assert base_shapes["laplace__fwhm"].terms == {id(base): 1.0}
    assert derived_shapes["laplace__fwhm"].terms == {id(base): 1.0}
    assert tuple(compiled.shape_sources["laplace__fwhm"]) == (id(base),)
    # The fixed default fraction stays a per-line constant.
    assert derived_shapes["laplace__fraction"].terms == {}
    assert derived_shapes["laplace__fraction"].fixed == pytest.approx(1.0)


def test_profile_template_stack_matches_scalar_loop() -> None:
    from noobfriend.inference.spectrum.workspace.compiler import (
        profile_template_stack,
    )

    wavelength = np.linspace(4900.0, 5100.0, 401)
    rng = np.random.default_rng(0)
    centers = 5000.0 + rng.uniform(-6.0, 6.0, size=5)
    fwhms = rng.uniform(150.0, 900.0, size=5)

    def gaussian_handle():
        return _local_spectrum().prepare([NoobLine("line", obs=5000.0)]).handles[0]

    def check(handle, *, resolving_power=None, kernels=()):
        stack = profile_template_stack(
            wavelength,
            handle=handle,
            centers=centers,
            fwhms_kms=fwhms,
            resolving_power=resolving_power,
            kernels=kernels,
        )
        for i in range(centers.size):
            scalar = profile_template(
                wavelength,
                handle=handle,
                center=centers[i],
                fwhm_kms=fwhms[i],
                resolving_power=resolving_power,
                kernels=tuple(
                    (kind, width[i], fraction[i]) for kind, width, fraction in kernels
                ),
            )
            np.testing.assert_allclose(stack[i], scalar, rtol=1e-12, atol=1e-15)

    check(gaussian_handle())
    check(gaussian_handle(), resolving_power=1600.0)

    # fraction[0] == 1.0 exercises the scalar prune vs batched no-prune branch.
    fractions = np.array([1.0, 0.8, 0.5, 0.3, 0.9])
    laplace = ("laplace", rng.uniform(200.0, 1500.0, size=5), fractions)
    check(gaussian_handle(), kernels=(laplace,))
    # A gaussian kernel quadrature-adds array widths -> exercises the array sqrt.
    gaussian = ("gaussian", rng.uniform(200.0, 1200.0, size=5), fractions)
    check(gaussian_handle(), kernels=(gaussian,))

    lorentzian = (
        _local_spectrum()
        .prepare([NoobLine("line", obs=5000.0, profile="lorentzian")])
        .handles[0]
    )
    check(lorentzian)
    # Exercises the Voigt LSF routing on the batched path.
    check(lorentzian, resolving_power=1000.0)

    exponential = (
        _local_spectrum()
        .prepare([NoobLine("line", obs=5000.0, profile="exponential")])
        .handles[0]
    )
    check(exponential)
    # Exercises the Normal–Laplace LSF routing on the batched path.
    check(exponential, resolving_power=1000.0)


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
