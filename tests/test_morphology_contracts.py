"""Tests for the rewritten morphology public contracts."""

from __future__ import annotations

import builtins
import importlib.util

import numpy as np
import pytest

from noobfriend.inference.morphology import (
    AnchorSlopeMultiStartConfig,
    AnchorSlopeMultiStartInitializer,
    Background,
    EllipticalOffsetFrom,
    FixedCenter,
    FitDiagnostics,
    FitResult,
    FreeCenter,
    KernelPSF,
    MissingSamplerBackendError,
    MorphologyWorkflow,
    MorphologyWorkflowConfig,
    NoobImage,
    NoobImageSet,
    PerBand,
    Point,
    PSFGate,
    PSFGateConfig,
    PSFGateResult,
    Scene,
    Sersic,
    SersicShape,
    TotalFractionFlux,
    TruncNormal,
    Uniform,
    WingSNRBaselineSelector,
    compare_psis_loo,
    inject_scene,
    psis_loo,
    render_scene,
    scene_parameters,
)
from noobfriend.inference.morphology.backend import require_numpyro_backend
from noobfriend.inference.morphology.components import elliptical_offset_xy
from noobfriend.inference.morphology.parameters import Parameter


def _image(name: str = "f444w") -> NoobImage:
    data = np.zeros((21, 21), dtype=float)
    err = np.ones_like(data)
    psf = KernelPSF(np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]))
    return NoobImage(
        name=name,
        data=data,
        error=err,
        psf=psf,
        pixel_scale=0.05,
        wavelength_um=4.44,
    )


def test_noob_image_set_keeps_band_state_together() -> None:
    """NoobImageSet stores complete per-band image objects."""
    f356 = _image("f356w")
    f444 = _image("f444w").with_error_rescale(2.0, reason="drizzle")
    images = NoobImageSet([f356, f444])

    assert images.names == ("f356w", "f444w")
    assert images["f444w"].raw_error is not None
    assert images["f444w"].meta["error_rescale_reason"] == "drizzle"
    with pytest.raises(KeyError):
        images["f200w"]


def test_scene_uses_explicit_component_list() -> None:
    """Scene composition is an ordered list with unique component names."""
    point = Point("agn", center=FixedCenter(), flux=1.0)
    host = Sersic(
        "host",
        center=FixedCenter(),
        flux=1.0,
        shape=SersicShape(r_eff=0.1, n=1.5, q=0.7, theta=30.0),
    )
    scene = Scene([point, host])

    assert [component.name for component in scene] == ["agn", "host"]
    with pytest.raises(ValueError):
        Scene([point, point])


def test_elliptical_offset_center_stays_inside_host_ellipse() -> None:
    """EllipticalOffsetFrom resolves host centre from rho/phi/r_eff/q/theta."""
    point = Point("agn", center=FixedCenter(), flux=1.0)
    host = Sersic(
        "host",
        center=EllipticalOffsetFrom(
            reference="agn",
            r_eff="host_r_eff",
            q="host_q",
            theta="host_theta",
            rho="host_offset_rho",
            phi="host_offset_phi",
        ),
        flux=1.0,
        shape=SersicShape(r_eff="host_r_eff", n=1.5, q="host_q", theta="host_theta"),
    )
    scene = Scene([point, host])
    values = {
        "host_r_eff": 0.2,
        "host_q": 0.5,
        "host_theta": 30.0,
        "host_offset_rho": 0.8,
        "host_offset_phi": 90.0,
    }

    host_x, host_y = scene.component_center("host", values)
    expected = elliptical_offset_xy(
        r_eff=0.2, q=0.5, theta_deg=30.0, rho=0.8, phi_deg=90.0
    )
    assert (host_x, host_y) == pytest.approx(expected)


def test_render_scene_and_inject_total_fraction_flux() -> None:
    """Renderer exposes component images and injection into backgrounds."""
    images = NoobImageSet([_image("f444w")])
    split = TotalFractionFlux(
        total=PerBand(prefix="total_flux"),
        fraction=PerBand(prefix="host_fraction"),
    )
    point = Point("agn", center=FixedCenter(), flux=split.part("point"))
    host = Sersic(
        "host",
        center=FixedCenter(),
        flux=split.part("host"),
        shape=SersicShape(r_eff=0.15, n=1.0, q=1.0, theta=0.0),
    )
    scene = Scene([point, host])
    params = {"total_flux_f444w": 10.0, "host_fraction_f444w": 0.2}

    rendered = render_scene(images, scene, params)
    total = rendered.total["f444w"]
    assert np.sum(rendered.components["f444w"]["agn"]) == pytest.approx(8.0)
    assert np.sum(rendered.components["f444w"]["host"]) == pytest.approx(2.0)
    assert np.sum(total) == pytest.approx(10.0)

    injected = inject_scene(images, scene, params)
    assert np.sum(injected["f444w"].data) == pytest.approx(10.0)


def test_background_component_is_rendered_per_band() -> None:
    """Background is an explicit scene component, not a hidden likelihood term."""
    images = NoobImageSet([_image("f444w")])
    scene = Scene([Background("sky", level=PerBand(prefix="background"))])
    rendered = render_scene(images, scene, {"background_f444w": 0.3})

    assert np.all(rendered.components["f444w"]["sky"] == 0.3)
    assert np.sum(rendered.total["f444w"]) == pytest.approx(0.3 * 21 * 21)


def test_baseline_selector_uses_positive_wing_snr() -> None:
    """Default baseline selector is modular and driven by PSF gate output."""
    point_scene = Scene([Point("agn", center=FixedCenter(), flux=1.0)])
    sersic_scene = Scene(
        [
            Sersic(
                "galaxy",
                center=FixedCenter(),
                flux=1.0,
                shape=SersicShape(r_eff=0.1, n=1.5, q=0.8, theta=0.0),
            )
        ]
    )
    selector = WingSNRBaselineSelector()

    point_gate = PSFGateResult(point_params={}, wing_snr_by_band={"f444w": -9.0})
    assert (
        selector.choose(
            point_gate, point_scene=point_scene, sersic_scene=sersic_scene
        ).kind
        == "point"
    )

    extended_gate = PSFGateResult(point_params={}, wing_snr_by_band={"f444w": 7.0})
    assert (
        selector.choose(
            extended_gate, point_scene=point_scene, sersic_scene=sersic_scene
        ).kind
        == "sersic"
    )


def test_psf_gate_recovers_point_source_parameters() -> None:
    """Deterministic PSF gate fits shared centre, flux, and background."""
    image = _image("f444w")
    scene = Scene([Point("agn", center=FixedCenter(0.05, -0.03), flux=10.0)])
    rendered = render_scene(NoobImageSet([image]), scene, {})
    data = rendered.total["f444w"] + 0.2
    observed = NoobImage(
        "f444w",
        data,
        np.full_like(data, 0.05),
        image.psf,
        pixel_scale=image.pixel_scale,
        wavelength_um=image.wavelength_um,
    )

    gate = PSFGate(PSFGateConfig(max_offset_arcsec=0.12)).fit(NoobImageSet([observed]))

    assert gate.point_params["point_x"] == pytest.approx(0.05, abs=0.01)
    assert gate.point_params["point_y"] == pytest.approx(-0.03, abs=0.01)
    assert gate.point_params["point_flux_f444w"] == pytest.approx(10.0, rel=0.01)
    assert gate.point_params["background_f444w"] == pytest.approx(0.2, abs=0.01)
    assert abs(gate.wing_snr_by_band["f444w"]) < 1e-5
    assert gate.chi2_per_pixel is not None


def test_missing_numpyro_backend_message_uses_uv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lazy backend errors should not suggest pip commands."""
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *args: object, **kwargs: object) -> object:
        if name in {"jax", "numpyro"}:
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    with pytest.raises(MissingSamplerBackendError) as exc:
        require_numpyro_backend()
    message = str(exc.value)
    assert "uv add" in message
    assert "pip install" not in message


def test_import_does_not_require_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing morphology should not import optional sampler packages."""
    imported: list[str] = []
    real_import = builtins.__import__

    def tracking_import(name: str, *args: object, **kwargs: object) -> object:
        imported.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", tracking_import)
    __import__("noobfriend.inference.morphology")
    assert "jax" not in imported
    assert "numpyro" not in imported


def test_parameter_contract_keeps_priors_declarative() -> None:
    """Parameter objects store names and priors without importing a sampler."""
    param = Parameter("host_fraction_f444w", prior=Uniform(0.001, 0.9), init=0.05)
    assert param.name == "host_fraction_f444w"
    assert param.init == 0.05


def test_anchor_slope_multistart_recovers_known_fraction_seed() -> None:
    """Preview initializer uses anchor-band grid and wavelength slopes."""
    base = _image("f444w")
    f444 = NoobImage(
        name="f444w",
        data=np.zeros_like(base.data),
        error=np.full_like(base.data, 0.05),
        psf=base.psf,
        pixel_scale=base.pixel_scale,
        wavelength_um=4.44,
    )
    f356 = NoobImage(
        name="f356w",
        data=np.zeros_like(f444.data),
        error=np.full_like(f444.data, 0.05),
        psf=f444.psf,
        pixel_scale=f444.pixel_scale,
        wavelength_um=3.56,
    )
    images = NoobImageSet([f356, f444])
    split = TotalFractionFlux(
        total=PerBand(prefix="total_flux"),
        fraction=PerBand(prefix="host_fraction"),
    )
    scene = Scene(
        [
            Point(
                "agn",
                center=FreeCenter(
                    Parameter("point_x", prior=Uniform(-0.1, 0.1), init=0.0),
                    Parameter("point_y", prior=Uniform(-0.1, 0.1), init=0.0),
                ),
                flux=split.part("point"),
            ),
            Sersic(
                "host",
                center=EllipticalOffsetFrom(
                    reference="agn",
                    r_eff="host_r_eff",
                    q="host_q",
                    theta="host_theta",
                    rho="host_offset_rho",
                    phi="host_offset_phi",
                ),
                flux=split.part("host"),
                shape=SersicShape(
                    r_eff=Parameter("host_r_eff", prior=Uniform(0.03, 0.3), init=0.12),
                    n=Parameter("host_n", prior=Uniform(0.5, 4.0), init=1.8),
                    q=Parameter("host_q", prior=Uniform(0.3, 1.0), init=0.8),
                    theta=Parameter("host_theta", prior=Uniform(0.0, 180.0), init=35.0),
                ),
            ),
            Background("sky", level=PerBand(prefix="background")),
        ]
    )
    true_params = {
        "point_x": 0.0,
        "point_y": 0.0,
        "host_r_eff": 0.12,
        "host_n": 1.8,
        "host_q": 0.8,
        "host_theta": 35.0,
        "host_offset_rho": 0.0,
        "host_offset_phi": 0.0,
        "total_flux_f444w": 10.0,
        "host_fraction_f444w": 0.2,
        "background_f444w": 0.05,
        "total_flux_f356w": 6.0,
        "host_fraction_f356w": 0.0936492724630127,
        "background_f356w": 0.02,
    }
    injected = images.inject(render_scene(images, scene, true_params).total)
    gate = PSFGateResult(
        point_params={
            "point_x": 0.0,
            "point_y": 0.0,
            "point_flux_f444w": 8.0,
            "point_flux_f356w": 5.438104365221924,
            "background_f444w": 0.05,
            "background_f356w": 0.02,
        }
    )
    initializer = AnchorSlopeMultiStartInitializer(
        AnchorSlopeMultiStartConfig(
            anchor_fractions=(0.1, 0.2, 0.5),
            fraction_slopes=(0.0, 4.0),
            r_eff_values=(0.12,),
            q_values=(0.8,),
            theta_values=(35.0,),
            offset_rho_values=(0.0,),
            offset_phi_values=(0.0,),
            preview_workers=2,
        )
    )

    preview = initializer.preview(injected, scene, gate)
    best = preview.best.params

    assert preview.anchor_band == "f444w"
    assert preview.timings["total_preview_s"] >= 0.0
    assert best["host_fraction_f444w"] == pytest.approx(0.2)
    assert best["host_fraction_f356w"] == pytest.approx(0.0936492724630127)
    assert preview.best.reduced_chi2 == pytest.approx(0.0, abs=1e-8)
    assert {param.name for param in scene_parameters(scene)} >= {
        "point_x",
        "point_y",
        "host_r_eff",
        "host_n",
    }


def test_psis_loo_requires_recorded_log_likelihood() -> None:
    """Model comparison reports an invalid metric when log likelihood is absent."""
    result = FitResult(posterior={}, diagnostics=FitDiagnostics())

    metric = psis_loo(result)

    assert metric.valid is False
    assert metric.value is None
    assert "log_likelihood" in str(metric.reason)


def test_compare_psis_loo_returns_metric_details() -> None:
    """PSIS-LOO comparison consumes pointwise chain/draw log likelihood."""
    rng = np.random.default_rng(42)
    log_likelihood = -0.5 + 0.01 * rng.normal(size=(2, 40, 5))
    result = FitResult(
        posterior={},
        diagnostics=FitDiagnostics(),
        log_likelihood=log_likelihood,
    )

    comparison = compare_psis_loo({"point": result})
    metric = comparison.metrics["point"]

    assert metric.name == "psis_loo"
    assert metric.value is not None
    assert "se" in metric.details
    assert "high_pareto_k" in metric.details


def test_workflow_runs_gate_initialization_fits_loo_and_timings() -> None:
    """Workflow stitches gate, baseline, initialization, fits, LOO, and timings."""
    images, full_scene, point_scene, sersic_scene = _workflow_fixture()
    initializer = AnchorSlopeMultiStartInitializer(
        AnchorSlopeMultiStartConfig(
            anchor_fractions=(0.05, 0.2),
            fraction_slopes=(0.0,),
            r_eff_values=(0.12,),
            q_values=(0.8,),
            theta_values=(35.0,),
            offset_rho_values=(0.0,),
            offset_phi_values=(0.0,),
        )
    )
    workflow = MorphologyWorkflow(
        MorphologyWorkflowConfig(
            gate=PSFGate(PSFGateConfig(max_offset_arcsec=0.1)),
            initializer=initializer,
            sampler=_FakeSampler(),
            compute_loo=True,
        )
    )

    result = workflow.run(
        images,
        full_scene=full_scene,
        point_scene=point_scene,
        sersic_scene=sersic_scene,
        seed=3,
    )

    assert result.baseline.kind == "point"
    assert result.initialization is not None
    assert result.full_fit is not None
    assert result.baseline_fit is not None
    assert result.comparison is not None
    assert set(result.fits) == {"baseline:point", "full"}
    assert set(result.comparison.metrics) == {"baseline:point", "full"}
    assert result.full_fit.metadata["workflow_stage"] == "full"
    assert result.baseline_fit.metadata["workflow_stage"] == "baseline:point"
    assert result.full_fit.metadata["init_params"][
        "host_fraction_f444w"
    ] == pytest.approx(0.05)
    assert result.baseline_fit.metadata["init_params"]["point_flux_f444w"] > 0
    for key in [
        "psf_gate_s",
        "baseline_selection_s",
        "initialization_s",
        "initialization.anchor_preview_s",
        "initialization.slope_preview_s",
        "full_fit_s",
        "baseline_fit_s",
        "loo_comparison_s",
        "total_s",
    ]:
        assert key in result.timings
        assert result.timings[key] >= 0.0


def _workflow_fixture() -> tuple[NoobImageSet, Scene, Scene, Scene]:
    """Return a small two-band workflow fixture."""
    psf = KernelPSF(np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]))
    base_f444 = NoobImage(
        "f444w",
        np.zeros((17, 17)),
        np.full((17, 17), 0.05),
        psf,
        pixel_scale=0.05,
        wavelength_um=4.44,
    )
    base_f356 = NoobImage(
        "f356w",
        np.zeros((17, 17)),
        np.full((17, 17), 0.05),
        psf,
        pixel_scale=0.05,
        wavelength_um=3.56,
    )
    empty = NoobImageSet([base_f356, base_f444])
    split = TotalFractionFlux(
        total=PerBand(
            values={
                "f356w": Parameter(
                    "total_flux_f356w", prior=Uniform(0.1, 20.0), init=6.0
                ),
                "f444w": Parameter(
                    "total_flux_f444w", prior=Uniform(0.1, 20.0), init=10.0
                ),
            }
        ),
        fraction=PerBand(
            values={
                "f356w": Parameter(
                    "host_fraction_f356w", prior=Uniform(0.001, 0.95), init=0.05
                ),
                "f444w": Parameter(
                    "host_fraction_f444w", prior=Uniform(0.001, 0.95), init=0.05
                ),
            }
        ),
    )
    point = Point(
        "agn",
        center=FreeCenter(
            Parameter("point_x", prior=TruncNormal(0.0, 0.03, -0.1, 0.1), init=0.0),
            Parameter("point_y", prior=TruncNormal(0.0, 0.03, -0.1, 0.1), init=0.0),
        ),
        flux=split.part("point"),
    )
    host = Sersic(
        "host",
        center=EllipticalOffsetFrom(
            reference="agn",
            r_eff="host_r_eff",
            q="host_q",
            theta="host_theta",
            rho="host_offset_rho",
            phi="host_offset_phi",
        ),
        flux=split.part("host"),
        shape=SersicShape(
            r_eff=Parameter("host_r_eff", prior=Uniform(0.03, 0.3), init=0.12),
            n=Parameter("host_n", prior=Uniform(0.5, 4.0), init=1.8),
            q=Parameter("host_q", prior=Uniform(0.3, 1.0), init=0.8),
            theta=Parameter("host_theta", prior=Uniform(0.0, 180.0), init=35.0),
        ),
    )
    background = Background(
        "sky",
        level=PerBand(
            values={
                "f356w": Parameter(
                    "background_f356w", prior=Uniform(-1.0, 1.0), init=0.02
                ),
                "f444w": Parameter(
                    "background_f444w", prior=Uniform(-1.0, 1.0), init=0.05
                ),
            }
        ),
    )
    full_scene = Scene([point, host, background])
    point_scene = Scene(
        [
            Point(
                "agn",
                center=point.center,
                flux=PerBand(
                    values={
                        "f356w": Parameter(
                            "point_flux_f356w", prior=Uniform(0.1, 20.0), init=6.0
                        ),
                        "f444w": Parameter(
                            "point_flux_f444w", prior=Uniform(0.1, 20.0), init=10.0
                        ),
                    }
                ),
            ),
            background,
        ]
    )
    sersic_scene = Scene(
        [
            Sersic(
                "host",
                center=FixedCenter(),
                flux=PerBand(
                    values={
                        "f356w": Parameter(
                            "total_flux_f356w", prior=Uniform(0.1, 20.0), init=6.0
                        ),
                        "f444w": Parameter(
                            "total_flux_f444w", prior=Uniform(0.1, 20.0), init=10.0
                        ),
                    }
                ),
                shape=host.shape,
            ),
            background,
        ]
    )
    truth = {
        "point_x": 0.0,
        "point_y": 0.0,
        "host_r_eff": 0.12,
        "host_n": 1.8,
        "host_q": 0.8,
        "host_theta": 35.0,
        "host_offset_rho": 0.0,
        "host_offset_phi": 0.0,
        "total_flux_f444w": 10.0,
        "host_fraction_f444w": 0.05,
        "background_f444w": 0.05,
        "total_flux_f356w": 6.0,
        "host_fraction_f356w": 0.05,
        "background_f356w": 0.02,
    }
    images = empty.inject(render_scene(empty, full_scene, truth).total)
    return images, full_scene, point_scene, sersic_scene


class _FakeSampler:
    """Small sampler test double for workflow orchestration."""

    def fit(
        self,
        images: NoobImageSet,
        scene: Scene,
        *,
        fixed_params: dict[str, float] | None = None,
        init_params: dict[str, float] | None = None,
        seed: int = 0,
    ) -> FitResult:
        """Return deterministic fake posterior output."""
        _ = fixed_params
        n_obs = int(sum(image.mask.sum() for image in images))
        _ = seed
        draw_axis = np.linspace(-1.0, 1.0, 80).reshape(2, 40)
        obs_axis = np.arange(n_obs, dtype=float)
        log_likelihood = -0.5 + 0.01 * draw_axis[:, :, None]
        log_likelihood = log_likelihood + 1e-6 * obs_axis[None, None, :]
        return FitResult(
            posterior={
                param.name: np.zeros((2, 40)) for param in scene_parameters(scene)
            },
            diagnostics=FitDiagnostics(divergences=0),
            log_likelihood=log_likelihood,
            metadata={
                "component_count": len(scene),
                "init_params": dict(init_params or {}),
            },
        )
