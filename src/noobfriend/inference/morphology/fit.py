"""High-level fit configuration and preview initialisation."""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import product
from typing import Any

from noobfriend.inference.morphology.backend import NumpyroNUTSBackend
from noobfriend.inference.morphology.data import NoobImageSet
from noobfriend.inference.morphology.gate import PSFGate, PSFGateResult
from noobfriend.inference.morphology.introspection import scene_parameters
from noobfriend.inference.morphology.parameters import (
    LogUniform,
    Normal,
    Parameter,
    TruncNormal,
    Uniform,
)
from noobfriend.inference.morphology.render import render_scene
from noobfriend.inference.morphology.result import FitResult
from noobfriend.inference.morphology.scene import Scene


@dataclass(frozen=True)
class FitConfig:
    """Top-level morphology fit configuration."""

    sampler: NumpyroNUTSBackend = NumpyroNUTSBackend()
    initializer: "AnchorSlopeMultiStartInitializer | None" = None


@dataclass(frozen=True)
class MorphologyFitter:
    """High-level fitter bound to an explicit fit configuration."""

    config: FitConfig = FitConfig()

    def fit(
        self,
        images: NoobImageSet,
        scene: Scene,
        *,
        gate: PSFGateResult | None = None,
        fixed_params: dict[str, float] | None = None,
        init_params: dict[str, float] | None = None,
        seed: int = 0,
    ) -> FitResult:
        """Fit ``scene`` to ``images`` using the configured sampler."""
        sampler_init = dict(init_params or {})
        preview = None
        if self.config.initializer is not None:
            if gate is None:
                gate = PSFGate().fit(images)
            preview = self.config.initializer.preview(images, scene, gate)
            if preview.selected:
                seeded = dict(preview.selected[0].params)
                seeded.update(sampler_init)
                sampler_init = seeded
        result = self.config.sampler.fit(
            images,
            scene,
            fixed_params=fixed_params,
            init_params=sampler_init,
            seed=seed,
        )
        if preview is None:
            return result
        metadata = dict(result.metadata)
        metadata["initialization"] = preview
        return result.with_metadata(metadata)


@dataclass(frozen=True)
class PreviewCandidate:
    """One deterministic multi-start preview candidate."""

    name: str
    params: dict[str, float]
    chi2: float
    n_pixels: int
    stage: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def reduced_chi2(self) -> float:
        """Return ``chi2 / n_pixels`` when pixels were scored."""
        if self.n_pixels <= 0:
            return float("nan")
        return self.chi2 / self.n_pixels


@dataclass(frozen=True)
class MultiStartPreviewResult:
    """Deterministic preview output used to seed posterior sampling."""

    anchor_band: str
    candidates: tuple[PreviewCandidate, ...]
    selected: tuple[PreviewCandidate, ...]
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def best(self) -> PreviewCandidate:
        """Return the best selected candidate."""
        if not self.selected:
            raise ValueError("multi-start preview did not select any candidate.")
        return self.selected[0]


@dataclass(frozen=True)
class AnchorSlopeMultiStartConfig:
    """Anchor-band, wavelength-slope multi-start configuration."""

    anchor_band: str | None = None
    anchor_fractions: tuple[float, ...] = (0.05, 0.15, 0.30, 0.55, 0.80, 0.95)
    fraction_slopes: tuple[float, ...] = (0.0, 4.0, 7.0)
    r_eff_values: tuple[float, ...] = (0.04, 0.08, 0.16, 0.30)
    q_values: tuple[float, ...] = (0.45, 0.70, 1.0)
    theta_values: tuple[float, ...] = (0.0, 45.0, 90.0, 135.0)
    offset_rho_values: tuple[float, ...] = (0.0, 0.5)
    offset_phi_values: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)
    n_start: float = 1.8
    selected_count: int = 1
    anchor_keep: int = 48
    preview_workers: int = 1
    pilot_workers: int = 1
    fraction_floor: float = 0.001
    fraction_ceiling: float = 0.98
    point_x_name: str = "point_x"
    point_y_name: str = "point_y"
    total_flux_prefix: str = "total_flux"
    host_fraction_prefix: str = "host_fraction"
    background_prefix: str = "background"
    point_flux_prefix: str = "point_flux"
    host_r_eff_name: str = "host_r_eff"
    host_n_name: str = "host_n"
    host_q_name: str = "host_q"
    host_theta_name: str = "host_theta"
    host_offset_rho_name: str = "host_offset_rho"
    host_offset_phi_name: str = "host_offset_phi"


@dataclass(frozen=True)
class AnchorSlopeMultiStartInitializer:
    """Deterministic anchor-band + wavelength-slope preview initializer."""

    config: AnchorSlopeMultiStartConfig = AnchorSlopeMultiStartConfig()

    def preview(
        self, images: NoobImageSet, scene: Scene, gate: PSFGateResult
    ) -> MultiStartPreviewResult:
        """Score deterministic starts and return the best seeds."""
        start = time.perf_counter()
        images = images if isinstance(images, NoobImageSet) else NoobImageSet(images)
        anchor_band = self.config.anchor_band or self._choose_anchor_band(images, gate)
        anchor_images = NoobImageSet([images[anchor_band]])
        base = self._base_params(images, scene, gate)

        anchor_start = time.perf_counter()
        anchor_candidates = self._score_anchor_grid(anchor_images, scene, base, gate)
        anchor_candidates = sorted(anchor_candidates, key=lambda item: item.chi2)
        anchor_elapsed = time.perf_counter() - anchor_start

        slope_start = time.perf_counter()
        keep = anchor_candidates[: max(1, self.config.anchor_keep)]
        slope_candidates = self._score_slope_grid(
            images, scene, keep, gate, anchor_band
        )
        slope_candidates = sorted(slope_candidates, key=lambda item: item.chi2)
        slope_elapsed = time.perf_counter() - slope_start

        selected = tuple(slope_candidates[: max(1, self.config.selected_count)])
        return MultiStartPreviewResult(
            anchor_band=anchor_band,
            candidates=tuple(anchor_candidates + slope_candidates),
            selected=selected,
            timings={
                "anchor_preview_s": anchor_elapsed,
                "slope_preview_s": slope_elapsed,
                "total_preview_s": time.perf_counter() - start,
            },
        )

    def initial_params(
        self, images: NoobImageSet, scene: Scene, gate: PSFGateResult
    ) -> dict[str, float]:
        """Return only the best initial-parameter dictionary."""
        return dict(self.preview(images, scene, gate).best.params)

    def _choose_anchor_band(self, images: NoobImageSet, gate: PSFGateResult) -> str:
        scores: dict[str, float] = {}
        for image in images:
            flux = gate.point_params.get(
                f"{self.config.point_flux_prefix}_{image.name}", 0.0
            )
            noise = float(image.error[image.mask].mean()) if image.mask.any() else 1.0
            scores[image.name] = flux / max(noise, 1e-30)
        return max(scores, key=scores.get)

    def _base_params(
        self, images: NoobImageSet, scene: Scene, gate: PSFGateResult
    ) -> dict[str, float]:
        params: dict[str, float] = {
            self.config.point_x_name: gate.point_params.get("point_x", 0.0),
            self.config.point_y_name: gate.point_params.get("point_y", 0.0),
            self.config.host_n_name: self.config.n_start,
        }
        for image in images:
            band = image.name
            background = gate.point_params.get(
                f"{self.config.background_prefix}_{band}"
            )
            if background is None:
                background = (
                    float(image.data[image.mask].mean()) if image.mask.any() else 0.0
                )
            params[f"{self.config.background_prefix}_{band}"] = float(background)
            point_flux = self._point_flux_seed(image, gate)
            params[f"{self.config.total_flux_prefix}_{band}"] = point_flux
            params[f"{self.config.host_fraction_prefix}_{band}"] = (
                self.config.fraction_floor
            )

        for param in scene_parameters(scene):
            params.setdefault(param.name, _parameter_start(param))
        return params

    def _score_anchor_grid(
        self,
        images: NoobImageSet,
        scene: Scene,
        base: dict[str, float],
        gate: PSFGateResult,
    ) -> list[PreviewCandidate]:
        combos = list(
            product(
                self.config.anchor_fractions,
                self.config.r_eff_values,
                self.config.q_values,
                self.config.theta_values,
                self.config.offset_rho_values,
                self.config.offset_phi_values,
            )
        )

        def build(
            combo: tuple[float, float, float, float, float, float],
        ) -> PreviewCandidate:
            fraction, r_eff, q, theta, offset_rho, offset_phi = combo
            params = dict(base)
            params[self.config.host_r_eff_name] = float(r_eff)
            params[self.config.host_q_name] = float(q)
            params[self.config.host_theta_name] = float(theta)
            params[self.config.host_offset_rho_name] = float(offset_rho)
            params[self.config.host_offset_phi_name] = float(offset_phi)
            for image in images:
                self._set_band_fraction(params, image.name, fraction, gate)
            return self._score(
                images,
                scene,
                params,
                stage="anchor",
                metadata={
                    "anchor_fraction": float(fraction),
                    "r_eff": float(r_eff),
                    "q": float(q),
                    "theta": float(theta),
                    "offset_rho": float(offset_rho),
                    "offset_phi": float(offset_phi),
                },
            )

        return _map_candidates(build, combos, workers=self.config.preview_workers)

    def _score_slope_grid(
        self,
        images: NoobImageSet,
        scene: Scene,
        seeds: list[PreviewCandidate],
        gate: PSFGateResult,
        anchor_band: str,
    ) -> list[PreviewCandidate]:
        anchor_image = images[anchor_band]
        anchor_wave = anchor_image.wavelength_um
        combos = list(product(seeds, self.config.fraction_slopes))

        def build(combo: tuple[PreviewCandidate, float]) -> PreviewCandidate:
            seed, slope = combo
            params = dict(seed.params)
            anchor_fraction = seed.metadata["anchor_fraction"]
            for image in images:
                fraction = _fraction_from_slope(
                    anchor_fraction=anchor_fraction,
                    anchor_wavelength=anchor_wave,
                    band_wavelength=image.wavelength_um,
                    slope=slope,
                    floor=self.config.fraction_floor,
                    ceiling=self.config.fraction_ceiling,
                )
                self._set_band_fraction(params, image.name, fraction, gate)
            return self._score(
                images,
                scene,
                params,
                stage="slope",
                metadata={**seed.metadata, "fraction_slope": float(slope)},
            )

        return _map_candidates(build, combos, workers=self.config.preview_workers)

    def _set_band_fraction(
        self,
        params: dict[str, float],
        band: str,
        fraction: float,
        gate: PSFGateResult,
    ) -> None:
        clipped = _clip_fraction(
            fraction,
            floor=self.config.fraction_floor,
            ceiling=self.config.fraction_ceiling,
        )
        params[f"{self.config.host_fraction_prefix}_{band}"] = clipped
        point_flux = gate.point_params.get(f"{self.config.point_flux_prefix}_{band}")
        if point_flux is not None:
            total = float(point_flux) / max(1.0 - clipped, 0.02)
            params[f"{self.config.total_flux_prefix}_{band}"] = max(total, 1e-12)

    def _point_flux_seed(self, image: Any, gate: PSFGateResult) -> float:
        point_flux = gate.point_params.get(
            f"{self.config.point_flux_prefix}_{image.name}"
        )
        if point_flux is not None:
            return max(float(point_flux), 1e-12)
        background = gate.point_params.get(
            f"{self.config.background_prefix}_{image.name}", 0.0
        )
        positive = (image.data - background)[image.mask]
        return max(float(positive[positive > 0].sum()), 1e-12)

    def _score(
        self,
        images: NoobImageSet,
        scene: Scene,
        params: dict[str, float],
        *,
        stage: str,
        metadata: dict[str, Any],
    ) -> PreviewCandidate:
        rendered = render_scene(images, scene, params)
        chi2 = 0.0
        n_pixels = 0
        for image in images:
            use = image.mask
            resid = (image.data - rendered.total[image.name]) / image.error
            chi2 += float((resid[use] ** 2).sum())
            n_pixels += int(use.sum())
        name = f"{stage}:{len(metadata)}:{chi2:.6g}"
        return PreviewCandidate(
            name=name,
            params=params,
            chi2=chi2,
            n_pixels=n_pixels,
            stage=stage,
            metadata=metadata,
        )


def _map_candidates(
    func: Any, items: list[Any], *, workers: int
) -> list[PreviewCandidate]:
    """Map preview scoring with optional thread parallelism."""
    if workers <= 1 or len(items) <= 1:
        return [func(item) for item in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(func, items))


def _fraction_from_slope(
    *,
    anchor_fraction: float,
    anchor_wavelength: float | None,
    band_wavelength: float | None,
    slope: float,
    floor: float,
    ceiling: float,
) -> float:
    """Map host fraction from anchor band using a logit wavelength slope."""
    if anchor_wavelength is None or band_wavelength is None or anchor_wavelength <= 0:
        return _clip_fraction(anchor_fraction, floor=floor, ceiling=ceiling)
    anchor = _clip_fraction(anchor_fraction, floor=floor, ceiling=ceiling)
    logit = math.log(anchor / (1.0 - anchor))
    logit += slope * math.log(max(band_wavelength, 1e-12) / anchor_wavelength)
    fraction = 1.0 / (1.0 + math.exp(-logit))
    return _clip_fraction(fraction, floor=floor, ceiling=ceiling)


def _clip_fraction(value: float, *, floor: float, ceiling: float) -> float:
    """Clip a candidate host fraction into the configured interval."""
    if floor < 0 or ceiling > 1 or floor >= ceiling:
        raise ValueError(
            "fraction floor/ceiling must satisfy 0 <= floor < ceiling <= 1."
        )
    return min(max(float(value), floor), ceiling)


def _parameter_start(param: Parameter) -> float:
    """Return a deterministic start for a declarative parameter."""
    if param.init is not None:
        return float(param.init)
    prior = param.prior
    if isinstance(prior, Uniform):
        return 0.5 * (prior.low + prior.high)
    if isinstance(prior, LogUniform):
        return math.sqrt(prior.low * prior.high)
    if isinstance(prior, Normal):
        return float(prior.loc)
    if isinstance(prior, TruncNormal):
        return min(max(float(prior.loc), prior.low), prior.high)
    return 0.0
