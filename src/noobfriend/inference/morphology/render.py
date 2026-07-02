"""Pure NumPy rendering utilities for morphology scenes."""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import fftconvolve

from noobfriend.inference.morphology.components import Background, Point, Sersic
from noobfriend.inference.morphology.data import NoobImageSet
from noobfriend.inference.morphology.parameters import resolve_flux
from noobfriend.inference.morphology.result import RenderedScene
from noobfriend.inference.morphology.scene import Scene


def render_scene(
    images: NoobImageSet,
    scene: Scene,
    params: dict[str, float],
    *,
    oversample: int = 1,
) -> RenderedScene:
    """Render ``scene`` onto all images in ``images``."""
    totals: dict[str, np.ndarray] = {}
    components: dict[str, dict[str, np.ndarray]] = {}
    for image in images:
        per_component: dict[str, np.ndarray] = {}
        total = np.zeros(image.shape, dtype=float)
        kernel = image.psf_kernel(oversample=oversample)
        for component in scene:
            if isinstance(component, Background):
                model = np.full(
                    image.shape, resolve_flux(component.level, image.name, params)
                )
            elif isinstance(component, Point):
                flux = resolve_flux(component.flux, image.name, params)
                model = _render_point(
                    image.shape,
                    image.pixel_scale,
                    component.center.xy(scene, params),
                    flux,
                )
                model = _convolve(model, kernel)
            elif isinstance(component, Sersic):
                if component.shape is None:
                    raise ValueError("Sersic component has no shape.")
                flux = resolve_flux(component.flux, image.name, params)
                model = _render_sersic(image, scene, component, params, flux)
                model = _convolve(model, kernel)
            else:
                raise TypeError(f"unsupported component type {type(component)!r}")
            per_component[component.name] = model
            total = total + model
        totals[image.name] = total
        components[image.name] = per_component
    return RenderedScene(total=totals, components=components)


def inject_scene(
    images: NoobImageSet,
    scene: Scene,
    params: dict[str, float],
    *,
    oversample: int = 1,
) -> NoobImageSet:
    """Render and inject a scene into an image set."""
    rendered = render_scene(images, scene, params, oversample=oversample)
    return images.inject(rendered.total)


def _render_point(
    shape: tuple[int, int],
    pixel_scale: float,
    center: tuple[float, float],
    flux: float,
) -> np.ndarray:
    """Render a subpixel point source as a bilinear delta image."""
    out = np.zeros(shape, dtype=float)
    cx = (shape[1] - 1) / 2.0 + center[0] / pixel_scale
    cy = (shape[0] - 1) / 2.0 + center[1] / pixel_scale
    x0 = math.floor(cx)
    y0 = math.floor(cy)
    dx = cx - x0
    dy = cy - y0
    for yy, wy in [(y0, 1.0 - dy), (y0 + 1, dy)]:
        if yy < 0 or yy >= shape[0]:
            continue
        for xx, wx in [(x0, 1.0 - dx), (x0 + 1, dx)]:
            if 0 <= xx < shape[1]:
                out[yy, xx] += flux * wy * wx
    return out


def _render_sersic(
    image: object,
    scene: Scene,
    component: Sersic,
    params: dict[str, float],
    flux: float,
) -> np.ndarray:
    """Render an unconvolved Sersic profile on the native grid."""
    x, y = image.arcsec_grid()
    x0, y0 = component.center.xy(scene, params)
    shape = component.shape.resolved(params)
    r_eff = max(shape["r_eff"], 1e-6)
    n = max(shape["n"], 0.05)
    q = np.clip(shape["q"], 1e-3, 1.0)
    theta = np.deg2rad(shape["theta"])
    dx = x - x0
    dy = y - y0
    xp = dx * np.cos(theta) + dy * np.sin(theta)
    yp = -dx * np.sin(theta) + dy * np.cos(theta)
    r_ell = np.sqrt(xp * xp + (yp / q) ** 2)
    bn = 2.0 * n - 1.0 / 3.0 + 4.0 / (405.0 * n)
    profile = np.exp(-bn * ((r_ell / r_eff) ** (1.0 / n) - 1.0))
    total = float(np.sum(profile))
    if not np.isfinite(total) or total <= 0:
        return np.zeros_like(profile)
    return profile * (flux / total)


def _convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve while preserving native shape."""
    return fftconvolve(image, kernel, mode="same")
