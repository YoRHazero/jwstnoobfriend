"""Deterministic point-source gate contracts and implementation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from noobfriend.inference.morphology.data import NoobImage, NoobImageSet
from noobfriend.inference.morphology.render import _convolve, _render_point


@dataclass(frozen=True)
class RadialResidual:
    """One radial residual diagnostic row."""

    band: str
    annulus: str
    r_lo: float
    r_hi: float
    snr: float
    mean_sigma: float
    npix: int


@dataclass(frozen=True)
class PSFGateResult:
    """Deterministic point-source gate output used for init and diagnostics."""

    point_params: dict[str, float]
    chi2: float | None = None
    n_pixels: int = 0
    radial_residuals: tuple[RadialResidual, ...] = ()
    wing_snr_by_band: dict[str, float] = field(default_factory=dict)

    @property
    def combined_positive_wing_snr(self) -> float:
        """Quadrature sum of positive wing SNR values."""
        return sum(max(v, 0.0) ** 2 for v in self.wing_snr_by_band.values()) ** 0.5

    @property
    def max_positive_wing_snr(self) -> float:
        """Maximum positive wing SNR."""
        return max((max(v, 0.0) for v in self.wing_snr_by_band.values()), default=0.0)

    @property
    def chi2_per_pixel(self) -> float | None:
        """Return ``chi2 / n_pixels`` when available."""
        if self.chi2 is None or self.n_pixels <= 0:
            return None
        return self.chi2 / self.n_pixels


@dataclass(frozen=True)
class PSFGateConfig:
    """Configuration for the deterministic point-source gate."""

    max_offset_arcsec: float = 0.15
    fit_background: bool = True
    annuli: tuple[tuple[str, float, float], ...] = (
        ("core", 0.0, 0.10),
        ("inner_wing", 0.10, 0.25),
        ("outer_wing", 0.25, 0.50),
    )
    wing_annulus: tuple[float, float] = (0.10, 0.50)
    max_nfev: int = 300


@dataclass(frozen=True)
class PSFGate:
    """Fit a shared-centre point-source model to a ``NoobImageSet``."""

    config: PSFGateConfig = PSFGateConfig()

    def fit(self, images: NoobImageSet) -> PSFGateResult:
        """Run the deterministic point-only gate."""
        if not isinstance(images, NoobImageSet):
            images = NoobImageSet(images)
        init = _initial_vector(images, self.config)
        lower, upper = _bounds(images, self.config)
        opt = least_squares(
            _residual_vector,
            init,
            args=(images, self.config),
            bounds=(lower, upper),
            max_nfev=self.config.max_nfev,
        )
        params = _vector_to_params(opt.x, images, self.config)
        chi2 = float(np.sum(opt.fun * opt.fun))
        n_pixels = int(opt.fun.size)
        radial, wing = _radial_diagnostics(images, params, self.config)
        return PSFGateResult(
            point_params=params,
            chi2=chi2,
            n_pixels=n_pixels,
            radial_residuals=tuple(radial),
            wing_snr_by_band=wing,
        )


def _initial_vector(images: NoobImageSet, config: PSFGateConfig) -> np.ndarray:
    """Return deterministic initial values for the point gate."""
    x0, y0 = _centroid_seed(images)
    x0 = float(np.clip(x0, -config.max_offset_arcsec, config.max_offset_arcsec))
    y0 = float(np.clip(y0, -config.max_offset_arcsec, config.max_offset_arcsec))
    values = [x0, y0]
    for image in images:
        mask = np.asarray(image.mask, dtype=bool)
        background = _background_seed(image)
        flux = float(np.nansum(np.maximum(image.data[mask] - background, 0.0)))
        values.append(max(flux, 1e-6))
        if config.fit_background:
            values.append(background)
    return np.asarray(values, dtype=float)


def _bounds(images: NoobImageSet, config: PSFGateConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return least-squares lower/upper bounds."""
    lo = [-config.max_offset_arcsec, -config.max_offset_arcsec]
    hi = [config.max_offset_arcsec, config.max_offset_arcsec]
    for _image in images:
        lo.append(0.0)
        hi.append(np.inf)
        if config.fit_background:
            lo.append(-np.inf)
            hi.append(np.inf)
    return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)


def _vector_to_params(
    vector: np.ndarray, images: NoobImageSet, config: PSFGateConfig
) -> dict[str, float]:
    """Convert a fit vector into named public gate parameters."""
    params = {"point_x": float(vector[0]), "point_y": float(vector[1])}
    index = 2
    for image in images:
        params[f"point_flux_{image.name}"] = float(vector[index])
        index += 1
        if config.fit_background:
            params[f"background_{image.name}"] = float(vector[index])
            index += 1
        else:
            params[f"background_{image.name}"] = 0.0
    return params


def _residual_vector(
    vector: np.ndarray, images: NoobImageSet, config: PSFGateConfig
) -> np.ndarray:
    """Return concatenated whitened residuals for least-squares fitting."""
    params = _vector_to_params(vector, images, config)
    parts = []
    for image in images:
        model = _point_model(image, params)
        good = np.asarray(image.mask, dtype=bool)
        parts.append(((image.data - model) / image.error)[good])
    return np.concatenate(parts)


def _point_model(image: NoobImage, params: dict[str, float]) -> np.ndarray:
    """Render the point-gate model for one image."""
    point = _render_point(
        image.shape,
        image.pixel_scale,
        (params["point_x"], params["point_y"]),
        params[f"point_flux_{image.name}"],
    )
    model = _convolve(point, image.psf_kernel())
    return model + params.get(f"background_{image.name}", 0.0)


def _centroid_seed(images: NoobImageSet) -> tuple[float, float]:
    """Return a positive-flux centroid seed in arcsec."""
    num_x = num_y = den = 0.0
    for image in images:
        mask = np.asarray(image.mask, dtype=bool)
        background = _background_seed(image)
        weight = np.maximum(image.data - background, 0.0) / np.maximum(image.error, 1e-30)
        weight = np.where(mask & np.isfinite(weight), weight, 0.0)
        x, y = image.arcsec_grid()
        num_x += float(np.sum(weight * x))
        num_y += float(np.sum(weight * y))
        den += float(np.sum(weight))
    if den <= 0 or not np.isfinite(den):
        return 0.0, 0.0
    return num_x / den, num_y / den


def _background_seed(image: NoobImage) -> float:
    """Return a robust scalar background seed."""
    mask = np.asarray(image.mask, dtype=bool)
    values = image.data[mask & np.isfinite(image.data)]
    if values.size == 0:
        return 0.0
    return float(np.nanmedian(values))


def _radial_diagnostics(
    images: NoobImageSet, params: dict[str, float], config: PSFGateConfig
) -> tuple[list[RadialResidual], dict[str, float]]:
    """Return per-band radial residual rows and signed wing SNR."""
    rows: list[RadialResidual] = []
    wings: dict[str, float] = {}
    for image in images:
        model = _point_model(image, params)
        sigma = (image.data - model) / image.error
        x, y = image.arcsec_grid()
        radius = np.hypot(x - params["point_x"], y - params["point_y"])
        good = np.asarray(image.mask, dtype=bool) & np.isfinite(sigma)
        for name, r_lo, r_hi in config.annuli:
            use = good & (radius >= r_lo) & (radius < r_hi)
            rows.append(_radial_row(image.name, name, r_lo, r_hi, sigma, use))
        wing_lo, wing_hi = config.wing_annulus
        wing_use = good & (radius >= wing_lo) & (radius < wing_hi)
        wings[image.name] = _snr(sigma, wing_use)
    return rows, wings


def _radial_row(
    band: str,
    name: str,
    r_lo: float,
    r_hi: float,
    sigma: np.ndarray,
    use: np.ndarray,
) -> RadialResidual:
    """Build one radial residual diagnostic row."""
    npix = int(np.count_nonzero(use))
    mean = float(np.nanmean(sigma[use])) if npix else float("nan")
    return RadialResidual(
        band=band,
        annulus=name,
        r_lo=float(r_lo),
        r_hi=float(r_hi),
        snr=_snr(sigma, use),
        mean_sigma=mean,
        npix=npix,
    )


def _snr(sigma: np.ndarray, use: np.ndarray) -> float:
    """Return signed summed residual SNR over selected pixels."""
    npix = int(np.count_nonzero(use))
    if npix == 0:
        return float("nan")
    return float(np.nansum(sigma[use]) / np.sqrt(npix))
