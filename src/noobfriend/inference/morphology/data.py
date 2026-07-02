"""Image containers for morphology fitting.

``NoobImage`` is the unit consumed by the new morphology stack: one band, one
science array, one effective error map, one likelihood mask, one PSF provider,
and the metadata needed to render a shared arcsec-frame scene onto that image.
Data-preparation policies such as masking and correlated-noise rescaling happen
before construction; fitting code only sees the prepared arrays.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PSFProvider(Protocol):
    """Protocol consumed by morphology renderers for PSF kernels."""

    def kernel_for(self, image: "NoobImage", *, oversample: int = 1) -> np.ndarray:
        """Return a 2-D, odd-sized, sum-normalised kernel for ``image``."""


@dataclass(frozen=True)
class KernelPSF:
    """Simple PSF provider backed by a fixed kernel array."""

    kernel: np.ndarray

    def kernel_for(self, image: "NoobImage", *, oversample: int = 1) -> np.ndarray:
        """Return the fixed kernel, normalised to unit sum."""
        kernel = np.asarray(self.kernel, dtype=float)
        if kernel.ndim != 2:
            raise ValueError("PSF kernel must be a 2-D array.")
        if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
            raise ValueError("PSF kernel must have odd dimensions.")
        total = float(np.nansum(kernel))
        if not np.isfinite(total) or total <= 0:
            raise ValueError("PSF kernel must have positive finite sum.")
        return kernel / total


@dataclass(frozen=True)
class NoobImage:
    """A prepared single-band image for morphology decomposition.

    Parameters
    ----------
    name : str
        Stable band identifier, e.g. ``"f444w"``.
    data, error : numpy.ndarray
        2-D science image and effective Gaussian error map. ``error`` should
        already include any data-preparation rescaling for correlated noise.
    psf : PSFProvider or numpy.ndarray
        PSF provider from the extraction PSF module, or a fixed kernel array.
    mask : numpy.ndarray, optional
        Boolean likelihood mask where ``True`` means the pixel is used. When
        omitted, finite data/error pixels are used.
    wcs : object, optional
        WCS object kept as metadata for callers. Rendering currently uses the
        local arcsec grid from ``pixel_scale``.
    pixel_scale : float
        Arcsec per pixel in the local tangent-plane frame.
    wavelength_um : float, optional
        Central wavelength in microns.
    raw_error : numpy.ndarray, optional
        Original error map before rescaling, retained for provenance.
    """

    name: str
    data: np.ndarray
    error: np.ndarray
    psf: PSFProvider | np.ndarray
    mask: np.ndarray | None = None
    wcs: Any | None = None
    pixel_scale: float = 1.0
    wavelength_um: float | None = None
    raw_error: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate arrays and normalise simple PSF inputs."""
        data = np.asarray(self.data, dtype=float)
        error = np.asarray(self.error, dtype=float)
        if data.ndim != 2:
            raise ValueError("NoobImage.data must be 2-D.")
        if error.shape != data.shape:
            raise ValueError("NoobImage.error must match data shape.")
        if self.pixel_scale <= 0 or not np.isfinite(self.pixel_scale):
            raise ValueError("NoobImage.pixel_scale must be a positive finite value.")
        mask = self.mask
        if mask is None:
            mask_arr = np.isfinite(data) & np.isfinite(error) & (error > 0)
        else:
            mask_arr = np.asarray(mask, dtype=bool)
            if mask_arr.shape != data.shape:
                raise ValueError("NoobImage.mask must match data shape.")
            mask_arr = mask_arr & np.isfinite(data) & np.isfinite(error) & (error > 0)
        psf = self.psf if isinstance(self.psf, PSFProvider) else KernelPSF(self.psf)
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "error", error)
        object.__setattr__(self, "mask", mask_arr)
        object.__setattr__(self, "psf", psf)
        if self.raw_error is not None:
            raw = np.asarray(self.raw_error, dtype=float)
            if raw.shape != data.shape:
                raise ValueError("NoobImage.raw_error must match data shape.")
            object.__setattr__(self, "raw_error", raw)

    @property
    def shape(self) -> tuple[int, int]:
        """Return the native image shape."""
        return self.data.shape

    def arcsec_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(x, y)`` pixel-centre offsets in arcsec.

        The grid is centred on the array centre. ``x`` increases with column and
        ``y`` increases with row; callers should pass north-up cutouts when they
        need astronomical PA semantics.
        """
        rows, cols = np.indices(self.shape, dtype=float)
        cy = (self.shape[0] - 1) / 2.0
        cx = (self.shape[1] - 1) / 2.0
        return (cols - cx) * self.pixel_scale, (rows - cy) * self.pixel_scale

    def psf_kernel(self, *, oversample: int = 1) -> np.ndarray:
        """Return this image's PSF kernel."""
        return self.psf.kernel_for(self, oversample=oversample)

    def with_error_rescale(
        self, factor: float, *, reason: str | None = None
    ) -> "NoobImage":
        """Return a copy with ``error`` multiplied by ``factor``.

        The previous effective error is stored in ``raw_error`` when no raw map
        was already present.
        """
        if factor <= 0 or not np.isfinite(factor):
            raise ValueError("error rescale factor must be positive and finite.")
        meta = dict(self.meta)
        meta["error_rescale_factor"] = float(factor)
        if reason is not None:
            meta["error_rescale_reason"] = reason
        return replace(
            self,
            error=self.error * factor,
            raw_error=self.raw_error if self.raw_error is not None else self.error,
            meta=meta,
        )

    def inject(self, model: np.ndarray, *, copy: bool = True) -> "NoobImage":
        """Return a copy with ``model`` added to the science image."""
        arr = np.asarray(model, dtype=float)
        if arr.shape != self.shape:
            raise ValueError("injected model must match image shape.")
        data = np.array(self.data, copy=copy)
        data = data + arr
        return replace(self, data=data)


@dataclass(frozen=True)
class NoobImageSet(Sequence[NoobImage]):
    """Ordered collection of prepared morphology images."""

    images: tuple[NoobImage, ...]

    def __init__(self, images: Iterable[NoobImage]) -> None:
        """Create an image set from an iterable of ``NoobImage`` objects."""
        items = tuple(images)
        if not items:
            raise ValueError("NoobImageSet needs at least one image.")
        names = [img.name for img in items]
        if len(set(names)) != len(names):
            raise ValueError(f"NoobImageSet names must be unique, got {names}.")
        object.__setattr__(self, "images", items)

    def __len__(self) -> int:
        """Return the number of images."""
        return len(self.images)

    def __iter__(self) -> Iterator[NoobImage]:
        """Iterate over images in rendering order."""
        return iter(self.images)

    def __getitem__(self, key: int | str) -> NoobImage:
        """Return an image by integer index or band name."""
        if isinstance(key, str):
            for image in self.images:
                if image.name == key:
                    return image
            raise KeyError(key)
        return self.images[key]

    @property
    def names(self) -> tuple[str, ...]:
        """Return band names in rendering order."""
        return tuple(image.name for image in self.images)

    def inject(self, rendered: dict[str, np.ndarray]) -> "NoobImageSet":
        """Return a copy with per-band rendered arrays injected into data."""
        return NoobImageSet(
            image.inject(rendered[image.name]) if image.name in rendered else image
            for image in self.images
        )
