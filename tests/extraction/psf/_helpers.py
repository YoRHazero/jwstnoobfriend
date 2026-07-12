"""Shared fake WCS and synthetic-frame builder for the PSF extraction tests."""

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord


class LinearWCS:
    """A minimal APE-14 WCS: pixel (x, y) -> sky via a linear scale."""

    def __init__(self, ra0: float = 53.0, dec0: float = -27.0, scale: float = 1e-4):
        self.ra0, self.dec0, self.scale = ra0, dec0, scale

    def pixel_to_world(self, x: np.ndarray, y: np.ndarray) -> SkyCoord:
        return SkyCoord(
            (self.ra0 + self.scale * np.asarray(x)) * u.deg,
            (self.dec0 + self.scale * np.asarray(y)) * u.deg,
        )


class TransformWCS:
    """A ``TransformWCS``-shaped fake: ``available_frames`` + ``get_transform``.

    Mirrors how the package consumes ``NooBook.wcs`` (a compiled ``NoobWCS``):
    ``get_transform("detector", "world")`` returns a ``(x, y) -> (ra, dec)``
    transform in degrees, with no APE-14 ``pixel_to_world``.
    """

    available_frames = ("detector", "world")

    def __init__(self, ra0: float = 53.0, dec0: float = -27.0, scale: float = 1e-4):
        self.ra0, self.dec0, self.scale = ra0, dec0, scale

    def get_transform(self, from_frame: str, to_frame: str):
        assert (from_frame, to_frame) == ("detector", "world")

        def transform(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return (
                self.ra0 + self.scale * np.asarray(x, dtype=float),
                self.dec0 + self.scale * np.asarray(y, dtype=float),
            )

        return transform


def make_frame(
    positions: list[tuple],
    *,
    peak: float = 300.0,
    sigma: float = 2.0,
    shape: tuple[int, int] = (120, 120),
    noise: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Synthetic frame; each position is ``(y, x)`` or ``(y, x, sigma_x)``."""
    rng = np.random.default_rng(seed)
    img = rng.normal(0.0, noise, size=shape)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    for pos in positions:
        y, x = pos[0], pos[1]
        sx = pos[2] if len(pos) > 2 else sigma
        img += peak * np.exp(-0.5 * (((yy - y) / sigma) ** 2 + ((xx - x) / sx) ** 2))
    return img
