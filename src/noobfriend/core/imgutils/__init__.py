"""Thin, pixel-space image-array utilities built on scikit-image / scipy.

A small namespace of reusable, mechanism-only image helpers with no
astronomical interpretation — callers add WCS and photometry on top. The first
entry point is :func:`segment`, which turns a 2-D image into an integer label
map; it is reused by both :mod:`noobfriend.extraction.photometry` (as an
aperture-growth ``label_map``) and the navigation layer.

:func:`noise_autocovariance` and :func:`correlated_sum_variance` add the other
mechanism-only primitive: the variance of a weighted pixel sum under spatially
*correlated* noise, estimated from blank sky — the honest error bar for aperture
photometry on drizzled frames, where the independent-pixel formula underestimates.
"""

from noobfriend.core.imgutils._noise import (
    NoiseAutocovariance,
    correlated_sum_variance,
    noise_autocovariance,
)
from noobfriend.core.imgutils._segment import segment

__all__ = [
    "NoiseAutocovariance",
    "correlated_sum_variance",
    "noise_autocovariance",
    "segment",
]
