"""Thin, pixel-space image-array utilities built on scikit-image / scipy.

A small namespace of reusable, mechanism-only image helpers with no
astronomical interpretation — callers add WCS and photometry on top. The first
entry point is :func:`segment`, which turns a 2-D image into an integer label
map; it is reused by both :mod:`noobfriend.extraction.photometry` (as an
aperture-growth ``label_map``) and the navigation layer.
"""

from noobfriend.core.imgutils._segment import segment

__all__ = ["segment"]
