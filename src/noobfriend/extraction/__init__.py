"""Extraction of data from JWST products, organised by extraction semantics.

This package is a pure namespace: it re-exports nothing, because — like the
top-level ``noobfriend`` package — it has no sibling ``.py`` modules that define
public names, only subpackages (one per kind of extraction) plus the internal
``_wcs`` helper shared between them.

Import from the subpackage that owns the name::

    from noobfriend.extraction.cutout import Cutout
"""
