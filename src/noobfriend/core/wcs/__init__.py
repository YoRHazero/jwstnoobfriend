"""Compiled WCS: gwcs / FITS-TAN pipelines on the noobase fast evaluator.

``from_gwcs`` (or ``from_fits_wcs``) compiles a WCS once into flat op-list
programs; the resulting :class:`NoobWCS` answers ``get_transform`` with
callables that are ~1000x faster per scalar call and several times faster
on full-frame grids than the astropy model tree, at machine precision.
"""

from ._compile import (
    UnsupportedTransformError,
    compile_fits_tan,
    compile_transform,
    concat_specs,
    grism_wavelength_model,
)
from ._correction import TangentCorrection, apply_correction_to_gwcs
from ._noobwcs import NoobWCS, from_fits_wcs, from_gwcs
from ._protocol import TransformWCS

__all__ = [
    "NoobWCS",
    "TangentCorrection",
    "TransformWCS",
    "UnsupportedTransformError",
    "apply_correction_to_gwcs",
    "compile_fits_tan",
    "compile_transform",
    "concat_specs",
    "from_fits_wcs",
    "from_gwcs",
    "grism_wavelength_model",
]
