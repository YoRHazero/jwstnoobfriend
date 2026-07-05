"""WFSS (grism)-specific per-frame helpers.

Dispersed-trace masking and the per-(module, direction) sky-residual template
(downsampled grid, cross-frame combine, scalar fit and subtract) used by the
grism stage-2 chain.
"""

from noobfriend.reduction.grism._sky_template import (
    combine_sky_template,
    fit_template_scalar,
    sky_residual_grid,
    subtract_sky_template,
)
from noobfriend.reduction.grism._trace_mask import grism_trace_mask

__all__ = [
    "combine_sky_template",
    "fit_template_scalar",
    "grism_trace_mask",
    "sky_residual_grid",
    "subtract_sky_template",
]
