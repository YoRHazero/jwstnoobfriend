"""Workspace compilation helpers shared by fit backends."""

from noobfriend.inference.spectrum.workspace.compiler.expressions import (
    BASE_SHAPE,
    NORMAL_RANGE_SIGMAS,
    FluxExpression,
    ParameterExpression,
    ParameterOffsets,
    VariableSpec,
    clip_initial,
    collect_shape_sources,
    contribution_sign,
    evaluate_expression,
    flux_bounds,
    line_center,
    line_fwhm_kms,
    line_kernels_kms,
    pack_parameter_specs,
    prior_initial,
    prior_range,
    scale_prior,
    shape_expressions,
)
from noobfriend.inference.spectrum.workspace.compiler.graph import (
    CompiledLineGraph,
    compile_line_graph,
)
from noobfriend.inference.spectrum.workspace.compiler.profiles import (
    C_KMS,
    profile_template,
    profile_template_stack,
)
from noobfriend.inference.spectrum.workspace.compiler.symbolic import (
    symbolic_expression,
    symbolic_prior_logp,
    symbolic_profile,
)

__all__ = [
    "BASE_SHAPE",
    "C_KMS",
    "CompiledLineGraph",
    "FluxExpression",
    "ParameterExpression",
    "ParameterOffsets",
    "VariableSpec",
    "clip_initial",
    "collect_shape_sources",
    "compile_line_graph",
    "contribution_sign",
    "evaluate_expression",
    "flux_bounds",
    "line_center",
    "line_fwhm_kms",
    "NORMAL_RANGE_SIGMAS",
    "line_kernels_kms",
    "pack_parameter_specs",
    "prior_initial",
    "prior_range",
    "scale_prior",
    "profile_template",
    "profile_template_stack",
    "shape_expressions",
    "symbolic_expression",
    "symbolic_prior_logp",
    "symbolic_profile",
]
