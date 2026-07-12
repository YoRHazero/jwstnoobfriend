"""Compile a prepared workspace into reusable line parameter expressions."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from noobfriend.inference.spectrum.workspace.compiler.expressions import (
    FluxExpression,
    ParameterExpression,
    VariableSpec,
    center_expression,
    flux_expression,
    fwhm_expression,
    ordered_specs,
)

if TYPE_CHECKING:
    from noobfriend.inference.spectrum.workspace.handles import LineHandle
    from noobfriend.inference.spectrum.workspace.workspace import NoobFitWorkspace


@dataclass(frozen=True, slots=True)
class CompiledLineGraph:
    """Line rule expressions resolved against a prepared workspace."""

    workspace: NoobFitWorkspace
    handle_by_line: dict[int, LineHandle]
    flux_expressions: tuple[FluxExpression, ...]
    center_expressions: tuple[ParameterExpression, ...]
    center_sources: OrderedDict[int, VariableSpec]
    fwhm_expressions: tuple[ParameterExpression, ...]
    fwhm_sources: OrderedDict[int, VariableSpec]


def compile_line_graph(workspace: NoobFitWorkspace) -> CompiledLineGraph:
    """Compile workspace line rules without choosing a fit backend."""
    handle_by_line = {id(handle.line): handle for handle in workspace.handles}
    flux_expressions = tuple(flux_expression(handle.line, handle_by_line) for handle in workspace.handles)
    center_expressions = tuple(
        center_expression(handle, handle_by_line)
        for handle in workspace.handles
    )
    fwhm_expressions = tuple(fwhm_expression(handle, handle_by_line) for handle in workspace.handles)
    return CompiledLineGraph(
        workspace=workspace,
        handle_by_line=handle_by_line,
        flux_expressions=flux_expressions,
        center_expressions=center_expressions,
        center_sources=ordered_specs(center_expressions),
        fwhm_expressions=fwhm_expressions,
        fwhm_sources=ordered_specs(fwhm_expressions),
    )
