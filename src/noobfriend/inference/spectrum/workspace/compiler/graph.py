"""Compile a prepared workspace into reusable line parameter expressions."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from noobfriend.inference.spectrum.workspace.compiler.expressions import (
    BASE_SHAPE,
    FluxExpression,
    ParameterExpression,
    VariableSpec,
    center_expression,
    collect_shape_sources,
    flux_expression,
    ordered_specs,
    shape_expressions,
)

if TYPE_CHECKING:
    from noobfriend.inference.spectrum.workspace.handles import LineHandle
    from noobfriend.inference.spectrum.workspace.workspace import NoobFitWorkspace


@dataclass(frozen=True, slots=True)
class CompiledLineGraph:
    """Line rule expressions resolved against a prepared workspace.

    Shape parameters are grouped by name: ``shape_expressions`` holds one
    ordered ``{name: expression}`` mapping per handle (``BASE_SHAPE`` first),
    and ``shape_sources`` holds the independent sources of each shape name
    keyed by line identity. Expressions of one shape name only ever reference
    sources of the same name, so evaluation namespaces stay per-name.
    """

    workspace: NoobFitWorkspace
    handle_by_line: dict[int, LineHandle]
    flux_expressions: tuple[FluxExpression, ...]
    center_expressions: tuple[ParameterExpression, ...]
    center_sources: OrderedDict[int, VariableSpec]
    shape_expressions: tuple[OrderedDict[str, ParameterExpression], ...]
    shape_sources: OrderedDict[str, OrderedDict[int, VariableSpec]]

    @property
    def fwhm_expressions(self) -> tuple[ParameterExpression, ...]:
        """Base-width expressions per handle (the ``BASE_SHAPE`` entries)."""
        return tuple(shapes[BASE_SHAPE] for shapes in self.shape_expressions)

    @property
    def fwhm_sources(self) -> OrderedDict[int, VariableSpec]:
        """Independent base-width sources keyed by line identity."""
        return self.shape_sources[BASE_SHAPE]


def compile_line_graph(workspace: NoobFitWorkspace) -> CompiledLineGraph:
    """Compile workspace line rules without choosing a fit backend."""
    handle_by_line = {id(handle.line): handle for handle in workspace.handles}
    flux_expressions = tuple(
        flux_expression(handle.line, handle_by_line) for handle in workspace.handles
    )
    center_expressions = tuple(
        center_expression(handle, handle_by_line) for handle in workspace.handles
    )
    per_handle_shapes = tuple(
        shape_expressions(handle, handle_by_line) for handle in workspace.handles
    )
    return CompiledLineGraph(
        workspace=workspace,
        handle_by_line=handle_by_line,
        flux_expressions=flux_expressions,
        center_expressions=center_expressions,
        center_sources=ordered_specs(center_expressions),
        shape_expressions=per_handle_shapes,
        shape_sources=collect_shape_sources(per_handle_shapes),
    )
