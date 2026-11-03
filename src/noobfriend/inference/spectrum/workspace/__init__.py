"""Fit workspaces: the resolved plan between declarations and fit backends.

``NoobSpectrum.prepare`` calls :func:`prepare` and returns a
:class:`NoobFitWorkspace`: fitted pixels and noise per frame, line handles
(:class:`LineHandle`) and the :class:`ContinuumHandle`, and the single
:class:`ParameterTable` with every prior written out. Backends evaluate the
mean model through :meth:`NoobFitWorkspace.physical` and
:meth:`NoobFitWorkspace.frame_model`.
"""

from noobfriend.inference.spectrum.workspace._expressions import (
    Const,
    Expression,
    Product,
    Ref,
    Scale,
    Sum,
)
from noobfriend.inference.spectrum.workspace._handles import ContinuumHandle, LineHandle
from noobfriend.inference.spectrum.workspace._model import FrameModel, ModelView
from noobfriend.inference.spectrum.workspace._protocols import (
    ContinuumLike,
    FrameLike,
    KernelLike,
    LineLike,
    ParameterRuleLike,
    SpectrumLike,
)
from noobfriend.inference.spectrum.workspace._resolve import DataScales
from noobfriend.inference.spectrum.workspace._summary import format_rows
from noobfriend.inference.spectrum.workspace._table import (
    JointBlock,
    Parameter,
    ParameterKind,
    ParameterTable,
    Quantity,
)
from noobfriend.inference.spectrum.workspace._workspace import NoobFitWorkspace, prepare

__all__ = [
    "Const",
    "ContinuumHandle",
    "ContinuumLike",
    "DataScales",
    "Expression",
    "FrameLike",
    "FrameModel",
    "JointBlock",
    "KernelLike",
    "LineHandle",
    "LineLike",
    "ModelView",
    "NoobFitWorkspace",
    "Parameter",
    "ParameterKind",
    "ParameterRuleLike",
    "ParameterTable",
    "Product",
    "Quantity",
    "Ref",
    "Scale",
    "SpectrumLike",
    "Sum",
    "format_rows",
    "prepare",
]
