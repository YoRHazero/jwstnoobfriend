"""Prepared line workspaces."""

from noobfriend.inference.spectrum.workspace.continuum import (
    ContinuumSharing,
    ContinuumSpec,
)
from noobfriend.inference.spectrum.workspace.handles import LineHandle
from noobfriend.inference.spectrum.workspace.workspace import NoobFitWorkspace

__all__ = [
    "ContinuumSharing",
    "ContinuumSpec",
    "LineHandle",
    "NoobFitWorkspace",
]
