"""Site naming shared by the model, the results and saved files.

Physical parameters keep their flat workspace names. Internal sampling
coordinates end in ``.latent`` (no physical name can, since component names
contain no dots), and each frame's likelihood is the observed site
``frame.<id>.flux``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from noobfriend.inference.spectrum.workspace import NoobFitWorkspace

LATENT = ".latent"


def latent(name: str) -> str:
    """Name of an internal site derived from ``name``."""
    return f"{name}{LATENT}"


def observation_site(frame_id: str) -> str:
    """Name of a frame's observed likelihood site."""
    return f"frame.{frame_id}.flux"


def site_frame(site: str) -> str:
    """Frame id of a library observation site, else the site name itself."""
    if site.startswith("frame.") and site.endswith(".flux"):
        return site[len("frame.") : -len(".flux")]
    return site


def physical_names(workspace: NoobFitWorkspace) -> tuple[str, ...]:
    """Flat names reported in the ``posterior`` group, in table order.

    Every line quantity (``center``, ``delta_v``, ``fwhm``, ``flux``, kernel
    parameters, fixed or not), every free or pooled row (relative line
    parameters only when free), and the fixed continuum and noise rows.
    """
    names = dict.fromkeys(workspace.parameters.quantities)
    line_names = {line.name for line in workspace.lines}
    for row in workspace.parameters.values():
        if row.kind == "fixed" and row.owner in line_names:
            continue
        names.setdefault(row.name)
    return tuple(names)
