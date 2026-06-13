"""Declarative primitives shared by the env reader and the ``env`` CLI.

These are the small, dependency-free building blocks that both the runtime
settings model (:mod:`noobfriend.core.env.settings`) and the ``noobfriend env``
command group rely on, so the configuration layout is described in exactly one
place.
"""

from dataclasses import dataclass
from enum import Enum


class EnvGroup(str, Enum):
    """Logical section a configuration variable belongs to.

    The order of the members is the order in which sections are rendered into a
    ``.env`` file.
    """

    setup = "Setup"
    noob = "Noob"
    storage = "Storage"


@dataclass(frozen=True)
class EnvField:
    """A single configuration variable as seen by the ``env`` CLI.

    A flattened, render-friendly view of one :class:`NoobSettings` field,
    produced by :func:`noobfriend.core.env.env_fields`. Keeping this as plain
    data (rather than handing the CLI a live pydantic field) lets the renderer
    and checker stay decoupled from pydantic internals.

    Attributes
    ----------
    name : str
        The environment-variable name, e.g. ``"NOOB_SERVER"``.
    group : EnvGroup
        Section the variable is rendered under.
    comment : str
        Human-readable description, written as a comment above the variable.
    default : str or None
        Default value to pre-fill, or ``None`` when the variable has no default.
    is_path : bool
        Whether the value denotes a filesystem path (drives ``env check``).
    """

    name: str
    group: EnvGroup
    comment: str
    default: str | None
    is_path: bool


def stage_path_var(stage: str) -> str:
    """Return the canonical env-var name holding a stage's directory.

    Parameters
    ----------
    stage : str
        Stage label, e.g. ``"1b"``.

    Returns
    -------
    str
        The variable name, e.g. ``"STAGE_1B_PATH"``.
    """
    return f"STAGE_{stage.upper()}_PATH"
