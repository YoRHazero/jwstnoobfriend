"""Declarative primitives shared by the env reader and the ``env`` CLI.

These are the small, dependency-free building blocks that both the runtime
settings model (:mod:`noobfriend.core.env.settings`) and the ``noobfriend env``
command group rely on, so the configuration layout is described in exactly one
place.
"""

from enum import Enum


class EnvGroup(str, Enum):
    """Logical section a configuration variable belongs to.

    The order of the members is the order in which sections are rendered into a
    ``.env`` file.
    """

    setup = "Setup"
    storage = "Storage"
    remote = "Remote"


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
