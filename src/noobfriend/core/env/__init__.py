"""Locate and load layered ``.env`` files, and access them as typed settings.

Public surface:

- :func:`find_project_root`, :func:`load_environment` — discover and merge the
  layered ``.env`` files into :data:`os.environ`.
- :class:`NoobSettings`, :func:`get_settings` — validated, typed access to the
  configuration variables.
- :class:`EnvGroup`, :func:`stage_path_var` — declarative primitives shared
  with the ``noobfriend env`` CLI.
"""

from noobfriend.core.env._loader import find_project_root, load_environment
from noobfriend.core.env.schema import EnvGroup, stage_path_var
from noobfriend.core.env.settings import NoobSettings, get_settings

__all__ = [
    "EnvGroup",
    "NoobSettings",
    "find_project_root",
    "get_settings",
    "load_environment",
    "stage_path_var",
]
