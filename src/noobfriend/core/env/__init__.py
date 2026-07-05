"""Locate and load layered ``.env`` files, and access them as typed settings.

Public surface:

- :func:`find_project_root`, :func:`load_environment` — discover and merge the
  layered ``.env`` files into :data:`os.environ`.
- :class:`NoobSettings`, :func:`get_settings`, :func:`env_fields` — validated,
  typed access to the configuration variables, plus a render-friendly view.
- :class:`EnvField`, :class:`EnvGroup`, :class:`EnvPathKind`,
  :func:`stage_path_var` — declarative primitives shared with the
  ``noobfriend env`` CLI.
- :class:`MountState`, :func:`find_mount_state`, :func:`to_canonical`,
  :func:`to_local` — read the ``env mount`` sidecar and resolve canonical
  ``host:server_path`` locations against the local mount.
"""

from noobfriend.core.env._loader import find_project_root, load_environment
from noobfriend.core.env._mount import (
    SIDECAR_NAME,
    MountState,
    find_mount_state,
    load_state,
    sidecar_path,
    to_canonical,
    to_local,
)
from noobfriend.core.env.schema import EnvField, EnvGroup, EnvPathKind, stage_path_var
from noobfriend.core.env.settings import NoobSettings, env_fields, get_settings

__all__ = [
    "SIDECAR_NAME",
    "EnvField",
    "EnvGroup",
    "EnvPathKind",
    "MountState",
    "NoobSettings",
    "env_fields",
    "find_mount_state",
    "find_project_root",
    "get_settings",
    "load_environment",
    "load_state",
    "sidecar_path",
    "stage_path_var",
    "to_canonical",
    "to_local",
]
