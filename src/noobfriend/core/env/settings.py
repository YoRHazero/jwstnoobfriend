"""Typed, validated access to noobfriend's ``.env`` configuration.

:class:`NoobSettings` is the single source of truth for the configuration
variables: both the runtime readers and the ``noobfriend env`` CLI derive the
variable set from this model, so names and defaults never drift between the
code that *writes* a ``.env`` file and the code that *reads* it.

The blessed entry point is :func:`get_settings`, which merges the layered
``.env`` files into :data:`os.environ` and then builds a validated instance.
"""

import logging
import os
from pathlib import Path
from typing import get_args

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from noobfriend.core.env._loader import load_environment
from noobfriend.core.env.schema import EnvField, EnvGroup, EnvPathKind, stage_path_var

logger = logging.getLogger(__name__)


class NoobSettings(BaseSettings):
    """Validated view over the layered ``.env`` configuration.

    Each field maps to an upper-cased environment variable (``noob_server`` ->
    ``NOOB_SERVER``) read from :data:`os.environ`, which :func:`get_settings`
    populates from the project / cwd ``.env`` files beforehand. Unknown
    variables (including the dynamic ``STAGE_<STAGE>_PATH`` family) are ignored
    here and accessed through :meth:`start_stage_path`.

    Notes
    -----
    The ``json_schema_extra["group"]`` on each field records the
    :class:`~noobfriend.core.env.schema.EnvGroup` the variable belongs to; the
    ``env`` CLI introspects it when rendering and checking ``.env`` files.
    """

    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    crds_server_url: str = Field(
        default="https://jwst-crds.stsci.edu",
        description="URL of the CRDS reference-file server (used by the JWST CRDS client).",
        json_schema_extra={"group": EnvGroup.setup},
    )
    crds_path: Path | None = Field(
        default=None,
        description="Path to the CRDS cache directory.",
        json_schema_extra={"group": EnvGroup.setup},
    )
    start_stage: str | None = Field(
        default=None,
        description="Stage to start the reduction from, e.g. '1b'.",
        json_schema_extra={"group": EnvGroup.setup},
    )
    noobox_path: Path | None = Field(
        default=None,
        description="Path to the NooBox manifest file used by NooBox.save / load.",
        json_schema_extra={"group": EnvGroup.noob, "path_kind": EnvPathKind.file},
    )
    noob_server: str = Field(
        default="localhost",
        description=(
            "Default download host (an ~/.ssh/config alias); "
            "unset / 'localhost' / 'local' mean the local machine."
        ),
        json_schema_extra={"group": EnvGroup.storage},
    )
    data_root_path: Path | None = Field(
        default=None,
        description="Root directory of the data; base for auto-named stage dirs.",
        json_schema_extra={"group": EnvGroup.storage},
    )

    def start_stage_path(self) -> str | None:
        """Resolve the directory configured for :attr:`start_stage`.

        Reads the dynamic ``STAGE_<START_STAGE>_PATH`` variable from
        :data:`os.environ`. The value is returned verbatim (not created)
        because it may name a directory on a remote host.

        Returns
        -------
        str or None
            The configured stage path, or ``None`` if :attr:`start_stage` is
            unset or its corresponding path variable is missing.
        """
        if not self.start_stage:
            return None
        var = stage_path_var(self.start_stage)
        value = os.getenv(var)
        if value is None:
            logger.warning(
                "START_STAGE is set to %s, but %s is not set. "
                "Check whether the .env file is written correctly.",
                self.start_stage,
                var,
            )
        return value

    def server_host(self) -> str | None:
        """Return :attr:`noob_server` as an ssh host, or ``None`` when local.

        The empty string and the ``localhost`` / ``local`` aliases all mean the
        local machine and yield ``None``; any other value is the ssh destination
        the ``storage`` paths (including ``STAGE_<STAGE>_PATH``) live on.

        Returns
        -------
        str or None
            The remote host, or ``None`` when storage is on the local machine.
        """
        host = self.noob_server.strip()
        return None if host.lower() in {"", "localhost", "local"} else host


def get_settings() -> NoobSettings:
    """Load the layered ``.env`` files and return a validated settings object.

    Returns
    -------
    NoobSettings
        Settings populated from :data:`os.environ` after
        :func:`noobfriend.core.env.load_environment` has merged the project and
        cwd ``.env`` files into it.
    """
    load_environment()
    return NoobSettings()


def env_fields() -> list[EnvField]:
    """Derive the render-friendly view of the configuration from :class:`NoobSettings`.

    Returns one :class:`EnvField` per model field, in declaration order, so the
    ``env`` CLI's rendering and checking stay in lockstep with the runtime
    model: adding a field to :class:`NoobSettings` automatically makes it appear
    in generated ``.env`` files.

    Returns
    -------
    list of EnvField
        The configuration variables, ordered as declared on the model.
    """
    fields: list[EnvField] = []
    for name, info in NoobSettings.model_fields.items():
        extra = info.json_schema_extra or {}
        group = (
            extra.get("group", EnvGroup.setup)
            if isinstance(extra, dict)
            else EnvGroup.setup
        )
        annotation = info.annotation
        is_path = annotation is Path or Path in get_args(annotation)
        path_kind = extra.get("path_kind", EnvPathKind.directory) if is_path else None
        if isinstance(path_kind, str):
            path_kind = EnvPathKind(path_kind)
        default = info.default
        fields.append(
            EnvField(
                name=name.upper(),
                group=group,
                comment=info.description or "",
                default=None if default is None else str(default),
                is_path=is_path,
                path_kind=path_kind,
            )
        )
    return fields
