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

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from noobfriend.core.env._loader import load_environment
from noobfriend.core.env.schema import EnvGroup, stage_path_var

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

    start_stage: str | None = Field(
        default=None,
        description="Stage to start the reduction from, e.g. '1b'.",
        json_schema_extra={"group": EnvGroup.setup},
    )
    crds_path: Path | None = Field(
        default=None,
        description="Path to the CRDS cache directory.",
        json_schema_extra={"group": EnvGroup.setup},
    )
    data_root_path: Path | None = Field(
        default=None,
        description="Root directory of the data; base for auto-named stage dirs.",
        json_schema_extra={"group": EnvGroup.storage},
    )
    noob_server: str = Field(
        default="localhost",
        description=(
            "Default download host (an ~/.ssh/config alias); "
            "unset / 'localhost' / 'local' mean the local machine."
        ),
        json_schema_extra={"group": EnvGroup.remote},
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
