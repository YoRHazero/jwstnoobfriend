"""Render the base ``.env`` file text from the configuration schema.

This replaces the previous Jinja2 templates: the variable set, comments,
defaults and section grouping all come from :func:`noobfriend.core.env.env_fields`
(i.e. from :class:`~noobfriend.core.env.NoobSettings`), so a single source of
truth drives both reading and writing. Per-stage path variables are not rendered
here; they are added in place by ``env add-stage`` (see
:func:`noobfriend.cli.env._io.upsert_var`).
"""

from collections.abc import Mapping

from noobfriend.core.env import EnvGroup, env_fields

_HEADER = (
    "# noobfriend environment configuration\n"
    "# Managed by `noobfriend env`. Unset options are left commented out."
)


def render_base(values: Mapping[str, str | None]) -> str:
    """Render the static configuration section, grouped by :class:`EnvGroup`.

    Parameters
    ----------
    values : mapping of str to (str or None)
        Override values keyed by environment-variable name (e.g. ``"NOOB_SERVER"``).
        A missing key falls back to the field's default; an unset value (no
        override and no default) is written as a commented-out ``# KEY=`` line so
        the file still documents the option.

    Returns
    -------
    str
        The rendered ``.env`` body, ending with a trailing newline.
    """
    lines: list[str] = [_HEADER, ""]
    fields = env_fields()
    for group in EnvGroup:
        members = [field for field in fields if field.group is group]
        if not members:
            continue
        lines.append(f"# --- {group.value} ---")
        for field in members:
            if field.comment:
                lines.append(f"# {field.comment}")
            value = values.get(field.name)
            if value is None:
                value = field.default
            lines.append(f"{field.name}={value}" if value else f"# {field.name}=")
        lines.append("")
    return "\n".join(lines) + "\n"
