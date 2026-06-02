"""Filesystem helpers for the ``env`` CLI: read, write, and inspect ``.env``."""

from collections.abc import Mapping
from pathlib import Path

from dotenv import dotenv_values, set_key


def read_env_file(path: Path) -> dict[str, str | None]:
    """Read a ``.env`` file into a name -> value mapping.

    Unlike :func:`noobfriend.core.env.load_environment`, this does **not** touch
    :data:`os.environ`; it is a pure read used to seed re-rendering and checks.

    Parameters
    ----------
    path : Path
        The ``.env`` file to read. A missing file yields an empty mapping.

    Returns
    -------
    dict of str to (str or None)
        Variable names mapped to their values (``None`` for bare ``KEY=``).
    """
    if not path.exists():
        return {}
    return dict(dotenv_values(path))


def write_env_file(path: Path, content: str) -> None:
    """Write ``content`` to ``path``, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def upsert_var(path: Path, key: str, value: str) -> None:
    """Set ``key`` to ``value`` in the .env at ``path``, preserving the rest.

    Uses python-dotenv's :func:`~dotenv.set_key` (a line-level edit), so
    comments, ordering and unrelated variables are left untouched. An existing
    key is updated in place; a new key is appended.
    """
    set_key(str(path), key, value, quote_mode="auto")


def find_overrides(
    declared: Mapping[str, str | None], shell_env: Mapping[str, str]
) -> list[tuple[str, str, str]]:
    """Find declared variables that the shell environment overrides.

    A variable already present in ``shell_env`` shadows the ``.env`` value (the
    shell wins in :func:`noobfriend.core.env.load_environment`). This reports the
    cases where the effective shell value differs from what the file declares —
    the silent-override hazard worth surfacing.

    Parameters
    ----------
    declared : mapping of str to (str or None)
        Variables declared in the ``.env`` file.
    shell_env : mapping of str to str
        The process environment captured before any ``.env`` is applied.

    Returns
    -------
    list of (str, str, str)
        ``(name, declared_value, shell_value)`` per overridden variable.
    """
    overrides: list[tuple[str, str, str]] = []
    for name, declared_value in declared.items():
        if declared_value is None:
            continue
        shell_value = shell_env.get(name)
        if shell_value is not None and shell_value != declared_value:
            overrides.append((name, declared_value, shell_value))
    return overrides


def split_existing(path: Path) -> tuple[Path, Path | None]:
    """Split ``path`` into its existing prefix and first missing component.

    Walks ``path`` from the root and returns the deepest ancestor that exists
    together with the first component that does not. Used to colour
    partially-existing paths in ``env check``.

    Parameters
    ----------
    path : Path
        An absolute path to inspect.

    Returns
    -------
    existing : Path
        The deepest existing ancestor (possibly ``path`` itself).
    missing : Path or None
        The remainder that does not exist, or ``None`` when ``path`` fully
        exists.
    """
    if path.exists():
        return path, None
    existing = path
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    missing = path.relative_to(existing)
    return existing, missing
