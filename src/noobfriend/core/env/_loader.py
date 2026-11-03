"""Locate and load layered ``.env`` files into :data:`os.environ`.

The two public entry points are:

- :func:`find_project_root` — walk up from the current working directory
  looking for the conventional project markers (``.git``, ``pyproject.toml``).
- :func:`load_environment` — read ``.env`` from the project root (base) and
  the current working directory (override) and apply the merged result to
  :data:`os.environ`.
- :func:`read_environment` — the same merged view, returned without touching
  :data:`os.environ`.

Auto-loading on ``import noobfriend`` is **opt-in**: set the environment
variable ``NOOBFRIEND_AUTOLOAD_ENV=1`` to enable it; the top-level
``noobfriend`` package picks the flag up at import time.
"""

import logging
import os
from pathlib import Path

from dotenv import dotenv_values

logger = logging.getLogger(__name__)

_ENV_FILENAME = ".env"
_PROJECT_MARKERS = (".git", "pyproject.toml")


def find_project_root() -> Path | None:
    """Walk up from the current working directory looking for a project marker.

    Returns
    -------
    Path or None
        The first ancestor of :func:`Path.cwd` (or ``cwd`` itself) that
        contains either a ``.git`` directory or a ``pyproject.toml`` file.
        ``None`` if no such ancestor exists up to the filesystem root.

    Notes
    -----
    ``.env`` is intentionally **not** treated as a project marker; a stray
    ``.env`` in some intermediate directory would otherwise hijack the root
    search and shadow the real project layout.
    """
    current = Path.cwd()
    while True:
        if any((current / marker).exists() for marker in _PROJECT_MARKERS):
            return current
        if current == current.parent:
            return None
        current = current.parent


def load_environment(*, override: bool = False) -> list[str]:
    """Load layered ``.env`` files into :data:`os.environ`.

    Two files are considered, in order:

    1. ``<project_root>/.env`` (base layer; project-wide defaults).
    2. ``<cwd>/.env`` (override layer; per-run customisation).

    Values from the cwd file always win over the project-root file when both
    define the same key (this is the "layered" semantic). Whether either of
    them is allowed to overwrite pre-existing shell exports is controlled by
    ``override``.

    Parameters
    ----------
    override : bool, default ``False``
        If ``True``, values from the ``.env`` files overwrite variables that
        are already set in the environment (typical shell exports). If
        ``False``, existing shell variables take precedence.

    Returns
    -------
    list of str
        Sorted names of the variables that were actually written to
        :data:`os.environ` during this call. Variables already present with
        the same value (or skipped because ``override=False``) are not
        reported.

    Notes
    -----
    Only the **names** of loaded variables are logged, never their values, so
    secrets in ``.env`` files do not leak into log output.
    """
    merged, found = _layered_values()
    if not found:
        logger.warning(
            "noobfriend.load_environment: no .env file found in project root or cwd"
        )
        return []

    applied: list[str] = []
    for key, value in merged.items():
        if override or key not in os.environ:
            os.environ[key] = value
            applied.append(key)

    applied.sort()
    logger.info(
        "noobfriend.load_environment: applied %d vars from %s: %s",
        len(applied),
        ", ".join(str(p) for p in found),
        applied,
    )
    return applied


def read_environment() -> dict[str, str]:
    """Return the merged layered ``.env`` values without applying them.

    The side-effect-free counterpart of :func:`load_environment`: the same two
    files are merged with the same precedence (cwd over project root), but
    :data:`os.environ` is left untouched and nothing is logged. Callers that
    must not mutate the process environment (for example code run at import
    time) combine this with :data:`os.environ` themselves.

    Returns
    -------
    dict of str to str
        Variable names mapped to their merged values. Keys declared without a
        value are omitted. Empty when neither ``.env`` file exists.
    """
    merged, _ = _layered_values()
    return merged


def _layered_values() -> tuple[dict[str, str], list[Path]]:
    """Merge the project-root and cwd ``.env`` files, cwd winning."""
    project_root = find_project_root()
    cwd = Path.cwd()

    candidates: list[Path] = []
    if project_root is not None:
        candidates.append(project_root / _ENV_FILENAME)
    if cwd != project_root:
        candidates.append(cwd / _ENV_FILENAME)

    merged: dict[str, str] = {}
    found: list[Path] = []
    for path in candidates:
        if path.exists():
            found.append(path)
            merged.update(
                {
                    key: value
                    for key, value in dotenv_values(path).items()
                    if value is not None
                }
            )
    return merged, found
