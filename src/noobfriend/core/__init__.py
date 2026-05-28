"""Cross-cutting infrastructure for the ``noobfriend`` package.

Re-exports the public names defined in the modules directly inside ``core/``
(``console.py``, ``log.py``). Names defined in subpackages such as
``core.display`` are *not* re-exported here; import them from the subpackage
that owns them.
"""

from noobfriend.core.console import console
from noobfriend.core.log import setup_logging

__all__ = ["console", "setup_logging"]
