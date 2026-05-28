"""Cross-cutting infrastructure for the ``noobfriend`` package.

Re-exports the public surface of ``noobfriend.core`` so callers can write
``from noobfriend.core import console`` without knowing the internal layout.
"""

from noobfriend.core.console import console
from noobfriend.core.display import AttrView
from noobfriend.core.log import setup_logging

__all__ = ["AttrView", "console", "setup_logging"]
