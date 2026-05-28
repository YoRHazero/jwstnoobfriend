"""noobfriend — helpers for JWST data reduction and product management."""

import logging

from noobfriend.core import console, setup_logging

logging.getLogger("noobfriend").addHandler(logging.NullHandler())

__all__ = ["console", "setup_logging"]
