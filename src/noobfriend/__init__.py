"""noobfriend — helpers for JWST data reduction and product management.

The top-level package intentionally re-exports nothing: per project convention,
public names are imported from the subpackage that defines them
(``from noobfriend.core import setup_logging``, ``from noobfriend.core.display
import AttrView``, etc.).

The only work this module does at import time is attach a
:class:`logging.NullHandler` to the ``noobfriend`` logger so that library code
emitting records before any handler is configured does not trigger the
``lastResort`` warning.
"""

import logging

logging.getLogger("noobfriend").addHandler(logging.NullHandler())
