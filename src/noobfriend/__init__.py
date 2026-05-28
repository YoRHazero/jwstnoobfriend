"""noobfriend — helpers for JWST data reduction and product management.

The top-level package intentionally re-exports nothing: per project convention,
public names are imported from the subpackage that defines them
(``from noobfriend.core import setup_logging``, ``from noobfriend.core.display
import AttrView``, etc.).

Two runtime side effects do run at import time:

1. A :class:`logging.NullHandler` is attached to the ``noobfriend`` logger so
   that library code emitting records before any handler is configured does
   not trigger the ``lastResort`` warning.
2. If the environment variable ``NOOBFRIEND_AUTOLOAD_ENV`` equals ``"1"``,
   :func:`noobfriend.core.env.load_environment` is invoked to merge layered
   ``.env`` files into :data:`os.environ`. This is opt-in so that the default
   ``import noobfriend`` remains free of surprising environment mutations.
"""

import logging
import os

logging.getLogger("noobfriend").addHandler(logging.NullHandler())

if os.environ.get("NOOBFRIEND_AUTOLOAD_ENV") == "1":
    from noobfriend.core.env import load_environment

    load_environment()
