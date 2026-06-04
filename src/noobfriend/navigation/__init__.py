"""High-level navigation over JWST products: :class:`NooBook` and :class:`NooBox`.

A :class:`NooBook` wraps a single product file — its identity, the metadata
worth keeping resident, and lazy access to the heavy contents. A
:class:`NooBox` manages a collection of NooBooks over one shared, bounded byte
cache. :class:`Footprint` is the serializable on-sky region a book carries.

``NooBook`` and ``NooBox`` live in the eponymous ``noobook`` / ``noobox``
subpackages — each a "fat module" for its one public class — and are surfaced
here because the subpackage name maps one-to-one onto the class. Nothing else
inside those subpackages (accessors, parsers) is lifted to this level.
"""

from noobfriend.navigation._footprint import Footprint
from noobfriend.navigation.noobook import NooBook
from noobfriend.navigation.noobox import NooBox

__all__ = ["Footprint", "NooBook", "NooBox"]
