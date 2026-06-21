"""Per-book extraction for :class:`~noobfriend.navigation.NooBook`.

:class:`BookExtract` is the ``book.extract`` accessor; :class:`FrameLineFinder`
is the navigation-side result of ``book.extract.linefinder(...)`` (the
single-exposure blind line heatmap). ``FrameLineFinder`` is a public *type* (for
annotation and inspection) but is obtained only through the accessor, never
constructed directly.
"""

from noobfriend.navigation.noobook.extract._core import BookExtract
from noobfriend.navigation.noobook.extract._linefinder import FrameLineFinder

__all__ = ["BookExtract", "FrameLineFinder"]
