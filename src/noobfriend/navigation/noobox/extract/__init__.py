"""Collection-level extraction for :class:`~noobfriend.navigation.NooBox`.

:class:`BoxExtract` is the ``box.extract`` accessor; :class:`BoxCutout` is the
navigation-side result of ``box.extract.cutout(...)`` (the PSF path returns a
:class:`~noobfriend.extraction.psf.SourceExtractor`, which lives in
:mod:`noobfriend.extraction`). ``BoxCutout`` is a public *type* (for annotation
and inspection) but is obtained only through the accessor, never constructed
directly.
"""

from noobfriend.navigation.noobox.extract._core import BoxExtract
from noobfriend.navigation.noobox.extract._cutout import BoxCutout

__all__ = ["BoxCutout", "BoxExtract"]
