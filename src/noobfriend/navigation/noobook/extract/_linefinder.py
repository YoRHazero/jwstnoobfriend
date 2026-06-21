"""The :class:`FrameLineFinder`: blind line heatmap for one grism exposure.

Obtained only through
:meth:`BookExtract.linefinder <noobfriend.navigation.noobook.extract._core.BookExtract.linefinder>`
(``book.extract.linefinder(...)``); not constructed directly. It wraps a
:class:`~noobfriend.extraction.grism.linefind.GrismLineFinder` configured for one
:class:`~noobfriend.navigation.NooBook` and exposes that exposure's heatmap,
peak catalog, and a plot. The deep, dither-combined product is the box-level
:class:`~noobfriend.navigation.noobox.extract._linefinder.BoxLineFinder`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from noobfriend.extraction.grism.linefind import Candidate, GrismLineFinder
    from noobfriend.navigation.noobook._core import NooBook


def _build(book: NooBook, **config: Any) -> FrameLineFinder:
    """Build a :class:`FrameLineFinder` (see :meth:`BookExtract.linefinder`)."""
    from noobfriend.navigation._linefinder import configure

    return FrameLineFinder(book=book, finder=configure(book, **config))


class FrameLineFinder:
    """One grism exposure's blind line-likelihood heatmap and peaks.

    Construct it with ``book.extract.linefinder(...)`` rather than directly.
    :attr:`heatmap` reads the exposure's pixels on first access (then caches);
    it is 1:1 with the detector frame, so the book's own WCS places it.

    Parameters
    ----------
    book : NooBook
        The grism exposure.
    finder : GrismLineFinder
        The finder configured for ``book`` (dispersion from its pupil).
    """

    def __init__(self, *, book: NooBook, finder: GrismLineFinder) -> None:
        """See the class docstring; built by ``book.extract.linefinder(...)``."""
        self._book = book
        self._finder = finder
        self._heatmap: np.ndarray | None = None

    @property
    def heatmap(self) -> np.ndarray:
        """Per-pixel band-pass SNR map, 1:1 with the frame (read on first access)."""
        if self._heatmap is None:
            error = self._book.err
            if error is None:
                raise ValueError(
                    f"grism book {self._book.id} has no ERR array for the heatmap."
                )
            self._heatmap = self._finder.exposure_heatmap(self._book.data, error)
        return self._heatmap

    def catalog(self) -> list[Candidate]:
        """Peak-find the heatmap into a candidate list (in frame pixels)."""
        return self._finder.catalog(self.heatmap)

    def plot(self, **kwargs: Any) -> Any:
        """Display the heatmap as a zoomable image (Bokeh ``imshow``)."""
        from noobfriend.core.display.plot import imshow

        kwargs.setdefault("title", f"{self._book.id} line heatmap")
        return imshow(self.heatmap, **kwargs)

    def __repr__(self) -> str:
        """Return a one-line summary naming the exposure."""
        return f"FrameLineFinder({self._book.id})"
