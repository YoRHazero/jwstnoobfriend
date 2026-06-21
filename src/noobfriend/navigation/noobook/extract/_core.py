"""The :class:`BookExtract` accessor: extraction sugar bound to one NooBook.

Reached as
:attr:`NooBook.extract <noobfriend.navigation.noobook._core.NooBook.extract>`.
Each method delegates to :mod:`noobfriend.extraction`, anchoring the extraction
on the book's own WCS / data; the extraction imports stay inside the methods.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from noobfriend.extraction.cutout import Cutout
    from noobfriend.navigation.noobook._core import NooBook
    from noobfriend.navigation.noobook.extract._linefinder import FrameLineFinder


class BookExtract:
    """Extraction sugar bound to one :class:`NooBook`.

    Parameters
    ----------
    book : NooBook
        The book whose WCS and data the extractions are anchored on.
    """

    def __init__(self, book: "NooBook") -> None:
        """See the class docstring for parameters."""
        self._book = book

    def cutout(self, ra: float, dec: float, **kwargs: Any) -> "Cutout":
        """Build a :class:`Cutout` centered on a sky position in this product.

        Delegates to :meth:`Cutout.from_world` using the book's WCS.

        Parameters
        ----------
        ra, dec : float
            Center world coordinates in degrees.
        **kwargs
            Forwarded to :meth:`Cutout.from_world` (the size specification).

        Returns
        -------
        Cutout

        Raises
        ------
        ValueError
            If the product has no assigned WCS.
        """
        from noobfriend.extraction.cutout import Cutout

        wcs = self._book.wcs
        if wcs is None:
            raise ValueError(
                f"{self._book.id} has no assigned WCS to build a cutout from."
            )
        return Cutout.from_world(wcs, ra, dec, **kwargs)

    def linefinder(self, **config: Any) -> "FrameLineFinder":
        """Blind emission-line heatmap for this one grism exposure.

        The single-frame entry to
        :class:`~noobfriend.extraction.grism.linefind.GrismLineFinder`: it
        configures a finder for this book (dispersion from its ``pupil``) and
        exposes the per-exposure band-pass SNR heatmap, its peak catalog, and a
        plot. For the deep, dither-combined heatmap use ``box.extract.linefinder``
        instead.

        Parameters
        ----------
        **config
            Detection parameters forwarded to
            :meth:`~noobfriend.extraction.grism.linefind.GrismLineFinder.configure`
            (e.g. ``line_sigmas``, ``threshold``).

        Returns
        -------
        FrameLineFinder

        Raises
        ------
        ValueError
            If this book is not a grism frame (pupil is not ``GRISM*``).
        """
        from noobfriend.navigation.noobook.extract._linefinder import _build

        return _build(self._book, **config)
