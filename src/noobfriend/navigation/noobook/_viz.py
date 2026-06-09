"""The :class:`BookViz` accessor: plotting sugar bound to one NooBook.

Reached as :attr:`NooBook.viz <noobfriend.navigation.noobook._core.NooBook.viz>`.
Each method is a thin delegate to :mod:`noobfriend.core.display.plot` that
supplies the book's own arrays; the (heavy) Bokeh import stays inside the
methods so importing the package never pulls it in.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from noobfriend.navigation.noobook._core import NooBook


class BookViz:
    """Plotting sugar bound to one :class:`NooBook`.

    Parameters
    ----------
    book : NooBook
        The book whose data these plots draw.
    """

    def __init__(self, book: "NooBook") -> None:
        """See the class docstring for parameters."""
        self._book = book

    def imshow(self, **kwargs: Any) -> Any:
        """Show the book's ``SCI`` array as an interactive Bokeh image.

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`noobfriend.core.display.plot.imshow`.

        Returns
        -------
        Any
            The Bokeh model returned by ``imshow``.
        """
        from noobfriend.core.display.plot import imshow

        return imshow(self._book.data, **kwargs)

    def imshow_blink(self, *others: "NooBook | np.ndarray", **kwargs: Any) -> Any:
        """Blink-compare this book's ``SCI`` array against other frames.

        This book is the first frame; each of ``others`` is overlaid after it
        and a segmented control flips between them. An ``other`` may be another
        :class:`NooBook` (its ``SCI`` array, labelled by id) or a raw
        :class:`numpy.ndarray` (e.g. a segmentation map, labelled by index) --
        the latter is what this single-book accessor offers over
        :meth:`NooBox.viz.imshow_blink <noobfriend.navigation.noobox._viz.BoxViz.imshow_blink>`.
        Frames are placed by translation only and expected to share this book's
        pixel grid; pass ``offsets`` to shift dithers into alignment, or
        ``align="wcs"`` to derive them from each frame's WCS (imaging NooBooks
        only -- raw-array frames have no WCS to align by).

        Parameters
        ----------
        *others : NooBook or numpy.ndarray
            Additional frames to blink against this book, in draw order.
        **kwargs
            Forwarded to
            :func:`~noobfriend.navigation._blink.blink_frames` -- ``align`` /
            ``atol`` for WCS placement, and ``offsets``, ``labels``, ``vmin``,
            ``vmax``, ``cmap``, ``stretch``, ``size``, ``title``, ``blink`` on to
            :func:`noobfriend.core.display.plot.imshow_blink`.

        Returns
        -------
        Any
            The display handle returned by ``imshow_blink``.
        """
        from noobfriend.navigation._blink import blink_frames

        return blink_frames([self._book, *others], **kwargs)
