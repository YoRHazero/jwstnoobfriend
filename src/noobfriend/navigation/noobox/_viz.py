"""The :class:`BoxViz` accessor: plotting sugar bound to one NooBox.

Reached as :attr:`NooBox.viz <noobfriend.navigation.noobox._core.NooBox.viz>`.
Methods draw collection-level views from the books' resident metadata (so
:meth:`footprints` opens no files); the Bokeh import stays inside the methods.
"""

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from noobfriend.navigation.noobox._core import NooBox


class BoxViz:
    """Plotting sugar bound to one :class:`NooBox`.

    Parameters
    ----------
    box : NooBox
        The collection these plots summarise.
    """

    def __init__(self, box: "NooBox") -> None:
        """See the class docstring for parameters."""
        self._box = box

    def footprints(self, **kwargs: Any) -> Any:
        """Draw the sky footprints of every member that has one.

        Books whose :attr:`~NooBook.footprint` is ``None`` (no assigned WCS, or
        not yet probed) are skipped. Each footprint is labelled with its book
        id unless ``labels`` is supplied. Reads only resident metadata, so no
        files are opened.

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`noobfriend.core.display.plot.plot_footprints`.

        Returns
        -------
        Any
            The Bokeh model returned by ``plot_footprints``.
        """
        from noobfriend.core.display.plot import plot_footprints

        books = [book for book in self._box if book.footprint is not None]
        polygons = [np.array(book.footprint.corners, dtype=float) for book in books]
        kwargs.setdefault("labels", [book.id for book in books])
        return plot_footprints(polygons, **kwargs)

    def imshow_blink(self, **kwargs: Any) -> Any:
        """Blink-compare the ``SCI`` array of every member.

        Each book becomes one frame, drawn in iteration order and labelled by
        its :attr:`~NooBook.id`; a segmented control flips between them. The
        members are expected to share a pixel grid, since frames are placed by
        translation only (no rescale, no rotation) -- curate first with
        :meth:`~NooBox.filter` so only co-gridded books remain (e.g. one
        pointing's dither set, or one product's stage ladder). Pass
        ``align="wcs"`` to place the members by their WCS (imaging only) instead
        of stacking them at a common origin. Files are read only when the frames
        are drawn.

        Parameters
        ----------
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

        return blink_frames(list(self._box), **kwargs)
