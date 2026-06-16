"""The :class:`BookViz` accessor: plotting sugar bound to one NooBook.

Reached as :attr:`NooBook.viz <noobfriend.navigation.noobook._core.NooBook.viz>`.
Each method is a thin delegate to :mod:`noobfriend.core.display.plot` that
supplies the book's own arrays; the (heavy) Bokeh import stays inside the
methods so importing the package never pulls it in.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal

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

    def imshow(
        self,
        *,
        vmin: float | None = None,
        vmax: float | None = None,
        pmin: float = 1.0,
        pmax: float = 99.0,
        cmap: str = "Greys",
        stretch: str = "linear",
        size: int = 620,
        title: str | None = None,
        transform_pixel_to_world: Callable[[Any, Any], tuple[Any, Any]] | None = None,
        coord_format: Literal["deg", "hms"] = "deg",
    ) -> Any:
        """Show the book's ``SCI`` array as an interactive Bokeh image.

        Parameters
        ----------
        vmin, vmax : float, optional
            Explicit lower/upper limits for the color stretch. Each takes
            precedence over the corresponding percentile cut; a side left as
            ``None`` falls back to ``pmin``/``pmax``.
        pmin, pmax : float, default 1.0, 99.0
            Percentile cuts used for whichever of ``vmin``/``vmax`` is not
            given.
        cmap : str, default ``"Greys"``
            Colormap; one of the keys supported by
            :func:`noobfriend.core.display.plot.imshow`.
        stretch : str, default ``"linear"``
            Intensity stretch, e.g. ``"linear"``, ``"log"`` or ``"eqhist"``.
        size : int, default 620
            Length of the longer figure edge, in pixels.
        title : str, optional
            Figure title.
        transform_pixel_to_world : callable, optional
            ``f(x, y) -> (ra, dec)`` mapping pixel coordinates to world
            coordinates in degrees. When given, the hover gains an RA/Dec row.
        coord_format : {"deg", "hms"}, default ``"deg"``
            RA/Dec hover format.

        Returns
        -------
        Any
            The Bokeh model returned by ``imshow``.
        """
        from noobfriend.core.display.plot import imshow

        return imshow(
            self._book.data,
            vmin=vmin,
            vmax=vmax,
            pmin=pmin,
            pmax=pmax,
            cmap=cmap,
            stretch=stretch,
            size=size,
            title=title,
            transform_pixel_to_world=transform_pixel_to_world,
            coord_format=coord_format,
        )

    def imshow_blink(
        self,
        *others: NooBook | np.ndarray,
        labels: Sequence[str] | None = None,
        align: Literal["wcs"] | None = None,
        atol: float = 1.0,
        offsets: Sequence[tuple[float, float]] | None = None,
        vmin: float | Sequence[float] | None = None,
        vmax: float | Sequence[float] | None = None,
        pmin: float = 1.0,
        pmax: float = 99.0,
        cmap: str = "Greys",
        stretch: str = "linear",
        size: int = 620,
        title: str | None = None,
        blink: bool = True,
    ) -> Any:
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
        labels : sequence of str, optional
            Per-frame button labels. Defaults to this book's id followed by
            each other book id or raw-array index.
        align : {"wcs", None}, optional
            When ``"wcs"``, derive pixel offsets from each frame's GWCS.
            Mutually exclusive with ``offsets`` and requires every frame to be
            an imaging :class:`NooBook`.
        atol : float, default 1.0
            Only with ``align="wcs"``: maximum allowed corner-offset spread, in
            pixels, before frames are rejected as more than a translation.
        offsets : sequence of (float, float), optional
            Per-frame ``(dx, dy)`` lower-left placement in pixels. Defaults to
            all frames stacked at a common origin.
        vmin, vmax : float or sequence of float, optional
            Color-stretch limits. A scalar applies to every frame; a sequence
            gives one limit per frame.
        pmin, pmax : float, default 1.0, 99.0
            Percentile cuts used per frame for whichever of ``vmin``/``vmax`` is
            not given.
        cmap : str, default ``"Greys"``
            Colormap shared by all frames.
        stretch : str, default ``"linear"``
            Intensity stretch shared by all frames.
        size : int, default 620
            Length of the longer figure edge, in pixels.
        title : str, optional
            Figure title.
        blink : bool, default True
            Add a play/pause button that auto-cycles through the frames.

        Returns
        -------
        Any
            The display handle returned by ``imshow_blink``.
        """
        from noobfriend.navigation._blink import blink_frames

        return blink_frames(
            [self._book, *others],
            labels=labels,
            align=align,
            atol=atol,
            offsets=offsets,
            vmin=vmin,
            vmax=vmax,
            pmin=pmin,
            pmax=pmax,
            cmap=cmap,
            stretch=stretch,
            size=size,
            title=title,
            blink=blink,
        )
