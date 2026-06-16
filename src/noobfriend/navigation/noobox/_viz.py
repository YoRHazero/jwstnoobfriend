"""The :class:`BoxViz` accessor: plotting sugar bound to one NooBox.

Reached as :attr:`NooBox.viz <noobfriend.navigation.noobox._core.NooBox.viz>`.
Methods draw collection-level views from the books' resident metadata (so
:meth:`footprints` opens no files); the Bokeh import stays inside the methods.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from noobfriend.navigation.noobook._core import NooBook
    from noobfriend.navigation.noobox._core import NooBox

DEFAULT_BOX_FOOTPRINT_COLOR = "#1f77b4"


def _group_value(book: NooBook, group_by: str | Callable[[NooBook], object]) -> object:
    """Return one grouping value for ``book``."""
    return getattr(book, group_by) if isinstance(group_by, str) else group_by(book)


def _grouped_colors(
    books: Sequence[NooBook],
    group_by: str | Callable[[NooBook], object] | None,
    *,
    color: str,
    group_colors: Mapping[object, str] | None,
    palette: Sequence[str] | None,
) -> str | list[str]:
    """Resolve NooBook grouping options into per-footprint colours."""
    if group_by is None:
        if group_colors is not None:
            raise ValueError("group_colors requires group_by")
        if palette is not None:
            raise ValueError("palette requires group_by")
        return color

    keys = [_group_value(book, group_by) for book in books]
    try:
        groups = list(dict.fromkeys(keys))
    except TypeError as exc:
        raise ValueError("group_by values must be hashable") from exc

    explicit = dict(group_colors or {})
    missing = [group for group in groups if group not in explicit]
    if palette is None:
        from noobfriend.core.display.plot._bokeh import categorical_palette

        palette_colors = categorical_palette(len(missing))
    else:
        palette_colors = list(palette)
    if missing and not palette_colors:
        raise ValueError("palette is empty")

    by_group = dict(explicit)
    for index, group in enumerate(missing):
        by_group[group] = palette_colors[index % len(palette_colors)]
    return [by_group[group] for group in keys]


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

    def footprints(
        self,
        *,
        group_by: str | Callable[[NooBook], object] | None = None,
        color: str = DEFAULT_BOX_FOOTPRINT_COLOR,
        group_colors: Mapping[object, str] | None = None,
        palette: Sequence[str] | None = None,
        points: np.ndarray | None = None,
        labels: Sequence[str] | None = None,
        colors: str | Sequence[str] | None = None,
        point_labels: Sequence[str] | None = None,
        point_colors: str | Sequence[str] | None = None,
        center_ra: float | None = None,
        center_dec: float | None = None,
        size: int = 620,
        title: str = "Footprints",
    ) -> Any:
        """Draw the sky footprints of every member that has one.

        Books whose :attr:`~NooBook.footprint` is ``None`` (no assigned WCS, or
        not yet probed) are skipped. Each footprint is labelled with its book
        id unless ``labels`` is supplied. Reads only resident metadata, so no
        files are opened.

        Parameters
        ----------
        group_by : str or callable, optional
            Group the plotted books before assigning footprint colours. A string
            names a :class:`NooBook` attribute such as ``"filter"``,
            ``"detector"`` or ``"stage"``; a callable receives each
            :class:`NooBook` and returns a hashable group value.
        color : str, default ``"#1f77b4"``
            Colour used for every footprint when ``group_by`` is not given.
        group_colors : mapping, optional
            Explicit colours for group values, e.g.
            ``{"F200W": "tab:blue", "F356W": "tab:orange"}``. Requires
            ``group_by``. Groups not present in the mapping are filled from
            ``palette``.
        palette : sequence of str, optional
            Colours assigned to groups not covered by ``group_colors``.
            Requires ``group_by``. If omitted, an automatic categorical palette
            is used for the number of missing groups.
        points : numpy.ndarray, optional
            ``(N, 2)`` array of ``(RA, Dec)`` source points to over-plot.
        labels : sequence of str, optional
            Per-footprint labels shown in the vertex hover. Defaults to the
            corresponding book ids.
        colors : str or sequence of str, optional
            Lower-level colour override passed directly to
            :func:`noobfriend.core.display.plot.plot_footprints`; use this for
            one colour for all footprints or one colour per footprint. Mutually
            exclusive with ``group_by``, ``group_colors`` and ``palette``.
        point_labels : sequence of str, optional
            Per-point labels shown in the point hover. Requires ``points``.
        point_colors : str or sequence of str, optional
            Source-point colours: one colour for all points or one per point.
            Requires ``points``.
        center_ra, center_dec : float, optional
            Initial view-centre direction in degrees. Defaults to the
            circular-mean RA and mean Dec of all footprint vertices.
        size : int, default 620
            Figure size in pixels (square).
        title : str, default ``"Footprints"``
            Figure title.

        Returns
        -------
        Any
            The Bokeh model returned by ``plot_footprints``.
        """
        from noobfriend.core.display.plot import plot_footprints

        books = [book for book in self._box if book.footprint is not None]
        polygons = [np.array(book.footprint.corners, dtype=float) for book in books]
        if colors is not None and (
            group_by is not None or group_colors is not None or palette is not None
        ):
            raise ValueError(
                "pass either colors or group_by/group_colors/palette, not both"
            )
        resolved_colors = (
            colors
            if colors is not None
            else _grouped_colors(
                books,
                group_by,
                color=color,
                group_colors=group_colors,
                palette=palette,
            )
        )
        return plot_footprints(
            polygons,
            points=points,
            labels=list(labels) if labels is not None else [book.id for book in books],
            colors=resolved_colors,
            point_labels=point_labels,
            point_colors=point_colors,
            center_ra=center_ra,
            center_dec=center_dec,
            size=size,
            title=title,
        )

    def imshow_blink(
        self,
        *,
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
        labels : sequence of str, optional
            Per-frame button labels. Defaults to each book's id.
        align : {"wcs", None}, optional
            When ``"wcs"``, derive pixel offsets from each member's GWCS.
            Mutually exclusive with ``offsets`` and requires every member to be
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
            list(self._box),
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
