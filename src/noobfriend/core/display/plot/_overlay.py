"""Image overlay interfaces and catalogue overlay implementation.

The generic layer is intentionally small: an image overlay is any object with a
``draw(fig, frame)`` method.  :class:`CatalogOverlay` is one implementation of
that interface, not the framework itself.  Future overlays (apertures, regions,
trace curves, masks) should add their own classes with the same draw contract
without reusing the catalogue-specific helpers below.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
from bokeh.models import ColumnDataSource, GlyphRenderer, HoverTool
from numpy.typing import ArrayLike, DTypeLike

if TYPE_CHECKING:
    from bokeh.plotting._figure import figure as BokehFigure


@dataclass(frozen=True)
class ImageFrame:
    """Pixel-frame metadata interpreted by image overlays.

    ``imshow`` passes this information as a plain mapping or tuple; overlay
    implementations call :meth:`from_spec` to get the richer helper methods.

    Parameters
    ----------
    n_rows, n_cols : int
        Image shape in NumPy order. The displayed pixel frame spans
        ``0 <= x <= n_cols`` and ``0 <= y <= n_rows``.
    """

    n_rows: int
    n_cols: int

    @classmethod
    def from_spec(cls, spec: object) -> "ImageFrame":
        """Return an :class:`ImageFrame` from a frame object passed by ``imshow``.

        Accepts an existing :class:`ImageFrame`, a mapping with ``"n_rows"`` and
        ``"n_cols"``, or a two-item ``(n_rows, n_cols)`` sequence.
        """
        if isinstance(spec, cls):
            return spec
        if isinstance(spec, Mapping):
            try:
                return cls(n_rows=int(spec["n_rows"]), n_cols=int(spec["n_cols"]))
            except KeyError as exc:
                raise ValueError("frame mapping must contain n_rows and n_cols") from exc
        if isinstance(spec, Sequence) and not isinstance(spec, str | bytes):
            if len(spec) != 2:
                raise ValueError("frame sequence must be (n_rows, n_cols)")
            return cls(n_rows=int(spec[0]), n_cols=int(spec[1]))
        raise TypeError(
            "frame must be an ImageFrame, a mapping with n_rows/n_cols, "
            "or a (n_rows, n_cols) sequence"
        )

    def finite_in_bounds(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return a finite, in-image mask for pixel coordinates."""
        return (
            np.isfinite(x)
            & np.isfinite(y)
            & (x >= 0.0)
            & (x <= float(self.n_cols))
            & (y >= 0.0)
            & (y <= float(self.n_rows))
        )


class ImageOverlay(Protocol):
    """Protocol for objects that can draw themselves on an image figure."""

    def draw(self, fig: BokehFigure, frame: object) -> Sequence[GlyphRenderer]:
        """Draw this overlay and return its Bokeh renderers."""


class _ColumnLookup(Protocol):
    """Protocol for table-like objects supporting string column lookup."""

    def __getitem__(self, key: str) -> object:
        """Return a named column."""


DEFAULT_CATALOG_STYLE: Mapping[str, object] = {
    "marker": "circle",
    "size": 9,
    "fill_color": "#ffcc33",
    "fill_alpha": 0.9,
    "line_color": "black",
    "line_alpha": 0.9,
    "line_width": 0.6,
}


@dataclass(frozen=True)
class CatalogOverlayData:
    """Validated Bokeh-ready data for one catalogue overlay."""

    data: dict[str, Sequence[object]]
    tooltips: list[tuple[str, str]]
    style: dict[str, object]


@dataclass(frozen=True)
class CatalogOverlay:
    """A table-like source catalogue to draw over an image.

    Parameters
    ----------
    catalog : object
        Table-like object. Columns may be exposed by mapping lookup
        (``catalog["x"]``), table/data-frame lookup, structured-array fields, or
        attributes (``catalog.x``), which covers
        :class:`noobfriend.core.imgutils.SourceCatalog`.
    pixel_cols : tuple of str or None, default ("x", "y")
        Column names for pixel coordinates. When present in ``catalog``, these
        are used directly for drawing. If unavailable, ``world_cols`` plus
        ``world_to_pixel`` must be available instead.
    world_cols : tuple of str or None, optional
        Column names for world coordinates in degrees. When present, these are
        used for RA/Dec hover. If unavailable, ``pixel_to_world`` may be used to
        derive them from the drawing coordinates.
    pixel_to_world : callable, optional
        ``f(x, y) -> (ra, dec)`` used to derive world coordinates for a pixel
        catalogue. Called with 1-D arrays.
    world_to_pixel : callable, optional
        ``f(ra, dec) -> (x, y)`` used to place a world-coordinate catalogue on
        the image. Called with 1-D arrays.
    hover_cols : sequence of str, optional
        Additional catalogue columns shown in the point hover.
    style : mapping, optional
        Bokeh ``figure.scatter`` keyword overrides for this overlay. Defaults to
        a small yellow circular marker with a dark outline.
    clip : bool, default True
        If true, upload only finite points inside the image extent.
    """

    catalog: object
    pixel_cols: tuple[str, str] | None = ("x", "y")
    world_cols: tuple[str, str] | None = None
    pixel_to_world: Callable[
        [ArrayLike, ArrayLike], tuple[ArrayLike, ArrayLike]
    ] | None = None
    world_to_pixel: Callable[
        [ArrayLike, ArrayLike], tuple[ArrayLike, ArrayLike]
    ] | None = None
    hover_cols: Sequence[str] | None = None
    style: Mapping[str, object] | None = None
    clip: bool = True

    def resolve(self, frame: object) -> CatalogOverlayData:
        """Return this catalogue overlay as Bokeh source data and hover config."""
        image_frame = ImageFrame.from_spec(frame)
        pixel = _catalog_numeric_pair(self.catalog, self.pixel_cols, "pixel_cols")
        world = _catalog_numeric_pair(self.catalog, self.world_cols, "world_cols")

        if pixel is None:
            if world is None or self.world_to_pixel is None:
                raise ValueError(
                    "CatalogOverlay needs pixel coordinates for drawing: provide "
                    "catalog columns via pixel_cols, or provide world_cols together "
                    "with world_to_pixel."
                )
            x, y = _call_catalog_transform(
                self.world_to_pixel, world[0], world[1], "world_to_pixel"
            )
        else:
            x, y = pixel

        if world is None and self.pixel_to_world is not None:
            ra, dec = _call_catalog_transform(self.pixel_to_world, x, y, "pixel_to_world")
        elif world is not None:
            ra, dec = world
        else:
            ra = dec = None

        mask = (
            image_frame.finite_in_bounds(x, y)
            if self.clip
            else np.isfinite(x) & np.isfinite(y)
        )

        data: dict[str, Sequence[object]] = {
            "x": x[mask].tolist(),
            "y": y[mask].tolist(),
        }
        tooltips = [("(x, y)", "@x{0.00}, @y{0.00}")]

        if ra is not None and dec is not None:
            if ra.size != x.size or dec.size != x.size:
                raise ValueError(
                    "world coordinate columns must match pixel coordinate length"
                )
            ra_clip = ra[mask]
            dec_clip = dec[mask]
            if np.isfinite(ra_clip).any() and np.isfinite(dec_clip).any():
                data["ra"] = ra_clip.tolist()
                data["dec"] = dec_clip.tolist()
                tooltips.append(("RA, Dec", "@ra{0.000000}, @dec{0.000000}"))

        for index, col in enumerate(self.hover_cols or ()):
            values = _catalog_column_1d(self.catalog, col)
            if values.size != x.size:
                raise ValueError(
                    f"hover column {col!r} has length {values.size}; expected {x.size}"
                )
            key = f"hover_{index}"
            data[key] = values[mask].tolist()
            tooltips.append((col, f"@{key}"))

        return CatalogOverlayData(
            data=data,
            tooltips=tooltips,
            style=_catalog_style(self.style),
        )

    def draw(self, fig: BokehFigure, frame: object) -> list[GlyphRenderer]:
        """Draw this catalogue on ``fig`` and return its point renderer."""
        resolved = self.resolve(frame)
        source = ColumnDataSource(resolved.data)
        glyph = fig.scatter("x", "y", source=source, **resolved.style)
        fig.add_tools(HoverTool(renderers=[glyph], tooltips=resolved.tooltips))
        return [glyph]


class _MissingColumn(KeyError):
    """Raised internally when a catalogue has no requested column."""


def _as_1d(values: object, *, dtype: DTypeLike | None = None) -> np.ndarray:
    """Return ``values`` as a one-dimensional numpy array."""
    arr = np.asarray(values, dtype=dtype)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.ravel()


def _catalog_column(catalog: object, name: str) -> object:
    """Return one named catalogue column, or raise ``_MissingColumn``."""
    if isinstance(catalog, Mapping):
        try:
            return cast(Mapping[str, object], catalog)[name]
        except KeyError as exc:
            raise _MissingColumn(name) from exc

    colnames = getattr(catalog, "colnames", None)
    if colnames is not None and name in colnames:
        return cast(_ColumnLookup, catalog)[name]
    columns = getattr(catalog, "columns", None)
    if columns is not None and name in columns:
        return cast(_ColumnLookup, catalog)[name]
    dtype = getattr(catalog, "dtype", None)
    dtype_names = getattr(dtype, "names", None)
    if dtype_names is not None and name in dtype_names:
        return cast(_ColumnLookup, catalog)[name]
    if hasattr(catalog, name):
        return getattr(catalog, name)

    try:
        return cast(_ColumnLookup, catalog)[name]
    except Exception as exc:
        raise _MissingColumn(name) from exc


def _catalog_column_1d(catalog: object, name: str) -> np.ndarray:
    """Return one catalogue column as a one-dimensional array."""
    try:
        return _as_1d(_catalog_column(catalog, name))
    except _MissingColumn as exc:
        raise ValueError(f"hover column {name!r} is not present in catalog") from exc


def _catalog_numeric_pair(
    catalog: object, cols: tuple[str, str] | None, role: str
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return a numeric catalogue column pair if both columns are available."""
    if cols is None:
        return None
    try:
        first = _as_1d(_catalog_column(catalog, cols[0]), dtype=float)
        second = _as_1d(_catalog_column(catalog, cols[1]), dtype=float)
    except _MissingColumn:
        return None
    if first.size != second.size:
        raise ValueError(
            f"{role} columns {cols!r} have lengths {first.size} and {second.size}"
        )
    return first, second


def _call_catalog_transform(
    transform: Callable[[ArrayLike, ArrayLike], tuple[ArrayLike, ArrayLike]],
    first: np.ndarray,
    second: np.ndarray,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Call a catalogue coordinate transform and validate its array output."""
    try:
        out_first, out_second = transform(first, second)
    except Exception as exc:
        raise ValueError(
            f"{name}(a, b) failed; it must accept two equal-length 1-D arrays "
            "and return two arrays of the same length."
        ) from exc
    arr_first = _as_1d(out_first, dtype=float)
    arr_second = _as_1d(out_second, dtype=float)
    if arr_first.size != first.size or arr_second.size != first.size:
        raise ValueError(
            f"{name} returned lengths {arr_first.size} and {arr_second.size}; "
            f"expected {first.size}"
        )
    return arr_first, arr_second


def _catalog_style(style: Mapping[str, object] | None) -> dict[str, object]:
    """Merge a user catalogue marker style with defaults."""
    resolved = dict(DEFAULT_CATALOG_STYLE)
    if style is not None:
        resolved.update(style)
    return resolved
