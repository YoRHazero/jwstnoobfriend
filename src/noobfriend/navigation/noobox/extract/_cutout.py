"""The :class:`BoxCutout`: one stamp per box member covering a sky position.

``BoxCutout`` is obtained **only** through
:meth:`BoxExtract.cutout <noobfriend.navigation.noobox.extract._core.BoxExtract.cutout>`
(``box.extract.cutout(ra, dec, ...)``); it is not meant to be constructed
directly. It holds a postage stamp for every
:class:`~noobfriend.navigation.NooBook` whose footprint covers the requested
position, keyed by the originating book so each stamp's provenance (filter,
detector, stage, WCS) travels with its pixels.

Because the result keys on :class:`~noobfriend.navigation.NooBook`, it depends
on navigation and therefore lives here rather than in
:mod:`noobfriend.extraction` -- the deliberate counterpart to the navigation-free
:class:`~noobfriend.extraction.psf.SourceExtractor`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from noobfriend.extraction.cutout import Cutout
    from noobfriend.navigation.noobook._core import NooBook
    from noobfriend.navigation.noobox._core import NooBox


def _tracked(books: list["NooBook"], progress: bool) -> Iterable["NooBook"]:
    """Wrap ``books`` in a progress bar when ``progress`` is set."""
    if not progress:
        return books
    from noobfriend.core.display import track

    return track(books, "Cutting out")


def _reference_book(books: list["NooBook"], on: "NooBook | str | None") -> "NooBook":
    """Pick the book whose WCS defines the common reprojection grid.

    With ``on`` given it selects that book (a :class:`NooBook`, or a string
    matching an :attr:`~NooBook.id` or :attr:`~NooBook.stage`); otherwise it
    prefers a stage-3 mosaic (level 3) and falls back to the first covering book.
    """
    if on is not None:
        from noobfriend.navigation.noobook._core import NooBook

        if isinstance(on, NooBook):
            return on
        for book in books:
            if book.id == on or book.stage == on:
                return book
        raise ValueError(f"on={on!r} matched no covering book in the box.")
    for book in books:
        if book.level == 3:
            return book
    return books[0]


def _build(
    box: "NooBox",
    ra: float,
    dec: float,
    *,
    half: int | None = None,
    half_width: int | None = None,
    half_height: int | None = None,
    left: int | None = None,
    right: int | None = None,
    top: int | None = None,
    bottom: int | None = None,
    half_world: float | None = None,
    on: "NooBook | str | None" = None,
    reproject: bool = True,
    with_err: bool = False,
    skip_missing_wcs: bool = True,
    probe: bool = True,
    progress: bool = True,
    coarse_step: tuple[int, int] | None = None,
    auto_pad: bool = True,
) -> "BoxCutout":
    """Build a :class:`BoxCutout` (see :meth:`BoxExtract.cutout`)."""
    from noobfriend.extraction.cutout import Cutout

    size: dict[str, Any] = {
        "half": half,
        "half_width": half_width,
        "half_height": half_height,
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
        "half_world": half_world,
    }

    covering: list[NooBook] = []
    for original in box:
        book = original.probe() if probe and not original.is_probed else original
        footprint = book.footprint
        if footprint is None:
            if skip_missing_wcs:
                continue
            raise ValueError(
                f"{book.id} has no footprint to test coverage "
                "(assign a WCS or probe the book)."
            )
        if footprint.contains(ra, dec):
            covering.append(book)

    if not covering:
        raise ValueError(f"no book in the box covers (ra, dec) = ({ra}, {dec}).")

    stamps: dict[NooBook, np.ndarray] = {}
    errs: dict[NooBook, np.ndarray] | None = {} if with_err else None

    if not reproject:
        for book in _tracked(covering, progress):
            cut = Cutout.from_world(book.wcs, ra, dec, **size)
            stamps[book] = cut.array(book.data)
            if errs is not None and book.err is not None:
                errs[book] = cut.array(book.err)
        return BoxCutout(
            ra=ra, dec=dec, grid=None, stamps=stamps, weights=None, errs=errs
        )

    reference = _reference_book(covering, on)
    reference_wcs = reference.wcs
    if reference_wcs is None:
        raise ValueError(
            f"reference book {reference.id} has no WCS for the cutout grid."
        )
    grid = Cutout.from_world(reference_wcs, ra, dec, **size)
    if coarse_step is not None and auto_pad:
        padded = grid.geometry.padded_to_multiple(*coarse_step)
        grid = Cutout.from_bounds(
            reference_wcs, x_bounds=padded.x_bounds, y_bounds=padded.y_bounds
        )
    weights: dict[NooBook, np.ndarray] = {}
    for book in _tracked(covering, progress):
        image, _footprint, weight = grid.reproject(
            book.data, book.wcs, coarse_step=coarse_step
        )
        stamps[book] = image
        weights[book] = weight
        if errs is not None and book.err is not None:
            err_image, _, _ = grid.reproject(
                book.err, book.wcs, coarse_step=coarse_step
            )
            errs[book] = err_image
    return BoxCutout(
        ra=ra, dec=dec, grid=grid, stamps=stamps, weights=weights, errs=errs
    )


class BoxCutout:
    """Postage stamps of one sky position across the books that cover it.

    Each entry is a :class:`~noobfriend.navigation.NooBook` that covers the
    requested ``(ra, dec)`` mapped to the cutout extracted from it. In the common
    -grid mode (``reproject=True``) every stamp is resampled onto a single shared
    :attr:`grid`, so the stamps are aligned and can be blinked or coadded; in the
    native mode (``reproject=False``) each stamp is an axis-aligned slice on its
    own book's grid and :attr:`grid` is ``None``.

    Construct it with ``box.extract.cutout(...)`` rather than directly.

    Parameters
    ----------
    ra, dec : float
        The cutout centre in degrees.
    grid : Cutout or None
        The shared target grid the stamps were reprojected onto, or ``None`` in
        native mode. Reusable to reproject further arrays onto the same grid.
    stamps : dict of {NooBook: numpy.ndarray}
        One stamp per covering book, in box iteration order.
    weights : dict of {NooBook: numpy.ndarray} or None
        Per-stamp reprojection weights (overlap fraction over valid pixels);
        ``None`` in native mode.
    errs : dict of {NooBook: numpy.ndarray} or None
        Per-stamp error stamps when ``with_err`` was set, else ``None``. Missing
        for books without an ``ERR`` array. Display-grade for reprojected data.
    """

    def __init__(
        self,
        *,
        ra: float,
        dec: float,
        grid: "Cutout | None",
        stamps: dict["NooBook", np.ndarray],
        weights: dict["NooBook", np.ndarray] | None = None,
        errs: dict["NooBook", np.ndarray] | None = None,
    ) -> None:
        """See the class docstring; built by ``box.extract.cutout(...)``."""
        self._ra = ra
        self._dec = dec
        self._grid = grid
        self._stamps = stamps
        self._weights = weights
        self._errs = errs

    @property
    def ra(self) -> float:
        """Right ascension of the cutout centre, in degrees."""
        return self._ra

    @property
    def dec(self) -> float:
        """Declination of the cutout centre, in degrees."""
        return self._dec

    @property
    def center(self) -> tuple[float, float]:
        """The ``(ra, dec)`` cutout centre, in degrees."""
        return (self._ra, self._dec)

    @property
    def grid(self) -> "Cutout | None":
        """The shared reprojection grid, or ``None`` in native mode."""
        return self._grid

    @property
    def reprojected(self) -> bool:
        """Whether the stamps share a common grid (``reproject=True``)."""
        return self._grid is not None

    @property
    def stamps(self) -> dict["NooBook", np.ndarray]:
        """The ``{book: stamp}`` mapping (do not mutate)."""
        return self._stamps

    @property
    def weights(self) -> dict["NooBook", np.ndarray] | None:
        """Per-stamp reprojection weights, or ``None`` in native mode."""
        return self._weights

    @property
    def errs(self) -> dict["NooBook", np.ndarray] | None:
        """Per-stamp error stamps, or ``None`` when ``with_err`` was not set."""
        return self._errs

    @property
    def books(self) -> list["NooBook"]:
        """The covering books, in stamp order."""
        return list(self._stamps.keys())

    @property
    def images(self) -> list[np.ndarray]:
        """The stamp arrays, in book order."""
        return list(self._stamps.values())

    @property
    def labels(self) -> list[str]:
        """The covering books' ids, in stamp order."""
        return [book.id for book in self._stamps]

    def items(self) -> list[tuple["NooBook", np.ndarray]]:
        """Return the ``(book, stamp)`` pairs."""
        return list(self._stamps.items())

    def coadd(self) -> np.ndarray:
        """Weighted mean of the aligned stamps onto the common grid.

        Stamps are combined pixel-wise weighting by :attr:`weights` (falling back
        to a valid-pixel count when weights are absent); pixels with no coverage
        are ``NaN``. Requires the common-grid mode.

        Returns
        -------
        numpy.ndarray
            The coadded stamp, of the common grid's shape.

        Raises
        ------
        ValueError
            If the stamps are not on a common grid (``reproject=False``).
        """
        if not self.reprojected:
            raise ValueError("coadd requires a common grid; build with reproject=True.")
        images = np.stack([np.nan_to_num(img, nan=0.0) for img in self.images])
        if self._weights:
            weights = np.stack(
                [np.nan_to_num(w, nan=0.0) for w in self._weights.values()]
            )
        else:
            weights = np.stack([np.isfinite(img).astype(float) for img in self.images])
        total = weights.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(total > 0, (images * weights).sum(axis=0) / total, np.nan)

    def blink(self, **kwargs: Any) -> Any:
        """Blink-compare the stamps with the core comparator.

        Forwards the stamp :attr:`images` and :attr:`labels` to
        :func:`noobfriend.core.display.plot.imshow_blink`. Most meaningful in the
        common-grid mode, where the stamps are already aligned; styling keywords
        (``vmin``, ``cmap``, ``stretch``, ...) pass straight through.

        Returns
        -------
        Any
            The display handle returned by ``imshow_blink``.
        """
        from noobfriend.core.display.plot import imshow_blink

        return imshow_blink(self.images, labels=self.labels, **kwargs)

    def __len__(self) -> int:
        """Return the number of stamps (covering books)."""
        return len(self._stamps)

    def __iter__(self) -> Iterable["NooBook"]:
        """Iterate over the covering books, like a ``{book: stamp}`` mapping."""
        return iter(self._stamps)

    def __getitem__(self, key: "NooBook | str | int") -> np.ndarray:
        """Return one stamp by book, by book id, or by positional index."""
        if isinstance(key, int):
            return list(self._stamps.values())[key]
        if isinstance(key, str):
            for book, stamp in self._stamps.items():
                if book.id == key:
                    return stamp
            raise KeyError(key)
        return self._stamps[key]

    def __repr__(self) -> str:
        """Return a one-line summary: stamp count, mode and centre."""
        mode = "reprojected" if self.reprojected else "native"
        return (
            f"BoxCutout({len(self)} stamps, {mode}, "
            f"center=({self._ra:.5f}, {self._dec:.5f}))"
        )
