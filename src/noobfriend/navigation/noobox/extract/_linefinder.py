"""The :class:`BoxLineFinder`: blind grism line search across a box's exposures.

Obtained only through
:meth:`BoxExtract.linefinder <noobfriend.navigation.noobox.extract._core.BoxExtract.linefinder>`
(``box.extract.linefinder(...)``); not constructed directly. ``BoxLineFinder``
drives :class:`~noobfriend.extraction.grism.linefind.GrismLineFinder` over the
box's grism exposures: it groups them into same-field dither sets and, per group,
**deep-combines the raw frames** (reproject + median in data space) into one
union-footprint heatmap. The per-exposure heatmaps are a collect-only QA
by-product (NOT what the combined heatmap is built from).

Grouping defaults to ``(observation, visit, detector, pupil)`` -- the dithers of
one pointing at one roll. The combine reprojects raw data onto a sky grid, so a
group must share position angle (a line lands at a roll-dependent sky position):
position angle tracks ``observation`` in the FRESCO layout, so the default keeps
observations apart. (This differs from ``BoxGrism.combine``, which stacks
source-anchored rectified spectra and is therefore PA-agnostic.)

The products (:class:`~noobfriend.extraction.grism.linefind.CombinedHeatmap`,
:class:`~noobfriend.extraction.grism.linefind.Candidate`) are navigation-free;
``BoxLineFinder`` is the navigation-side driver.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from noobfriend.extraction.grism.linefind import Candidate, CombinedHeatmap
    from noobfriend.navigation.noobook._core import NooBook
    from noobfriend.navigation.noobox._core import NooBox


def _default_group(book: NooBook) -> tuple[Any, ...]:
    """Default same-field dither key: ``(observation, visit, detector, pupil)``."""
    return (book.observation, book.visit, book.detector, book.pupil)


def _build(
    box: NooBox,
    *,
    group_by: Any = None,
    skip_missing_wcs: bool = True,
    probe: bool = True,
    **config: Any,
) -> BoxLineFinder:
    """Build a :class:`BoxLineFinder` (see :meth:`BoxExtract.linefinder`)."""
    from noobfriend.navigation._linefinder import _dispersion_of

    if group_by is None:
        keyfn = _default_group
    elif isinstance(group_by, str):
        keyfn = lambda book: getattr(book, group_by)  # noqa: E731
    else:
        keyfn = group_by

    groups: dict[Any, list[NooBook]] = {}
    for original in box:
        book = original.probe() if probe and not original.is_probed else original
        if _dispersion_of(book.pupil) is None:
            continue  # not a grism frame
        if book.footprint is None:
            if skip_missing_wcs:
                continue  # no WCS -> cannot reproject for combine
            raise ValueError(
                f"{book.id} is a grism frame with no footprint "
                "(assign a WCS or probe the book)."
            )
        groups.setdefault(keyfn(book), []).append(book)

    if not groups:
        raise ValueError("no grism (pupil=GRISM*) frames in the box.")
    return BoxLineFinder(box=box, groups=groups, config=config)


class BoxLineFinder:
    """A blind grism line search over a box, grouped into same-field dither sets.

    Construct it with ``box.extract.linefinder(...)`` rather than directly.
    :attr:`groups` is resolved up front from resident metadata (no pixels);
    :attr:`heatmaps` (the deep per-group combine) and :attr:`exposure_heatmaps`
    (per-frame QA) read pixels lazily on first access.

    Parameters
    ----------
    box : NooBox
        The collection whose grism exposures are searched.
    groups : dict of {hashable: list of NooBook}
        The grism exposures partitioned into same-field dither sets.
    config : dict
        Detection parameters forwarded to
        :meth:`~noobfriend.extraction.grism.linefind.GrismLineFinder.configure`.
    """

    def __init__(
        self,
        *,
        box: NooBox,
        groups: dict[Any, list[NooBook]],
        config: dict[str, Any],
    ) -> None:
        """See the class docstring; built by ``box.extract.linefinder(...)``."""
        self._box = box
        self._groups = groups
        self._config = config
        self._heatmaps: dict[Any, CombinedHeatmap] | None = None
        self._exposure: dict[NooBook, np.ndarray] | None = None

    @property
    def groups(self) -> dict[Any, list[NooBook]]:
        """The same-field dither sets, keyed by the grouping value (no pixels)."""
        return {key: list(books) for key, books in self._groups.items()}

    @staticmethod
    def _load_for(books_by_id: dict[str, NooBook]) -> Any:
        """Return a ``load(id) -> (data, error)`` over the given books."""

        def load(book_id: str) -> tuple[Any, Any]:
            book = books_by_id[book_id]
            error = book.err
            if error is None:
                raise ValueError(f"grism book {book.id} has no ERR array to combine.")
            return book.data, error

        return load

    @property
    def heatmaps(self) -> dict[Any, CombinedHeatmap]:
        """Deep combined heatmap per group (reads pixels, reprojects + combines).

        Each value is a
        :class:`~noobfriend.extraction.grism.linefind.CombinedHeatmap` on the
        group's union grid -- the **primary product**. Computed on first access
        and cached.

        Returns
        -------
        dict of {hashable: CombinedHeatmap}
        """
        if self._heatmaps is None:
            from noobfriend.extraction.grism import FrameMeta

            from noobfriend.navigation._linefinder import configure

            out: dict[Any, CombinedHeatmap] = {}
            for key, books in self._groups.items():
                finder = configure(books[0], **self._config)
                metas = [
                    FrameMeta(id=b.id, wcs=b.wcs, shape=tuple(b.shape[-2:]))
                    for b in books
                ]
                load = self._load_for({b.id: b for b in books})
                out[key] = finder.combine(metas, load)
            self._heatmaps = out
        return self._heatmaps

    @property
    def exposure_heatmaps(self) -> dict[NooBook, np.ndarray]:
        """Per-exposure heatmaps, keyed by NooBook (collect-only QA, reads pixels)."""
        if self._exposure is None:
            from noobfriend.navigation._linefinder import exposure_heatmap

            self._exposure = {
                book: exposure_heatmap(book, **self._config)
                for books in self._groups.values()
                for book in books
            }
        return self._exposure

    def catalog(self) -> dict[Any, list[Candidate]]:
        """Peak-find each group's combined heatmap into a candidate list.

        Candidates are in the group's union-grid pixels; map them to the sky with
        the corresponding ``CombinedHeatmap.grid`` (its ``reference_wcs`` and
        offsets) when needed.

        Returns
        -------
        dict of {hashable: list of Candidate}
        """
        from noobfriend.navigation._linefinder import configure

        return {
            key: configure(self._groups[key][0], **self._config).catalog(
                heatmap.heatmap
            )
            for key, heatmap in self.heatmaps.items()
        }

    def plot(self, key: Any = None, **kwargs: Any) -> Any:
        """Display one group's combined heatmap (Bokeh ``imshow``).

        Parameters
        ----------
        key : hashable, optional
            Which group to plot. Required when there is more than one group.
        **kwargs
            Forwarded to :func:`noobfriend.core.display.plot.imshow`.

        Returns
        -------
        Any
            The Bokeh image handle.
        """
        from noobfriend.core.display.plot import imshow

        heatmaps = self.heatmaps
        if key is None:
            if len(heatmaps) != 1:
                raise ValueError(
                    f"{len(heatmaps)} groups; pass key= to choose one. "
                    f"keys: {list(heatmaps)}"
                )
            key = next(iter(heatmaps))
        kwargs.setdefault("title", f"line heatmap {key}")
        return imshow(heatmaps[key].heatmap, **kwargs)

    def __len__(self) -> int:
        """Return the number of groups."""
        return len(self._groups)

    def __repr__(self) -> str:
        """Return a one-line summary: group count and group sizes."""
        sizes = sorted(len(books) for books in self._groups.values())
        return f"BoxLineFinder({len(self._groups)} group(s), sizes {sizes})"
