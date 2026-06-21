"""Shared internals for the blind grism line-finder views (book + box).

``book.extract.linefinder`` and ``box.extract.linefinder`` both drive
:class:`noobfriend.extraction.grism.linefind.GrismLineFinder`. The pieces they
share -- deriving the dispersion direction from a book's pupil, configuring a
finder for a book, and computing one exposure's heatmap -- live here, so neither
side depends on the other (the line-finder counterpart of
:mod:`noobfriend.navigation._blink`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from noobfriend.extraction.grism.linefind import GrismLineFinder
    from noobfriend.navigation.noobook._core import NooBook


def _dispersion_of(pupil: str | None) -> str | None:
    """Return the dispersion axis (``"row"``/``"column"``) from a NIRCam pupil."""
    if pupil is None:
        return None
    name = pupil.upper()
    if "GRISMR" in name:
        return "row"
    if "GRISMC" in name:
        return "column"
    return None


def configure(book: NooBook, **config: Any) -> GrismLineFinder:
    """Configure a :class:`GrismLineFinder` for ``book`` (dispersion from pupil).

    Parameters
    ----------
    book : NooBook
        A grism exposure; its ``pupil`` sets the dispersion direction.
    **config
        Detection parameters forwarded to
        :meth:`~noobfriend.extraction.grism.linefind.GrismLineFinder.configure`.

    Returns
    -------
    GrismLineFinder

    Raises
    ------
    ValueError
        If ``book`` is not a grism frame (pupil is not ``GRISM*``).
    """
    from noobfriend.extraction.grism.linefind import GrismLineFinder

    dispersion = _dispersion_of(book.pupil)
    if dispersion is None:
        raise ValueError(
            f"{book.id} is not a grism frame (pupil={book.pupil!r}); "
            "cannot configure a line finder."
        )
    return GrismLineFinder.configure(dispersion=dispersion, **config)


def exposure_heatmap(book: NooBook, **config: Any) -> np.ndarray:
    """Return one grism exposure's blind line-likelihood heatmap.

    Configures a finder for ``book`` and band-passes its ``SCI`` / ``ERR``; the
    heatmap is 1:1 with the frame (use the book's WCS to place it).

    Parameters
    ----------
    book : NooBook
        The grism exposure.
    **config
        Detection parameters forwarded to :func:`configure`.

    Returns
    -------
    numpy.ndarray
        The per-pixel band-pass SNR map.

    Raises
    ------
    ValueError
        If ``book`` is not a grism frame, or has no ``ERR`` array.
    """
    error = book.err
    if error is None:
        raise ValueError(f"grism book {book.id} has no ERR array for the line heatmap.")
    return configure(book, **config).exposure_heatmap(book.data, error)
