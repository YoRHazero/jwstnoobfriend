"""Read JWST FITS products: metadata, GWCS, science / error / data-quality arrays.

All readers are **stateless and uncached**: every call opens the file, extracts
what it needs, copies the result out of the memory-mapped region, and closes
the file before returning. This is deliberate — caching is rarely worth its
complexity in research workflows (file-on-disk changes, memory bloat, leaked
handles) and Python variables already serve as a perfectly fine per-call
cache. If a specific call site needs memoisation, wrap it locally with
:func:`functools.cache`.

Every reader accepts either a local :class:`~pathlib.Path` or an already-open
binary file object, so a file whose bytes were fetched from a remote host can
be read straight from memory without first writing it to disk::

    read_data(BytesIO(fetch_bytes("host:/path/x_cal.fits")))

Fetching the bytes (:func:`noobfriend.core.io.fetch_bytes`) and parsing them
stay separate concerns: the readers here never touch the network.

The readers raise the underlying library exceptions directly
(:exc:`FileNotFoundError`, :exc:`KeyError`, :exc:`OSError` ...) so the caller
can react to specific failures instead of seeing every problem flattened into
a generic ``ValueError``.
"""

import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
from asdf.exceptions import AsdfPackageVersionWarning
from astropy.io import fits
from gwcs import WCS
from stdatamodels import asdf_in_fits

from noobfriend.core.display import AttrView


@contextmanager
def _silence_asdf_version_warning() -> Any:
    """Locally silence :class:`asdf.exceptions.AsdfPackageVersionWarning`.

    These warnings appear whenever the file's ASDF schema versions differ from
    the installed versions; they are noisy and almost never actionable in a
    read-only pipeline.

    Yields
    ------
    None
        Used purely for its side effect.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=AsdfPackageVersionWarning)
        yield


def read_meta(source: Path | BinaryIO) -> AttrView:
    """Read the ASDF-in-FITS metadata tree as an :class:`AttrView`.

    Parameters
    ----------
    source : Path or file-like
        Path to a local JWST FITS file, or an open binary file object holding
        its content, that embeds an ASDF metadata tree.

    Returns
    -------
    AttrView
        Attribute-style view of the ``meta`` subtree. Navigate it with dot
        access (``meta.instrument.detector``) or render it in a notebook
        for an interactive folded view.

    Raises
    ------
    FileNotFoundError
        ``source`` is a path that does not exist.
    KeyError
        The file does not contain a ``meta`` ASDF subtree.
    """
    with _silence_asdf_version_warning(), asdf_in_fits.open(source) as af:
        return AttrView(af.tree["meta"])


def read_gwcs(source: Path | BinaryIO) -> WCS:
    """Read the GWCS object stored in the ASDF metadata tree.

    Parameters
    ----------
    source : Path or file-like
        Path to a local JWST FITS file, or an open binary file object holding
        its content, that embeds a GWCS in its ASDF tree.

    Returns
    -------
    gwcs.WCS
        The world-coordinate-system object.

    Raises
    ------
    FileNotFoundError
        ``source`` is a path that does not exist.
    KeyError
        The file does not contain a ``meta.wcs`` entry (typical for products
        before the assign_wcs step has been run).
    """
    with _silence_asdf_version_warning(), asdf_in_fits.open(source) as af:
        return af.tree["meta"]["wcs"]


def _read_image_extension(source: Path | BinaryIO, ext: str) -> np.ndarray:
    """Open ``source``, copy the data of HDU ``ext`` into memory, and return it.

    The ``.copy()`` detaches the array from astropy's memory map so the file
    can be closed safely before the caller uses the result.
    """
    with fits.open(source) as hdul:
        return hdul[ext].data.copy()


def read_data(source: Path | BinaryIO) -> np.ndarray:
    """Read the ``SCI`` extension of a JWST FITS file.

    Parameters
    ----------
    source : Path or file-like
        Path to a local FITS file, or an open binary file object holding it.

    Returns
    -------
    numpy.ndarray
        The science array, fully loaded into memory (not memory-mapped).

    Raises
    ------
    FileNotFoundError
        ``source`` is a path that does not exist.
    KeyError
        The file has no ``SCI`` extension.
    """
    return _read_image_extension(source, "SCI")


def read_err(source: Path | BinaryIO) -> np.ndarray:
    """Read the ``ERR`` extension of a JWST FITS file.

    Parameters
    ----------
    source : Path or file-like
        Path to a local FITS file, or an open binary file object holding it.

    Returns
    -------
    numpy.ndarray
        The error array, fully loaded into memory (not memory-mapped).

    Raises
    ------
    FileNotFoundError
        ``source`` is a path that does not exist.
    KeyError
        The file has no ``ERR`` extension.
    """
    return _read_image_extension(source, "ERR")


def read_dq(source: Path | BinaryIO) -> np.ndarray:
    """Read the ``DQ`` (data-quality) extension of a JWST FITS file.

    Parameters
    ----------
    source : Path or file-like
        Path to a local FITS file, or an open binary file object holding it.

    Returns
    -------
    numpy.ndarray
        The data-quality bitmask array, fully loaded into memory.

    Raises
    ------
    FileNotFoundError
        ``source`` is a path that does not exist.
    KeyError
        The file has no ``DQ`` extension.
    """
    return _read_image_extension(source, "DQ")
