"""Read JWST FITS products through a :class:`ByteAccessor`, byte-frugally.

Each reader pulls only the bytes it needs rather than the whole file:

* :func:`read_meta` / :func:`read_gwcs` read just the file's *tail* -- a JWST
  product's metadata (and GWCS) live in a trailing ``ASDF`` HDU -- and parse the
  embedded ASDF tree lazily, so the large image extensions it references are
  never fetched.
* :func:`read_layout` reads the file's leading header block to locate the ``SCI``
  extension; :func:`read_data` / :func:`read_err` / :func:`read_dq` /
  :func:`read_noise` then read just that one extension's byte range.

Decoding always goes through :mod:`astropy.io.fits` on a tiny in-memory FITS
rebuilt from the fetched header and data (see
:func:`noobfriend.core.io._layout.wrap_image`), so dtype / ``BZERO`` / ``BSCALE``
semantics are identical to reading the whole file.

The readers are **stateless and uncached**: every call asks the accessor afresh.
Whether those reads hit the local disk or a remote host -- and whether they are
cached -- is entirely the accessor's concern (see
:mod:`noobfriend.core.io.accessor`); the readers here never branch on location.

The readers raise the underlying library exceptions directly
(:exc:`KeyError` for a missing extension, :exc:`OSError`, ...) so the caller can
react to specific failures.
"""

import warnings
from collections.abc import Mapping
from contextlib import contextmanager
from io import BytesIO
from typing import Any

import asdf
import numpy as np
from asdf.exceptions import AsdfBlockIndexWarning, AsdfPackageVersionWarning
from astropy.io import fits
from gwcs import WCS
from stdatamodels import asdf_in_fits

from noobfriend.core.display import AttrView
from noobfriend.core.imgutils import NoiseAutocovariance
from noobfriend.core.io._layout import (
    FitsLayout,
    HduSpan,
    extract_asdf,
    parse_front,
    parse_header_at,
    wrap_image,
)
from noobfriend.core.io.accessor import ByteAccessor

#: Trailing bytes pulled to reach the ASDF metadata HDU (comfortably exceeds the
#: observed ASDF size of a few tens of KiB).
_TAIL: int = 1 << 18
#: Leading bytes pulled to reach the ``SCI`` header (well past its location).
_HEAD: int = 1 << 17
#: Bytes pulled to read a single extension header during ``ERR`` / ``DQ`` discovery.
_PROBE: int = 1 << 15


@contextmanager
def _silence_asdf_read_warnings() -> Any:
    """Locally silence the two noisy, non-actionable ASDF read warnings.

    - :class:`~asdf.exceptions.AsdfPackageVersionWarning`: the file's ASDF
      schema versions differ from the installed ones -- noisy, almost never
      actionable in a read-only pipeline.
    - :class:`~asdf.exceptions.AsdfBlockIndexWarning`: the block index of an
      ASDF *extracted from its FITS embedding* (as :func:`_meta_tree` does)
      does not parse, so asdf falls back to serial block reading -- which is
      always correct. It never fires for the whole-file path, so silencing it
      cannot hide a real read failure.

    Yields
    ------
    None
        Used purely for its side effect.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=AsdfPackageVersionWarning)
        warnings.filterwarnings("ignore", category=AsdfBlockIndexWarning)
        yield


def _meta_tree(acc: ByteAccessor) -> dict:
    """Return the ASDF ``meta`` subtree, reading only the file tail when possible.

    Pulls the trailing bytes, extracts the embedded ASDF file, and parses it with
    a lazy :func:`asdf.open` so the (separate) image extensions the tree references
    are not resolved. Falls back to reading the whole file with
    :func:`stdatamodels.asdf_in_fits.open` only when the ASDF HDU does not fit in
    the tail window.
    """
    try:
        asdf_bytes = extract_asdf(acc.read_tail(_TAIL))
    except KeyError:
        with _silence_asdf_read_warnings(), asdf_in_fits.open(acc.open()) as af:
            return af.tree["meta"]
    with (
        _silence_asdf_read_warnings(),
        asdf.open(BytesIO(asdf_bytes), lazy_load=True) as af,
    ):
        return af.tree["meta"]


def read_meta(acc: ByteAccessor) -> AttrView:
    """Read the ASDF-in-FITS metadata tree as an :class:`AttrView`.

    Parameters
    ----------
    acc : ByteAccessor
        Accessor for a JWST FITS product that embeds an ASDF metadata tree.

    Returns
    -------
    AttrView
        Attribute-style view of the ``meta`` subtree.

    Raises
    ------
    KeyError
        The file does not contain a ``meta`` ASDF subtree.
    """
    return AttrView(_meta_tree(acc))


def read_gwcs(acc: ByteAccessor) -> WCS:
    """Read the GWCS object stored in the ASDF metadata tree.

    Parameters
    ----------
    acc : ByteAccessor
        Accessor for a JWST FITS product that embeds a GWCS in its ASDF tree.

    Returns
    -------
    gwcs.WCS
        The world-coordinate-system object.

    Raises
    ------
    KeyError
        The file does not contain a ``meta.wcs`` entry (typical for products
        before the assign_wcs step has been run).
    """
    wcs = _meta_tree(acc).get("wcs")
    if wcs is None:
        raise KeyError("meta.wcs")
    return wcs


def read_meta_and_gwcs(acc: ByteAccessor) -> tuple[AttrView, WCS | None]:
    """Read the ``meta`` tree and its GWCS together, parsing the tail only once.

    Parameters
    ----------
    acc : ByteAccessor
        Accessor for a JWST FITS product.

    Returns
    -------
    AttrView
        The ``meta`` subtree.
    gwcs.WCS or None
        The GWCS, or ``None`` if the product has no assigned WCS yet.

    Raises
    ------
    KeyError
        The file does not contain a ``meta`` ASDF subtree.
    """
    meta = _meta_tree(acc)
    return AttrView(meta), meta.get("wcs")


def read_layout(acc: ByteAccessor) -> FitsLayout:
    """Resolve the ``SCI`` byte layout (and shape) from the file's leading header.

    Parameters
    ----------
    acc : ByteAccessor
        Accessor for a JWST FITS product.

    Returns
    -------
    FitsLayout
        With :attr:`~FitsLayout.sci` and :attr:`~FitsLayout.shape` filled, or both
        ``None`` if the file has no ``SCI`` extension.
    """
    spans = parse_front(acc.read_range(0, _HEAD))
    sci = next((span for span in spans if span.name == "SCI"), None)
    shape: tuple[int, ...] | None = None
    if sci is not None:
        naxis = int(sci.header["NAXIS"])
        shape = tuple(int(sci.header[f"NAXIS{axis}"]) for axis in range(naxis, 0, -1))
    return FitsLayout(sci=sci, shape=shape)


def _decode_image(header: fits.Header, data: bytes) -> np.ndarray:
    """Decode one extension's raw bytes via astropy, returning a detached copy."""
    with fits.open(BytesIO(wrap_image(header, data))) as hdul:
        return hdul[1].data.copy()


def _find_extension(acc: ByteAccessor, start_loc: int, name: str) -> HduSpan | None:
    """Walk forward from ``start_loc`` to the extension named ``name``.

    Reads one small header at a time, skipping each HDU's data by its computed
    span, until the named extension is found or the file ends.
    """
    loc = start_loc
    while True:
        span = parse_header_at(acc.read_range(loc, _PROBE), loc)
        if span is None:
            return None
        if span.name == name:
            return span
        loc = span.data_loc + span.data_span


def _resolve_extension(
    acc: ByteAccessor, layout: FitsLayout, name: str
) -> HduSpan | None:
    """Locate the extension ``name`` after ``SCI``, memoising it on ``layout``.

    The first lookup walks the headers forward from ``SCI``; the result (the
    span, or ``None`` if the extension is absent) is cached in
    :attr:`FitsLayout.extensions`, so repeated ``ERR`` / ``DQ`` reads on the same
    book never re-walk.
    """
    if name not in layout.extensions:
        span = None
        if layout.sci is not None:
            start = layout.sci.data_loc + layout.sci.data_span
            span = _find_extension(acc, start, name)
        layout.extensions[name] = span
    return layout.extensions[name]


def read_data(acc: ByteAccessor, layout: FitsLayout) -> np.ndarray:
    """Read the ``SCI`` extension, fetching only its bytes.

    Parameters
    ----------
    acc : ByteAccessor
        Accessor for a JWST FITS product.
    layout : FitsLayout
        The product's layout, as returned by :func:`read_layout`.

    Returns
    -------
    numpy.ndarray
        The science array, fully loaded into memory.

    Raises
    ------
    KeyError
        The file has no ``SCI`` extension.
    """
    if layout.sci is None:
        raise KeyError("SCI")
    sci = layout.sci
    return _decode_image(sci.header, acc.read_range(sci.data_loc, sci.data_span))


def read_err(acc: ByteAccessor, layout: FitsLayout) -> np.ndarray:
    """Read the ``ERR`` extension, fetching only its bytes.

    Parameters
    ----------
    acc : ByteAccessor
        Accessor for a JWST FITS product.
    layout : FitsLayout
        The product's layout, as returned by :func:`read_layout`.

    Returns
    -------
    numpy.ndarray
        The error array, fully loaded into memory.

    Raises
    ------
    KeyError
        The file has no ``ERR`` extension.
    """
    span = _resolve_extension(acc, layout, "ERR")
    if span is None:
        raise KeyError("ERR")
    return _decode_image(span.header, acc.read_range(span.data_loc, span.data_span))


def read_dq(acc: ByteAccessor, layout: FitsLayout) -> np.ndarray:
    """Read the ``DQ`` (data-quality) extension, fetching only its bytes.

    Parameters
    ----------
    acc : ByteAccessor
        Accessor for a JWST FITS product.
    layout : FitsLayout
        The product's layout, as returned by :func:`read_layout`.

    Returns
    -------
    numpy.ndarray
        The data-quality bitmask array, fully loaded into memory.

    Raises
    ------
    KeyError
        The file has no ``DQ`` extension.
    """
    span = _resolve_extension(acc, layout, "DQ")
    if span is None:
        raise KeyError("DQ")
    return _decode_image(span.header, acc.read_range(span.data_loc, span.data_span))


def read_noise(acc: ByteAccessor, layout: FitsLayout) -> NoiseAutocovariance:
    """Read the ``NOISEKERN`` extension into a noise-autocovariance model.

    A stage-3 coadd stores its noise autocovariance ``C(d)`` as a small,
    square, odd-sided ``NOISEKERN`` image plane (see
    :func:`noobfriend.reduction.mosaic.noise_kernel` and
    :func:`write_asdf_fits`). This rebuilds the
    :class:`~noobfriend.core.imgutils.NoiseAutocovariance` from that plane,
    recovering ``max_lag`` from the window's half-width. The build-time
    ``n_pairs`` diagnostic and the hybrid ``error_var`` are not persisted, so
    both come back ``None``: the model drives the per-pixel variance and the
    *stationary*
    :func:`~noobfriend.core.imgutils.correlated_sum_variance`; the hybrid
    (depth-aware) path additionally needs an ``error_var``.

    Parameters
    ----------
    acc : ByteAccessor
        Accessor for a JWST FITS product.
    layout : FitsLayout
        The product's layout, as returned by :func:`read_layout`.

    Returns
    -------
    NoiseAutocovariance
        With ``n_pairs`` and ``error_var`` set to ``None``.

    Raises
    ------
    KeyError
        The file has no ``NOISEKERN`` extension.
    ValueError
        The ``NOISEKERN`` plane is not a square, odd-sided lag window.
    """
    span = _resolve_extension(acc, layout, "NOISEKERN")
    if span is None:
        raise KeyError("NOISEKERN")
    cov = _decode_image(span.header, acc.read_range(span.data_loc, span.data_span))
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] % 2 == 0:
        raise ValueError(
            f"NOISEKERN must be a square, odd-sided lag window; got shape {cov.shape}."
        )
    max_lag = (cov.shape[0] - 1) // 2
    return NoiseAutocovariance(cov=cov, max_lag=max_lag, n_pairs=None, error_var=None)


def write_asdf_fits(planes: Mapping[str, np.ndarray], tree: Mapping[str, Any]) -> bytes:
    """Assemble image planes plus an embedded ASDF tree into FITS bytes.

    The write counterpart of :func:`read_meta_and_gwcs`: a ``PrimaryHDU``
    followed by one ``ImageHDU`` per named plane (``SCI`` / ``ERR`` / ``WHT`` /
    ...), with ``tree`` embedded as a trailing ASDF HDU via
    :mod:`stdatamodels.asdf_in_fits`. ``tree["meta"]`` carries what the readers
    expect -- ``wcs`` (a GWCS) and ``instrument`` fields -- so the product is
    read back through the same tail-ASDF path with no jwst dependency. The GWCS
    should be built from stock ``astropy`` models (e.g.
    :func:`noobfriend.reduction.mosaic.tile_gwcs`) so it serialises with standard
    ``asdf-astropy`` tags.

    Parameters
    ----------
    planes : mapping of str to numpy.ndarray
        Image extensions in order; the key is the ``EXTNAME``.
    tree : mapping
        The ASDF tree to embed (typically ``{"meta": {"wcs": ..., ...}}``).

    Returns
    -------
    bytes
        The complete FITS file, ready for
        :func:`noobfriend.core.io.write_bytes`.
    """
    hdus: list[fits.hdu.base._BaseHDU] = [fits.PrimaryHDU()]
    for name, array in planes.items():
        hdus.append(fits.ImageHDU(np.asarray(array), name=name))
    hdulist = asdf_in_fits.to_hdulist(dict(tree), fits.HDUList(hdus))
    buffer = BytesIO()
    hdulist.writeto(buffer)
    return buffer.getvalue()
