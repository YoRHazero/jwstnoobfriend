"""Internal: locate FITS HDUs by byte offset and rebuild a single one in memory.

A FITS file is a sequence of HDUs, each a run of 2880-byte header blocks followed
by its data padded to the next 2880-byte boundary. Because every header declares
its own data size (via ``BITPIX`` / ``NAXIS`` / ``PCOUNT`` / ``GCOUNT``), the byte
offset of each HDU can be computed by walking the headers alone -- without reading
the (large) data in between. :func:`parse_front` does that walk over a prefix of
the file, which is enough to locate the early extensions (notably ``SCI``).

Once a target extension's byte range is known, :func:`wrap_image` rebuilds a tiny
valid FITS file in memory -- an empty primary HDU followed by that one extension's
original header and raw data -- so the existing :mod:`astropy.io.fits` readers can
decode it with their full semantics (dtype, ``BZERO`` / ``BSCALE`` scaling, ...)
exactly as if the whole file had been read. Nothing here touches the network.
"""

from dataclasses import dataclass, field

from astropy.io import fits

_BLOCK: int = 2880


@dataclass(frozen=True)
class HduSpan:
    """The byte layout and header of one HDU.

    Attributes
    ----------
    name : str
        ``"PRIMARY"`` for the primary HDU, otherwise the HDU's ``EXTNAME``
        (or ``""`` if it has none).
    hdr_loc : int
        Byte offset where the header starts.
    data_loc : int
        Byte offset where the data starts (just past the header blocks).
    data_span : int
        Size of the data in bytes, padded to a 2880-byte boundary (``0`` for a
        header-only HDU).
    header : astropy.io.fits.Header
        The parsed header.
    """

    name: str
    hdr_loc: int
    data_loc: int
    data_span: int
    header: fits.Header


@dataclass
class FitsLayout:
    """The cached byte layout a NooBook needs to read its pixel extensions.

    Only the ``SCI`` span is resolved eagerly (from the file's leading header
    block); other extensions (``ERR`` / ``DQ``) are discovered on demand by
    walking forward from ``SCI`` and then memoised in :attr:`extensions`, so a
    book that re-reads them does not re-walk the headers each time.

    Attributes
    ----------
    sci : HduSpan or None
        The ``SCI`` extension's layout, or ``None`` if the file has no ``SCI``.
    shape : tuple of int or None
        The ``SCI`` array shape in NumPy order ``(NAXIS2, NAXIS1, ...)``, or
        ``None`` when there is no ``SCI``.
    extensions : dict of str to (HduSpan or None)
        Memo of extensions discovered after ``SCI``. A key's presence means it
        has been looked up; a ``None`` value records that the extension is
        absent (so a missing extension is not re-searched on every access).
    """

    sci: HduSpan | None
    shape: tuple[int, ...] | None
    extensions: dict[str, HduSpan | None] = field(default_factory=dict)


def data_span(header: fits.Header) -> int:
    """Return the padded byte size of the data unit declared by ``header``.

    Implements the standard FITS data-size rule
    ``|BITPIX| / 8 * GCOUNT * (PCOUNT + N1*N2*...)`` rounded up to the next
    2880-byte block, which covers both image HDUs (``GCOUNT=1``, ``PCOUNT=0``)
    and binary-table HDUs such as the ASDF metadata HDU (where ``PCOUNT`` is the
    heap size).

    Parameters
    ----------
    header : astropy.io.fits.Header
        A parsed HDU header.

    Returns
    -------
    int
        The data unit's size in bytes including trailing pad (``0`` when
        ``NAXIS`` is ``0``).
    """
    bitpix = int(header["BITPIX"])
    naxis = int(header["NAXIS"])
    if naxis == 0:
        nelem = 0
    else:
        nelem = 1
        for axis in range(1, naxis + 1):
            nelem *= int(header[f"NAXIS{axis}"])
    gcount = int(header.get("GCOUNT", 1))
    pcount = int(header.get("PCOUNT", 0))
    nbytes = abs(bitpix) // 8 * gcount * (pcount + nelem)
    return nbytes + (-nbytes) % _BLOCK


def _header_end(buf: bytes, start: int) -> int | None:
    """Return the offset just past the header that begins at ``start``.

    Scans 2880-byte blocks from ``start`` until one contains the ``END`` card.
    Returns ``None`` if ``buf`` ends before the ``END`` card is found (the prefix
    was too short to hold this whole header).
    """
    off = start
    while off + _BLOCK <= len(buf):
        block = buf[off : off + _BLOCK]
        for card in range(0, _BLOCK, 80):
            if block[card : card + 80].rstrip(b" ") == b"END":
                return off + _BLOCK
        off += _BLOCK
    return None


def parse_front(buf: bytes) -> list[HduSpan]:
    """Walk the HDU headers contained in a leading prefix of a FITS file.

    Reads each header, computes its data size to skip to the next header, and
    stops when the prefix runs out (a header is incomplete, or the next header
    would start past the end of ``buf``). A modest prefix (tens of KiB) is enough
    to reach the ``SCI`` extension; later extensions whose headers fall beyond the
    prefix are simply not returned.

    Parameters
    ----------
    buf : bytes
        A prefix of the file, starting at byte ``0``.

    Returns
    -------
    list of HduSpan
        The HDUs whose headers lie fully within ``buf``, in file order.
    """
    spans: list[HduSpan] = []
    off = 0
    first = True
    while off + _BLOCK <= len(buf):
        end = _header_end(buf, off)
        if end is None:
            break
        header = fits.Header.fromstring(buf[off:end].decode("ascii"))
        span = data_span(header)
        name = "PRIMARY" if first else str(header.get("EXTNAME", ""))
        spans.append(
            HduSpan(name=name, hdr_loc=off, data_loc=end, data_span=span, header=header)
        )
        off = end + span
        first = False
    return spans


def parse_header_at(buf: bytes, base: int) -> HduSpan | None:
    """Parse the single HDU header that starts at the beginning of ``buf``.

    Used to discover an extension (e.g. ``ERR`` / ``DQ``) whose header is not in
    the file's leading prefix: the caller fetches a small range at the expected
    offset and passes it here. Offsets in the result are absolute, shifted by
    ``base`` (the file offset of ``buf[0]``).

    Parameters
    ----------
    buf : bytes
        Bytes starting exactly at an HDU header boundary.
    base : int
        The file offset that ``buf[0]`` corresponds to.

    Returns
    -------
    HduSpan or None
        The parsed HDU's layout, or ``None`` if ``buf`` ends before the header's
        ``END`` card (the probe was too short, or end-of-file was reached).
    """
    end = _header_end(buf, 0)
    if end is None:
        return None
    header = fits.Header.fromstring(buf[:end].decode("ascii"))
    span = data_span(header)
    return HduSpan(
        name=str(header.get("EXTNAME", "")),
        hdr_loc=base,
        data_loc=base + end,
        data_span=span,
        header=header,
    )


def extract_asdf(tail: bytes) -> bytes:
    """Return the embedded ASDF tree bytes from the tail of a JWST product.

    A JWST product stores its metadata (and GWCS) as an ASDF tree in the heap of
    a trailing ``BINTABLE`` HDU named ``ASDF`` -- the last HDU in the file. Given
    enough trailing bytes, this locates that HDU's data and returns the embedded
    ASDF file (from its ``#ASDF`` magic onward), which :func:`asdf.open` can parse
    lazily to read ``meta`` and ``meta.wcs`` *without* the (large, separate) image
    extensions the tree references.

    Parameters
    ----------
    tail : bytes
        The trailing bytes of the file; must reach back far enough to include the
        whole ASDF HDU (header + data).

    Returns
    -------
    bytes
        The embedded ASDF file's bytes, starting at its ``#ASDF`` magic.

    Raises
    ------
    KeyError
        ``tail`` does not contain a complete ``ASDF`` ``BINTABLE`` HDU (it was too
        short, or this is not an ASDF-in-FITS product).
    """
    start = tail.rfind(b"XTENSION= 'BINTABLE'")
    if start == -1:
        raise KeyError("no ASDF (BINTABLE) HDU found in the file tail")
    end = _header_end(tail, start)
    if end is None:
        raise KeyError("incomplete ASDF HDU header in the file tail")
    magic = tail.find(b"#ASDF", end)
    if magic == -1:
        raise KeyError("no ASDF tree found after the ASDF HDU header")
    return tail[magic:]


def _primary_header_bytes() -> bytes:
    """Return the bytes of a minimal, extension-allowing primary header."""
    header = fits.PrimaryHDU().header
    header["EXTEND"] = True
    return header.tostring().encode("ascii")


def wrap_image(header: fits.Header, data: bytes) -> bytes:
    """Rebuild a one-extension FITS file in memory from a header and raw data.

    Produces ``empty primary HDU + (header, data)`` as FITS bytes, so the result
    can be handed to :func:`astropy.io.fits.open` and the extension read back with
    full decoding semantics. The single extension lands at index ``1``.

    Parameters
    ----------
    header : astropy.io.fits.Header
        The extension's original header (as returned in :attr:`HduSpan.header`).
    data : bytes
        The extension's raw data bytes. Bytes beyond :func:`data_span` are
        ignored; a short input is zero-padded up to it.

    Returns
    -------
    bytes
        A complete, valid FITS file holding just this extension.
    """
    span = data_span(header)
    body = data[:span]
    if len(body) < span:
        body = body + b"\x00" * (span - len(body))
    return _primary_header_bytes() + header.tostring().encode("ascii") + body
