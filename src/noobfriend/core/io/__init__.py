"""I/O helpers: HTTP session management, downloads, and JWST FITS readers."""

from noobfriend.core.io.fits import (
    read_data,
    read_dq,
    read_err,
    read_gwcs,
    read_meta,
)
from noobfriend.core.io.network import HTTPSession
from noobfriend.core.io.remote import RemoteReadError, fetch_bytes

__all__ = [
    "HTTPSession",
    "RemoteReadError",
    "fetch_bytes",
    "read_data",
    "read_dq",
    "read_err",
    "read_gwcs",
    "read_meta",
]
