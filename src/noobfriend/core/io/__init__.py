"""I/O helpers: HTTP sessions, downloads, JWST FITS readers, and data sources."""

from noobfriend.core.io.accessor import (
    ByteAccessor,
    BytesAccessor,
    LocalAccessor,
    RemoteAccessor,
    open_accessor,
)
from noobfriend.core.io.fits import (
    read_data,
    read_dq,
    read_err,
    read_gwcs,
    read_layout,
    read_meta,
    read_meta_and_gwcs,
    read_noise,
    write_asdf_fits,
)
from noobfriend.core.io.grizli_cutout import (
    GrizliCutout,
    load_grizli_cutout,
    read_grizli_cutout_bands,
)
from noobfriend.core.io.network import HTTPSession
from noobfriend.core.io.remote import (
    RemoteReadError,
    RemoteWriteError,
    fetch_bytes,
    fetch_range,
    fetch_tail,
    list_remote_dir,
    remote_exists,
    remote_makedirs,
    write_bytes,
)

__all__ = [
    "ByteAccessor",
    "BytesAccessor",
    "GrizliCutout",
    "HTTPSession",
    "LocalAccessor",
    "RemoteAccessor",
    "RemoteReadError",
    "RemoteWriteError",
    "fetch_bytes",
    "fetch_range",
    "fetch_tail",
    "list_remote_dir",
    "load_grizli_cutout",
    "open_accessor",
    "read_data",
    "read_dq",
    "read_err",
    "read_grizli_cutout_bands",
    "read_gwcs",
    "read_layout",
    "read_meta",
    "read_meta_and_gwcs",
    "read_noise",
    "remote_exists",
    "remote_makedirs",
    "write_asdf_fits",
    "write_bytes",
]
