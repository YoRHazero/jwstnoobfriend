"""The user-facing :class:`NooBook`: one JWST product file as a rich object."""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Self

import numpy as np
from gwcs import WCS
from pydantic import BaseModel, ConfigDict, PrivateAttr

if TYPE_CHECKING:
    from noobfriend.core.wcs import NoobWCS

from noobfriend.core.display import AttrView
from noobfriend.core.imgutils import NoiseAutocovariance
from noobfriend.core.io import (
    ByteAccessor,
    BytesAccessor,
    fetch_bytes,
    open_accessor,
    read_data,
    read_dq,
    read_err,
    read_gwcs,
    read_layout,
    read_meta,
    read_meta_and_gwcs,
    read_noise,
)
from noobfriend.core.io._layout import FitsLayout
from noobfriend.navigation._footprint import Footprint
from noobfriend.navigation._store import ByteStore, StoreAccessor
from noobfriend.navigation.noobook.extract import BookExtract
from noobfriend.navigation.noobook._naming import (
    parse_exposure_name,
    parse_program_id,
    strip_fits_suffix,
)
from noobfriend.navigation.noobook._viz import BookViz


def _basename(location: str) -> str:
    """Return the file-name component of a local or ``host:path`` location."""
    tail = location.rsplit("/", 1)[-1]
    return tail.rsplit(":", 1)[-1]


def _instrument_field(meta: AttrView, name: str) -> str | None:
    """Read ``meta.instrument.<name>`` as a string, or ``None`` if absent."""
    try:
        value: Any = getattr(meta.instrument, name)
    except AttributeError:
        return None
    return None if value is None else str(value)


def _make_accessor(
    store: "ByteStore | None", book_id: str, location: str
) -> ByteAccessor:
    """Return the accessor a book reads through: store-backed when bound, else direct."""
    if store is not None:
        return StoreAccessor(store, book_id, location)
    return open_accessor(location)


def _read_header_fields(acc: ByteAccessor) -> tuple[dict[str, Any], FitsLayout]:
    """Extract the header-derived fields and the SCI byte layout from a file.

    Reads only the file's leading header block (for the ``SCI`` shape and offset)
    and its trailing ASDF metadata -- never the large pixel extensions.

    Parameters
    ----------
    acc : ByteAccessor
        Accessor for a JWST FITS product.

    Returns
    -------
    dict
        ``pupil``, ``filter``, ``shape`` and ``footprint``, suitable for
        constructing or updating a :class:`NooBook`. ``footprint`` is ``None``
        when the file has no assigned WCS.
    FitsLayout
        The product's ``SCI`` byte layout, to cache on the book for later pixel
        reads.
    """
    layout = read_layout(acc)
    meta, wcs = read_meta_and_gwcs(acc)
    footprint = (
        Footprint.from_wcs(wcs, layout.shape)
        if wcs is not None and layout.shape is not None
        else None
    )
    fields = {
        "pupil": _instrument_field(meta, "pupil"),
        "filter": _instrument_field(meta, "filter"),
        "shape": layout.shape,
        "footprint": footprint,
    }
    return fields, layout


def _parent_ids_of(parents: "Iterable[NooBook | str] | None") -> tuple[str, ...]:
    """Return the parent-id tuple from an iterable of NooBooks and/or ids."""
    if parents is None:
        return ()
    return tuple(p.id if isinstance(p, NooBook) else str(p) for p in parents)


class NooBook(BaseModel):
    """One JWST product file, with its identity, metadata, and data access.

    A NooBook is a fully-populated, serializable record of a single file:
    name-derived identifiers (program, observation, ...), the few header-derived
    fields cheap enough to keep resident (``pupil``, ``filter``, ``shape``,
    ``footprint``), and lazy accessors for the heavy contents (``meta``,
    ``wcs``, ``data``, ``err``, ``dq``) that are read on demand and never
    pinned on the object.

    Build one three ways: :meth:`from_file` (parse the name and read the header
    once), the plain constructor (when a producing step already holds the
    fields), or :meth:`from_record` (rebuild from a persisted manifest row).
    :meth:`from_name` builds a metadata-incomplete record from the file name
    alone, for fast indexing before any file is opened.

    Capabilities that bridge to other subpackages are grouped behind namespace
    accessors rather than methods on the class itself: :attr:`viz` for plotting
    and :attr:`extract` for region extraction.

    Attributes
    ----------
    id : str
        Stable logical identity, ``"<name-without-.fits>@<stage>"``.
    location : str
        The file's ``[user@]host:path`` or local-path spec.
    stage : str
        Pipeline stage label, e.g. ``"1b"``, ``"2bi"``, ``"3a"``.
    program_id : str
        Five-digit program identifier.
    detector : str or None
        Detector name; ``None`` for stage-3 products that span detectors.
    observation, visit, ggsaa, exposure : tuple of str
        Exposure identifiers. Length one for an atomic product; the union over
        inputs for an aggregated (stage-3) product.
    parent_ids : tuple of str
        Ids of the immediate upstream products this one derives from.
    pupil, filter : str or None
        Optical elements; ``None`` until probed, or when not applicable.
    shape : tuple of int or None
        ``SCI`` array shape; ``None`` until probed.
    footprint : Footprint or None
        On-sky region; ``None`` when probed but the WCS is not yet assigned.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    location: str
    stage: str
    program_id: str
    detector: str | None = None
    observation: tuple[str, ...] = ()
    visit: tuple[str, ...] = ()
    ggsaa: tuple[str, ...] = ()
    exposure: tuple[str, ...] = ()
    parent_ids: tuple[str, ...] = ()
    pupil: str | None = None
    filter: str | None = None
    shape: tuple[int, ...] | None = None
    footprint: Footprint | None = None

    _store: ByteStore | None = PrivateAttr(default=None)
    _layout: FitsLayout | None = PrivateAttr(default=None)
    _noobwcs: "NoobWCS | None" = PrivateAttr(default=None)
    _noobwcs_compiled: bool = PrivateAttr(default=False)

    def __eq__(self, other: object) -> bool:
        """Two NooBooks are equal when they share an :attr:`id`."""
        return isinstance(other, NooBook) and other.id == self.id

    def __hash__(self) -> int:
        """Hash on :attr:`id`, so NooBooks key sets and dicts by identity."""
        return hash(self.id)

    # -- construction ---------------------------------------------------------

    @staticmethod
    def _name_identifiers(filename: str) -> dict[str, Any]:
        """Extract the name-derived identifier fields from a product file name."""
        exposure = parse_exposure_name(filename)
        if exposure is not None:
            return {
                "program_id": exposure.program_id,
                "detector": exposure.detector,
                "observation": (exposure.observation,),
                "visit": (exposure.visit,),
                "ggsaa": (exposure.ggsaa,),
                "exposure": (exposure.exposure,),
            }
        return {"program_id": parse_program_id(filename) or ""}

    @classmethod
    def from_name(cls, location: str, stage: str) -> Self:
        """Build a metadata-incomplete NooBook from the file name alone.

        No file is opened: only name-derived identifiers are filled, while the
        header-derived fields (:attr:`pupil`, :attr:`filter`, :attr:`shape`,
        :attr:`footprint`) stay ``None`` until :meth:`from_file` or a producing
        step supplies them. Use it to index a directory cheaply.

        Parameters
        ----------
        location : str
            The file's local-path or ``host:path`` spec.
        stage : str
            Pipeline stage label.

        Returns
        -------
        NooBook
        """
        filename = _basename(location)
        book_id = f"{strip_fits_suffix(filename)}@{stage}"
        return cls(
            id=book_id,
            location=location,
            stage=stage,
            **cls._name_identifiers(filename),
        )

    @classmethod
    def from_file(
        cls,
        location: str,
        stage: str,
        *,
        parents: "Iterable[NooBook | str] | None" = None,
        store: ByteStore | None = None,
        raw: bytes | None = None,
    ) -> Self:
        """Build a fully-populated NooBook, reading the file's header once.

        Parameters
        ----------
        location : str
            The file's local-path or ``host:path`` spec.
        stage : str
            Pipeline stage label.
        parents : iterable of NooBook or str, optional
            The upstream products this one was produced from; their ids become
            :attr:`parent_ids`. This is how a reduction step stamps lineage at
            production time -- notably the many-to-one edges that key matching
            cannot recover. When ``None`` (default), :attr:`parent_ids` is empty.
        store : ByteStore, optional
            Shared byte store to read through (and bind for later data access).
            When ``None``, the file is read directly and the book is left
            unbound.
        raw : bytes, optional
            The file's bytes, already in hand. When given they are used for the
            header instead of reading ``location`` -- so a reduction step that has
            just written (and uploaded) a product can register it without fetching
            it back over SSH. ``location`` is still recorded as the book's path.

        Returns
        -------
        NooBook
            With :attr:`pupil`, :attr:`filter`, :attr:`shape` and
            :attr:`footprint` populated from the file.
        """
        filename = _basename(location)
        book_id = f"{strip_fits_suffix(filename)}@{stage}"
        acc = (
            BytesAccessor(raw)
            if raw is not None
            else _make_accessor(store, book_id, location)
        )
        fields, layout = _read_header_fields(acc)
        book = cls(
            id=book_id,
            location=location,
            stage=stage,
            parent_ids=_parent_ids_of(parents),
            **fields,
            **cls._name_identifiers(filename),
        )
        book._layout = layout
        if store is not None:
            book._bind_store(store)
        return book

    def probe(self, *, store: ByteStore | None = None) -> Self:
        """Return a copy with header-derived fields filled by reading the file.

        Identity and name-derived fields — including :attr:`parent_ids` — are
        preserved; only :attr:`pupil`, :attr:`filter`, :attr:`shape` and
        :attr:`footprint` are (re)read from the file. Use it to upgrade a thin
        book built by :meth:`from_name` into a fully-populated one.

        Parameters
        ----------
        store : ByteStore, optional
            Byte store to read through. Defaults to the book's bound store, or
            a direct read when neither is set.

        Returns
        -------
        NooBook
            A populated copy, bound to ``store`` (or the existing store).
        """
        active = store if store is not None else self._store
        acc = _make_accessor(active, self.id, self.location)
        fields, layout = _read_header_fields(acc)
        book = self.model_copy(update=fields)
        book._layout = layout
        if active is not None:
            book._bind_store(active)
        return book

    @classmethod
    def from_record(
        cls, record: dict[str, Any], *, store: ByteStore | None = None
    ) -> Self:
        """Rebuild a NooBook from a persisted manifest row.

        Parameters
        ----------
        record : dict
            A mapping produced by :meth:`to_record`.
        store : ByteStore, optional
            Shared byte store to bind for data access.

        Returns
        -------
        NooBook
        """
        book = cls.model_validate(record)
        if store is not None:
            book._bind_store(store)
        return book

    def to_record(self) -> dict[str, Any]:
        """Return the serializable record (manifest row) for this NooBook."""
        return self.model_dump()

    def _bind_store(self, store: ByteStore) -> Self:
        """Attach a shared byte store (called by NooBox); returns ``self``."""
        self._store = store
        return self

    def _stamp_parents(self, parent_ids: tuple[str, ...]) -> Self:
        """Set :attr:`parent_ids` in place (NooBox merge resolution); returns self."""
        self.parent_ids = parent_ids
        return self

    # -- derived views --------------------------------------------------------

    @property
    def level(self) -> int:
        """Coarse pipeline level: the leading digit of :attr:`stage`."""
        return int(self.stage[0])

    @property
    def is_probed(self) -> bool:
        """Whether the header-derived fields have been populated from the file."""
        return self.shape is not None

    # -- subpackage accessors -------------------------------------------------

    @property
    def viz(self) -> BookViz:
        """Plotting sugar bound to this book, e.g. ``book.viz.imshow()``."""
        return BookViz(self)

    @property
    def extract(self) -> BookExtract:
        """Extraction sugar bound to this book, e.g. ``book.extract.cutout(...)``."""
        return BookExtract(self)

    # -- live data access (read on demand, never pinned) ----------------------

    def read_bytes(self) -> bytes:
        """Return the product file's raw bytes, read on demand.

        Fetched through the bound :class:`~noobfriend.navigation._store.ByteStore`
        when the book belongs to a NooBox -- so the read is cached and shared, and
        a remote ``host:path`` location is pulled over SSH -- or read directly
        otherwise. Passed to e.g. ``astropy.io.fits.open(BytesIO(...))`` it yields
        the whole file, which is how a reduction script opens a (possibly remote)
        product as a jwst datamodel.

        Returns
        -------
        bytes
            The complete bytes of the FITS product.
        """
        if self._store is not None:
            return self._store.fetch(self.id, self.location)
        return fetch_bytes(self.location)

    def _accessor(self) -> ByteAccessor:
        """Return the accessor for on-demand reads (store-backed when bound)."""
        return _make_accessor(self._store, self.id, self.location)

    def _ensure_layout(self) -> FitsLayout:
        """Return the cached ``SCI`` byte layout, reading it once on first need."""
        if self._layout is None:
            self._layout = read_layout(self._accessor())
        return self._layout

    @property
    def meta(self) -> AttrView:
        """The ASDF ``meta`` tree as an :class:`AttrView` (read on access)."""
        return read_meta(self._accessor())

    @property
    def gwcs(self) -> WCS | None:
        """The raw GWCS object, or ``None`` if the file has no assigned WCS yet.

        Read fresh from the file on each access. Use this only where a genuine
        :class:`gwcs.WCS` is required (handing off to ``jwst`` / ``grizli``,
        ASDF serialization); everything that just evaluates coordinate
        transforms should use :attr:`wcs` instead.
        """
        try:
            return read_gwcs(self._accessor())
        except KeyError:
            return None

    @property
    def wcs(self) -> "NoobWCS | WCS | None":
        """The WCS as a transform provider, or ``None`` without an assigned WCS.

        The product's GWCS compiled onto the noobase evaluator
        (:class:`~noobfriend.core.wcs.NoobWCS`) -- compiled once on first
        access and cached, answering the
        :class:`~noobfriend.core.wcs.TransformWCS` protocol hundreds of times
        faster per call than the astropy model tree. When the transform is not
        yet supported by the compiler
        (:class:`~noobfriend.core.wcs.UnsupportedTransformError`, e.g. NIRSpec
        PRISM), each access falls back to the raw :attr:`gwcs` transparently.
        """
        if not self._noobwcs_compiled:
            from noobfriend.core.wcs import UnsupportedTransformError, from_gwcs

            gwcs_obj = self.gwcs
            if gwcs_obj is not None:
                try:
                    self._noobwcs = from_gwcs(gwcs_obj)
                except UnsupportedTransformError:
                    self._noobwcs = None
            self._noobwcs_compiled = True
        if self._noobwcs is not None:
            return self._noobwcs
        return self.gwcs

    @property
    def data(self) -> np.ndarray:
        """The ``SCI`` array, read fresh on each access (not cached)."""
        return read_data(self._accessor(), self._ensure_layout())

    @property
    def err(self) -> np.ndarray | None:
        """The ``ERR`` array, or ``None`` if the product has no ``ERR``."""
        try:
            return read_err(self._accessor(), self._ensure_layout())
        except KeyError:
            return None

    @property
    def dq(self) -> np.ndarray | None:
        """The ``DQ`` array, or ``None`` if the product has no ``DQ``."""
        try:
            return read_dq(self._accessor(), self._ensure_layout())
        except KeyError:
            return None

    @property
    def noise_kernel(self) -> NoiseAutocovariance | None:
        """The persisted noise autocovariance, or ``None`` if the product has none.

        A stage-3 coadd carries a ``NOISEKERN`` plane recording the mosaic's
        noise autocovariance ``C(d)``
        (:func:`noobfriend.reduction.mosaic.noise_kernel`); this reads it back as
        a :class:`~noobfriend.core.imgutils.NoiseAutocovariance`. Products
        without that plane (e.g. detector-stage frames) return ``None``. The
        reconstituted model has ``n_pairs`` and ``error_var`` unset, so it drives
        the per-pixel variance and the stationary
        :func:`~noobfriend.core.imgutils.correlated_sum_variance`; the hybrid
        (depth-aware) path additionally needs an ``error_var``.
        """
        try:
            return read_noise(self._accessor(), self._ensure_layout())
        except KeyError:
            return None
