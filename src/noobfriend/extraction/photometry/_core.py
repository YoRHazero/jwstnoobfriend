"""The :class:`ApertureSED` orchestrator for multi-band aperture photometry.

The pipeline this conductor runs:

1. grow a binary aperture per band on its own native grid (:meth:`ApertureSED.draft`);
2. rasterize the union of those apertures once, on the finest band's grid (the
   single hard threshold lives where pixels are smallest, so its boundary bias
   is minimal);
3. express that union back on every band's native grid as fractional coverage
   and measure there, summing native pixels weighted by coverage
   (:meth:`ApertureSEDDraft.measure`).

Steps 2-3 live on :class:`~noobfriend.extraction.photometry._draft.ApertureSEDDraft`,
the inspectable intermediate that :meth:`ApertureSED.draft` returns, so the grown
apertures and candidate union can be previewed and the union membership chosen
before any flux is measured.

Science data is never resampled: only binary masks are reprojected (see
:mod:`noobfriend.extraction.photometry._coverage`). That keeps summed flux
correct under noobase's surface-brightness-conserving reprojection and keeps
per-pixel errors independent.
"""

import asyncio
import logging
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from noobfriend.core.imgutils import segment
from noobfriend.extraction._wcs import pixel_scale_per_deg, world_detector_transforms
from noobfriend.extraction.photometry._aperture import ApertureMask, grow_aperture_mask
from noobfriend.extraction.photometry._band import Band, BandSpec, normalize_band
from noobfriend.extraction.photometry._draft import ApertureSEDDraft
from noobfriend.extraction.photometry._floor import resolve_floor

logger = logging.getLogger(__name__)


def _auto_label_map(
    data: np.ndarray,
    seed_xy: tuple[float, float],
    segment_kwargs: Mapping[str, Any] | None,
) -> np.ndarray | None:
    """Segment ``data`` and keep the map only if the seed lands on a source.

    Runs :func:`noobfriend.core.imgutils.segment` and returns the label map when
    the seed pixel sits on a detected segment (label ``> 0``); otherwise returns
    ``None`` so the band grows unconstrained -- a non-detection at the seed has
    no segment worth confining the aperture to.
    """
    labels = segment(np.asarray(data, dtype=float), **dict(segment_kwargs or {}))
    seed_x, seed_y = int(round(seed_xy[0])), int(round(seed_xy[1]))
    n_rows, n_cols = labels.shape
    if not (0 <= seed_y < n_rows and 0 <= seed_x < n_cols):
        return None
    return labels if int(labels[seed_y, seed_x]) > 0 else None


class ApertureSED:
    """Build an SED by measuring every band through one union aperture.

    An ``ApertureSED`` is the source identity plus its bands and the union
    geometry settings; it carries no measured photometry. Call :meth:`draft` to
    grow the per-band apertures (the inspectable
    :class:`~noobfriend.extraction.photometry._draft.ApertureSEDDraft`), then
    :meth:`ApertureSEDDraft.measure` to obtain the
    :class:`~noobfriend.extraction.photometry._result.ApertureSEDResult`.

    Parameters
    ----------
    source_id : str
        Identifier for the source, carried onto results (and used as the default
        SED plot title).
    ra, dec : float
        Source world coordinate in degrees (ICRS), used as the default growth
        seed.
    bands : mapping, optional
        ``{band_name: spec}``; each ``spec`` is a
        :class:`~noobfriend.extraction.photometry._band.BandSpec` dict with a
        required ``"data"`` array plus optional ``"wcs"``, ``"error"``,
        ``"wavelength"``, ``"wavelength_error"``, ``"flux_scale_mjy"``,
        ``"flux_unit"``, ``"label_map"``, ``"label_allowed"``, and
        ``"allow_background"``. A ``BandSpec`` is a :class:`~typing.TypedDict`, so
        editors check the keys of a dict literal. At least one band is required;
        use :meth:`build` to gather bands from a remote database.
    cutout_size_arcsec : float, optional
        Angular size (each axis) of the cutouts. Required only when a band omits
        ``"wcs"``: a tangent-plane WCS is then synthesised, centred on the source
        with a pixel scale from this size and the band's shape (see
        :func:`noobfriend.extraction._wcs.tangent_plane_wcs`). Omitting both the
        ``"wcs"`` and this size raises.
    reference : {"finest"} or str, default "finest"
        Grid used to rasterize the union aperture. ``"finest"`` selects the band
        with the smallest local pixel scale near the source; any other value
        must be a band name.
    union_threshold : float, default 0.5
        A reference-grid pixel joins the union when its reprojected aperture
        coverage fraction is at least this value.
    flag_bad_fraction : float, default 0.10
        Bands whose aperture is at least this fraction (by area) non-finite are
        flagged.
    coarse_step : tuple[int, int], optional
        Passed to the coverage reprojection to speed up the gwcs inverse.
    """

    def __init__(
        self,
        source_id: str,
        ra: float,
        dec: float,
        *,
        bands: Mapping[str, BandSpec] | None = None,
        cutout_size_arcsec: float | None = None,
        reference: str = "finest",
        union_threshold: float = 0.5,
        flag_bad_fraction: float = 0.10,
        coarse_step: tuple[int, int] | None = None,
    ) -> None:
        self.source_id = str(source_id)
        self.ra = float(ra)
        self.dec = float(dec)
        self.cutout_size_arcsec = (
            None if cutout_size_arcsec is None else float(cutout_size_arcsec)
        )
        self._bands = tuple(
            normalize_band(
                name,
                spec,
                ra=self.ra,
                dec=self.dec,
                cutout_size_arcsec=self.cutout_size_arcsec,
            )
            for name, spec in (bands or {}).items()
        )
        if not self._bands:
            raise ValueError("bands must contain at least one band.")
        self._by_name = {band.name: band for band in self._bands}
        if len(self._by_name) != len(self._bands):
            raise ValueError("band names must be unique.")
        if reference != "finest" and reference not in self._by_name:
            raise ValueError(
                f"reference must be 'finest' or one of {sorted(self._by_name)}; "
                f"got {reference!r}."
            )
        if not (0.0 < union_threshold <= 1.0):
            raise ValueError("union_threshold must satisfy 0 < threshold <= 1.")
        if not (0.0 <= flag_bad_fraction <= 1.0):
            raise ValueError("flag_bad_fraction must satisfy 0 <= value <= 1.")
        self.reference = reference
        self.union_threshold = float(union_threshold)
        self.flag_bad_fraction = float(flag_bad_fraction)
        self.coarse_step = coarse_step

    @property
    def band_names(self) -> tuple[str, ...]:
        """Names of the assembled bands, in input order."""
        return tuple(band.name for band in self._bands)

    @classmethod
    def build(
        cls,
        source_id: str,
        ra: float,
        dec: float,
        *,
        bands: Mapping[str, BandSpec] | None = None,
        cutout_size_arcsec: float | None = None,
        extra_database: str | None = None,
        extra_filters: str | Sequence[str] = "sed-default",
        reference: str = "finest",
        union_threshold: float = 0.5,
        flag_bad_fraction: float = 0.10,
        coarse_step: tuple[int, int] | None = None,
        size: float | None = None,
        cache: bool = False,
        cache_dir: str | Path | None = None,
        overwrite: bool = False,
        allow_missing: bool = True,
        timeout: float = 120.0,
    ) -> "ApertureSED":
        """Assemble an :class:`ApertureSED`, optionally pulling bands from a database.

        Synchronous and notebook-friendly: the asynchronous data acquisition runs
        internally, so callers never need ``await``. With ``extra_database`` set,
        bands covering the source are fetched and merged with any ``bands`` passed
        directly; on a name conflict the explicitly-provided band wins and the
        fetched one is dropped (and logged).

        Parameters
        ----------
        source_id : str
            Identifier for the source.
        ra, dec : float
            Source world coordinate in degrees (ICRS).
        bands : mapping, optional
            Band specs to use directly, in addition to (and taking precedence
            over) anything ``extra_database`` returns.
        cutout_size_arcsec : float, optional
            Cutout angular size, forwarded to the constructor (to synthesise a
            WCS for any WCS-less ``bands``) and used as the default grizli fetch
            ``size``.
        extra_database : {"grizli"} or None, default None
            Remote database to pull additional bands from. ``None`` uses only
            ``bands``.
        extra_filters : str or sequence of str, default "sed-default"
            Filters to request from ``extra_database`` (an explicit list, a
            comma-separated string, or a preset such as ``"sed-default"`` /
            ``"jwst"`` / ``"all"``; see
            :func:`noobfriend.core.io.load_grizli_cutout`).
        reference, union_threshold, flag_bad_fraction, coarse_step
            Forwarded to the constructor.
        size : float, optional
            Cutout size for the grizli fetch. Defaults to ``cutout_size_arcsec``
            (or ``5.0`` when that too is unset); pass it to override the fetch
            size independently of the WCS-synthesis size.
        cache, cache_dir, overwrite, allow_missing, timeout
            Forwarded to :func:`noobfriend.core.io.load_grizli_cutout` when
            ``extra_database`` is set.

        Returns
        -------
        ApertureSED

        Raises
        ------
        ValueError
            If ``extra_database`` is unsupported, or no band could be assembled.
        """
        if size is not None:
            grizli_size = float(size)
        elif cutout_size_arcsec is not None:
            grizli_size = float(cutout_size_arcsec)
        else:
            grizli_size = 5.0
        return _run_coroutine_blocking(
            cls._build_async(
                source_id,
                ra,
                dec,
                bands=bands,
                cutout_size_arcsec=cutout_size_arcsec,
                extra_database=extra_database,
                extra_filters=extra_filters,
                reference=reference,
                union_threshold=union_threshold,
                flag_bad_fraction=flag_bad_fraction,
                coarse_step=coarse_step,
                size=grizli_size,
                cache=cache,
                cache_dir=cache_dir,
                overwrite=overwrite,
                allow_missing=allow_missing,
                timeout=timeout,
            )
        )

    @classmethod
    async def _build_async(
        cls,
        source_id: str,
        ra: float,
        dec: float,
        *,
        bands: Mapping[str, BandSpec] | None = None,
        cutout_size_arcsec: float | None = None,
        extra_database: str | None = None,
        extra_filters: str | Sequence[str] = "sed-default",
        reference: str = "finest",
        union_threshold: float = 0.5,
        flag_bad_fraction: float = 0.10,
        coarse_step: tuple[int, int] | None = None,
        size: float = 5.0,
        cache: bool = False,
        cache_dir: str | Path | None = None,
        overwrite: bool = False,
        allow_missing: bool = True,
        timeout: float = 120.0,
    ) -> "ApertureSED":
        """Async core of :meth:`build`: fetch + merge bands, then construct."""
        merged: dict[str, Any] = dict(bands or {})
        if extra_database is not None:
            if extra_database != "grizli":
                raise ValueError(
                    f"Unsupported extra_database {extra_database!r}; "
                    "expected 'grizli' or None."
                )
            from noobfriend.core.io import load_grizli_cutout

            cutout = await load_grizli_cutout(
                ra,
                dec,
                size_arcsec=size,
                filters=tuple(extra_filters)
                if not isinstance(extra_filters, str)
                else extra_filters,
                cache=cache,
                cache_dir=cache_dir,
                overwrite=overwrite,
                allow_missing=allow_missing,
                timeout=timeout,
            )
            conflicts = [name for name in cutout.bands if name in merged]
            for name, spec in cutout.bands.items():
                if name not in merged:
                    merged[name] = spec
            if conflicts:
                logger.warning(
                    "extra_database %r bands skipped; names already provided: %s",
                    extra_database,
                    conflicts,
                )
        return cls(
            source_id,
            ra,
            dec,
            bands=merged,
            cutout_size_arcsec=cutout_size_arcsec,
            reference=reference,
            union_threshold=union_threshold,
            flag_bad_fraction=flag_bad_fraction,
            coarse_step=coarse_step,
        )

    def draft(
        self,
        *,
        auto_segment: bool = False,
        segment_kwargs: Mapping[str, Any] | None = None,
        label_map_by_band: Mapping[str, np.ndarray] | None = None,
        label_allowed_by_band: Mapping[str, Sequence[int]] | None = None,
        grow_kwargs: Mapping[str, Any] | None = None,
        seed_world: tuple[float, float] | None = None,
        seed_xy_by_band: Mapping[str, tuple[float, float]] | None = None,
    ) -> ApertureSEDDraft:
        """Grow every band's aperture and return the inspectable draft.

        The seed defaults to this object's ``(ra, dec)``; pass ``seed_world`` or
        ``seed_xy_by_band`` to override (exactly one). Each band's growth is
        confined by a segmentation label map resolved with the precedence
        **explicit override > band spec > auto-segment**:

        - ``label_map_by_band[name]`` (and ``label_allowed_by_band[name]``), if
          given, always wins;
        - otherwise the band spec's own ``label_map`` / ``label_allowed``;
        - otherwise, when ``auto_segment`` is ``True`` and the band has no map, a
          map from :func:`noobfriend.core.imgutils.segment` -- but only if the
          source is detected at the seed (else the band grows unconstrained).

        Parameters
        ----------
        auto_segment : bool, default False
            Segment each unconstrained band and confine its growth to the seed's
            own segment (blocking neighbours) when the seed is on a detection.
        segment_kwargs : mapping, optional
            Keyword arguments forwarded to
            :func:`noobfriend.core.imgutils.segment` (``nsigma``, ``min_size``,
            ``deblend``, ``connectivity``) when ``auto_segment`` is in effect.
        label_map_by_band : mapping, optional
            Per-band segmentation maps that override both the spec and
            ``auto_segment`` for the named bands.
        label_allowed_by_band : mapping, optional
            Per-band allowed-label sequences overriding the spec for the named
            bands.
        grow_kwargs : mapping, optional
            Extra keyword arguments forwarded to noobase aperture growth. A
            ``"min_pixels_before_stop_check"`` here is the *base* floor at the
            finest band; each band's actual floor is that base scaled down by
            relative sky area and, where a label map is present, capped at the
            seed's own segment size. Omit it for the noobase default of 30.
        seed_world : tuple[float, float], optional
            Source world coordinate in degrees; defaults to ``(ra, dec)``.
        seed_xy_by_band : mapping, optional
            Per-band ``(x, y)`` source pixels (mutually exclusive with
            ``seed_world``).

        Returns
        -------
        ApertureSEDDraft

        Raises
        ------
        ValueError
            If both seed forms are given, ``seed_xy_by_band`` keys do not match
            the bands, or a seed is invalid for its band.
        """
        seeds = self._resolve_seeds(seed_world, seed_xy_by_band)
        scales = self._band_scales(seeds)
        reference = self._resolve_reference(scales)

        user_grow_kwargs = dict(grow_kwargs or {})
        base_floor = int(user_grow_kwargs.pop("min_pixels_before_stop_check", 30))
        finest_scale = max(scales.values(), default=0.0)

        lm_override = label_map_by_band or {}
        la_override = label_allowed_by_band or {}

        source_apertures: dict[str, ApertureMask] = {}
        label_maps: dict[str, np.ndarray | None] = {}
        label_allowed: dict[str, tuple[int, ...] | None] = {}
        for band in self._bands:
            label_map = self._resolve_label_map(
                band, seeds[band.name], lm_override, auto_segment, segment_kwargs
            )
            allowed = (
                tuple(int(v) for v in la_override[band.name])
                if band.name in la_override
                else band.label_allowed
            )
            label_maps[band.name] = label_map
            label_allowed[band.name] = allowed
            floor = resolve_floor(
                base_floor,
                scales[band.name],
                finest_scale,
                label_map,
                band.data,
                seeds[band.name],
            )
            source_apertures[band.name] = grow_aperture_mask(
                band.data,
                seed_xy=seeds[band.name],
                error=band.error,
                label_map=label_map,
                label_allowed=allowed,
                allow_background=band.allow_background,
                grow_kwargs={
                    **user_grow_kwargs,
                    "min_pixels_before_stop_check": floor,
                },
            )

        return ApertureSEDDraft(
            source_id=self.source_id,
            ra=self.ra,
            dec=self.dec,
            bands=self._bands,
            reference_band=reference.name,
            source_apertures=source_apertures,
            seeds=seeds,
            scales=scales,
            base_floor=base_floor,
            label_maps=label_maps,
            label_allowed=label_allowed,
            union_threshold=self.union_threshold,
            flag_bad_fraction=self.flag_bad_fraction,
            coarse_step=self.coarse_step,
        )

    def _resolve_label_map(
        self,
        band: Band,
        seed_xy: tuple[float, float],
        lm_override: Mapping[str, np.ndarray],
        auto_segment: bool,
        segment_kwargs: Mapping[str, Any] | None,
    ) -> np.ndarray | None:
        """Pick a band's growth label map: override > spec > auto-segment > None."""
        if band.name in lm_override:
            return np.asarray(lm_override[band.name])
        if band.label_map is not None:
            return band.label_map
        if auto_segment:
            return _auto_label_map(band.data, seed_xy, segment_kwargs)
        return None

    def _resolve_seeds(
        self,
        seed_world: tuple[float, float] | None,
        seed_xy_by_band: Mapping[str, tuple[float, float]] | None,
    ) -> dict[str, tuple[float, float]]:
        """Resolve the caller's seed input into per-band pixel seeds."""
        if seed_world is None and seed_xy_by_band is None:
            seed_world = (self.ra, self.dec)
        if (seed_world is None) == (seed_xy_by_band is None):
            raise ValueError("Provide exactly one of seed_world or seed_xy_by_band.")
        if seed_world is not None:
            ra, dec = seed_world
            seeds: dict[str, tuple[float, float]] = {}
            for band in self._bands:
                world_to_detector, _ = world_detector_transforms(band.wcs)
                x, y = world_to_detector(ra, dec)
                seeds[band.name] = (float(x), float(y))
            return seeds

        assert seed_xy_by_band is not None
        missing = set(self._by_name) - set(seed_xy_by_band)
        extra = set(seed_xy_by_band) - set(self._by_name)
        if missing or extra:
            raise ValueError(
                f"seed_xy_by_band keys must match band names; missing={missing}, "
                f"extra={extra}."
            )
        return {
            name: (float(seed_xy[0]), float(seed_xy[1]))
            for name, seed_xy in seed_xy_by_band.items()
        }

    def _band_scales(
        self, seeds: Mapping[str, tuple[float, float]]
    ) -> dict[str, float]:
        """Local pixel scale (pixels-per-degree) at the seed for every band.

        A larger value is a finer grid. Reused for both the reference choice and
        the area-scaled aperture floor.
        """
        scales: dict[str, float] = {}
        for band in self._bands:
            _, detector_to_world = world_detector_transforms(band.wcs)
            x, y = seeds[band.name]
            scale_x, scale_y = pixel_scale_per_deg(
                detector_to_world, int(np.round(x)), int(np.round(y))
            )
            positive = [v for v in (scale_x, scale_y) if v > 0]
            scales[band.name] = float(np.mean(positive)) if positive else 0.0
        return scales

    def _resolve_reference(self, scales: Mapping[str, float]) -> Band:
        """Return the band whose grid rasterizes the union (finest, by default)."""
        if self.reference != "finest":
            return self._by_name[self.reference]
        return max(self._bands, key=lambda band: scales[band.name])


def _run_coroutine_blocking(coro: Any) -> Any:
    """Run ``coro`` from sync code, including inside an existing event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001 - re-raise in caller thread.
            result["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result["value"]
