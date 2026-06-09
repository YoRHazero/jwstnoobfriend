"""The :class:`ApertureSED` orchestrator for multi-band aperture photometry.

The pipeline this conductor runs:

1. grow a binary aperture per band on its own native grid;
2. rasterize the union of those apertures once, on the finest band's grid (the
   single hard threshold lives where pixels are smallest, so its boundary bias
   is minimal);
3. express that union back on every band's native grid as fractional coverage
   and measure there, summing native pixels weighted by coverage.

Science data is never resampled: only binary masks are reprojected (see
:mod:`noobfriend.extraction.photometry._coverage`). That keeps summed flux
correct under noobase's surface-brightness-conserving reprojection and keeps
per-pixel errors independent.
"""

import asyncio
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from noobfriend.extraction._wcs import pixel_scale_per_deg, world_detector_transforms
from noobfriend.extraction.photometry._aperture import ApertureMask, grow_aperture_mask
from noobfriend.extraction.photometry._band import Band, normalize_band
from noobfriend.extraction.photometry._coverage import reproject_coverage
from noobfriend.extraction.photometry._measure import measure_band
from noobfriend.extraction.photometry._result import ApertureSEDResult


class ApertureSED:
    """Build an SED by measuring every band through one union aperture.

    Parameters
    ----------
    bands : mapping
        ``{band_name: spec}``; each ``spec`` is a plain mapping with required
        ``"data"`` and ``"wcs"`` entries, plus optional ``"error"``,
        ``"wavelength"``, ``"wavelength_error"``, ``"flux_scale_mjy"``,
        ``"flux_unit"``, ``"label_map"``, and ``"label_allowed"`` (see
        :func:`noobfriend.extraction.photometry._band.normalize_band`).
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
    source_metadata : mapping, optional
        Provenance metadata stored on results. Mostly used by remote loaders.
    default_seed_world : tuple[float, float], optional
        Default source coordinate used when :meth:`measure` is called without an
        explicit seed. Set by :meth:`from_grizli_cutout`.
    """

    def __init__(
        self,
        bands: Mapping[str, Mapping[str, Any]],
        *,
        reference: str = "finest",
        union_threshold: float = 0.5,
        flag_bad_fraction: float = 0.10,
        coarse_step: tuple[int, int] | None = None,
        source_metadata: Mapping[str, Any] | None = None,
        default_seed_world: tuple[float, float] | None = None,
    ) -> None:
        self._bands = tuple(normalize_band(name, spec) for name, spec in bands.items())
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
        self.source_metadata = dict(source_metadata or {})
        self.default_seed_world = default_seed_world

    @classmethod
    async def from_grizli_cutout_async(
        cls,
        ra: float,
        dec: float,
        *,
        size: float = 5.0,
        filters: str | tuple[str, ...] | list[str] = "sed-default",
        reference: str = "finest",
        union_threshold: float = 0.5,
        flag_bad_fraction: float = 0.10,
        coarse_step: tuple[int, int] | None = None,
        cache: bool = False,
        cache_dir: str | Path | None = None,
        overwrite: bool = False,
        allow_missing: bool = True,
        timeout: float = 120.0,
    ) -> "ApertureSED":
        """Build an :class:`ApertureSED` from grizli-cutout.

        Thin shim over :func:`noobfriend.core.io.load_grizli_cutout`: it fetches
        and parses the cutout, then hands the resulting band specs to
        :class:`ApertureSED`. The default ``"sed-default"`` preset selects every
        covered JWST imaging band plus covered HST ``f435w`` / ``f606w``.
        """
        from noobfriend.core.io import load_grizli_cutout

        cutout = await load_grizli_cutout(
            ra,
            dec,
            size_arcsec=size,
            filters=filters,
            cache=cache,
            cache_dir=cache_dir,
            overwrite=overwrite,
            allow_missing=allow_missing,
            timeout=timeout,
        )
        return cls(
            cutout.bands,
            reference=reference,
            union_threshold=union_threshold,
            flag_bad_fraction=flag_bad_fraction,
            coarse_step=coarse_step,
            source_metadata=cutout.metadata,
            default_seed_world=(float(ra), float(dec)),
        )

    @classmethod
    def from_grizli_cutout(
        cls,
        ra: float,
        dec: float,
        *,
        size: float = 5.0,
        filters: str | tuple[str, ...] | list[str] = "sed-default",
        reference: str = "finest",
        union_threshold: float = 0.5,
        flag_bad_fraction: float = 0.10,
        coarse_step: tuple[int, int] | None = None,
        cache: bool = False,
        cache_dir: str | Path | None = None,
        overwrite: bool = False,
        allow_missing: bool = True,
        timeout: float = 120.0,
    ) -> "ApertureSED":
        """Build from grizli-cutout through the synchronous convenience API."""
        return _run_coroutine_blocking(
            cls.from_grizli_cutout_async(
                ra,
                dec,
                size=size,
                filters=filters,
                reference=reference,
                union_threshold=union_threshold,
                flag_bad_fraction=flag_bad_fraction,
                coarse_step=coarse_step,
                cache=cache,
                cache_dir=cache_dir,
                overwrite=overwrite,
                allow_missing=allow_missing,
                timeout=timeout,
            )
        )

    def measure(
        self,
        *,
        seed_world: tuple[float, float] | None = None,
        seed_xy_by_band: Mapping[str, tuple[float, float]] | None = None,
        grow_kwargs: Mapping[str, Any] | None = None,
    ) -> ApertureSEDResult:
        """Measure every band through the union of their grown apertures.

        Exactly one seed form is required, unless this object was built by
        :meth:`from_grizli_cutout`, in which case its requested sky position is
        used by default:

        - ``seed_world=(ra, dec)`` maps the source through each band's WCS;
        - ``seed_xy_by_band={band: (x, y)}`` supplies band-local pixel seeds.

        Parameters
        ----------
        seed_world : tuple[float, float], optional
            Source world coordinate in degrees.
        seed_xy_by_band : mapping, optional
            Per-band ``(x, y)`` source pixels.
        grow_kwargs : mapping, optional
            Extra keyword arguments forwarded to noobase aperture growth.

        Returns
        -------
        ApertureSEDResult

        Raises
        ------
        ValueError
            If seed inputs are missing, over-specified, incomplete, or the union
            aperture is empty.
        """
        seeds = self._resolve_seeds(seed_world, seed_xy_by_band)
        reference = self._resolve_reference(seeds)

        source_apertures: dict[str, ApertureMask] = {
            band.name: grow_aperture_mask(
                band.data,
                seed_xy=seeds[band.name],
                error=band.error,
                label_map=band.label_map,
                label_allowed=band.label_allowed,
                grow_kwargs=grow_kwargs,
            )
            for band in self._bands
        }

        union = self._build_union(reference, source_apertures)
        if not bool(union.any()):
            raise ValueError("Union aperture is empty after mask alignment.")

        band_coverage: dict[str, np.ndarray] = {}
        measurements = []
        for band in self._bands:
            if band.name == reference.name:
                coverage = union.astype(np.float64)
            else:
                coverage = reproject_coverage(
                    union,
                    source_wcs=reference.wcs,
                    target_wcs=band.wcs,
                    target_shape=band.shape,
                    coarse_step=self.coarse_step,
                )
            band_coverage[band.name] = coverage
            measurements.append(
                measure_band(band, coverage, flag_bad_fraction=self.flag_bad_fraction)
            )

        measured = tuple(measurements)
        if all(m.wavelength is not None for m in measured):
            measured = tuple(sorted(measured, key=lambda m: float(m.wavelength)))

        return ApertureSEDResult(
            measurements=measured,
            reference_band=reference.name,
            union_mask=union,
            band_coverage=band_coverage,
            source_apertures=source_apertures,
            source_metadata=self.source_metadata,
        )

    def _build_union(
        self, reference: Band, source_apertures: Mapping[str, ApertureMask]
    ) -> np.ndarray:
        """Rasterize the union of all apertures on the reference grid."""
        union = np.zeros(reference.shape, dtype=bool)
        for band in self._bands:
            mask = source_apertures[band.name].mask
            if band.name == reference.name:
                coverage = mask.astype(np.float64)
            else:
                coverage = reproject_coverage(
                    mask,
                    source_wcs=band.wcs,
                    target_wcs=reference.wcs,
                    target_shape=reference.shape,
                    coarse_step=self.coarse_step,
                )
            union |= coverage >= self.union_threshold
        return union

    def _resolve_seeds(
        self,
        seed_world: tuple[float, float] | None,
        seed_xy_by_band: Mapping[str, tuple[float, float]] | None,
    ) -> dict[str, tuple[float, float]]:
        """Resolve the caller's seed input into per-band pixel seeds."""
        if seed_world is None and seed_xy_by_band is None:
            seed_world = self.default_seed_world
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

    def _resolve_reference(self, seeds: Mapping[str, tuple[float, float]]) -> Band:
        """Return the band whose grid rasterizes the union."""
        if self.reference != "finest":
            return self._by_name[self.reference]

        def score(band: Band) -> float:
            _, detector_to_world = world_detector_transforms(band.wcs)
            x, y = seeds[band.name]
            scale_x, scale_y = pixel_scale_per_deg(
                detector_to_world, int(np.round(x)), int(np.round(y))
            )
            positive = [v for v in (scale_x, scale_y) if v > 0]
            return float(np.mean(positive)) if positive else 0.0

        return max(self._bands, key=score)


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
