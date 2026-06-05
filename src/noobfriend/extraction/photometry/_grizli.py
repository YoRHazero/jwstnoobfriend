"""Load multi-band cutouts from the public grizli-cutout service.

The default path is intentionally in-memory: fetch the FITS response as bytes,
open it through :class:`io.BytesIO`, and only write to disk when the caller
explicitly asks for caching.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS as AstropyWCS

from noobfriend.core.io.network import HTTPSession

GRIZLI_BASE_URL = "https://grizli-cutout.herokuapp.com"
_C_AA_PER_S = 2.99792458e18

HST_BLUE_BASES = ("f435w", "f606w")
NIRCAM_BROAD_BASES = {
    "f070w",
    "f090w",
    "f115w",
    "f150w",
    "f200w",
    "f277w",
    "f356w",
    "f444w",
}
NIRCAM_IMAGING_BASES = NIRCAM_BROAD_BASES | {
    "f140m",
    "f150w2",
    "f162m",
    "f164n",
    "f182m",
    "f187n",
    "f210m",
    "f212n",
    "f250m",
    "f300m",
    "f322w2",
    "f323n",
    "f335m",
    "f360m",
    "f405n",
    "f410m",
    "f430m",
    "f460m",
    "f466n",
    "f470n",
    "f480m",
}
MIRI_IMAGING_BASES = {
    "f560w",
    "f770w",
    "f1000w",
    "f1130w",
    "f1280w",
    "f1500w",
    "f1800w",
    "f2100w",
    "f2550w",
}
JWST_IMAGING_BASES = NIRCAM_IMAGING_BASES | MIRI_IMAGING_BASES
ACS_IMAGING_BASES = {
    "f435w",
    "f475w",
    "f555w",
    "f606w",
    "f625w",
    "f775w",
    "f814w",
}

STATIC_FILTER_PRESETS: dict[str, tuple[str, ...]] = {
    "hst-blue": HST_BLUE_BASES,
    "jwst-nircam-broad": tuple(sorted(NIRCAM_BROAD_BASES)),
    "jwst-nircam": tuple(sorted(NIRCAM_IMAGING_BASES)),
    "jwst-miri": tuple(sorted(MIRI_IMAGING_BASES)),
}
DYNAMIC_FILTER_PRESETS = {"sed-default", "jwst", "jwst-imaging", "all"}
FILTER_PRESETS = set(STATIC_FILTER_PRESETS) | DYNAMIC_FILTER_PRESETS

FILTER_WAVELENGTH_MICRON: dict[str, float] = {
    "f435w": 0.435,
    "f475w": 0.475,
    "f555w": 0.555,
    "f606w": 0.606,
    "f625w": 0.625,
    "f775w": 0.775,
    "f814w": 0.814,
    "f105w": 1.05,
    "f110w": 1.10,
    "f125w": 1.25,
    "f140w": 1.40,
    "f160w": 1.60,
    "f070w": 0.70,
    "f090w": 0.90,
    "f115w": 1.15,
    "f140m": 1.40,
    "f150w": 1.50,
    "f150w2": 1.50,
    "f162m": 1.62,
    "f164n": 1.64,
    "f182m": 1.82,
    "f187n": 1.87,
    "f200w": 2.00,
    "f210m": 2.10,
    "f212n": 2.12,
    "f250m": 2.50,
    "f277w": 2.77,
    "f300m": 3.00,
    "f322w2": 3.22,
    "f323n": 3.23,
    "f335m": 3.35,
    "f356w": 3.56,
    "f360m": 3.60,
    "f405n": 4.05,
    "f410m": 4.10,
    "f430m": 4.30,
    "f444w": 4.44,
    "f460m": 4.60,
    "f466n": 4.66,
    "f470n": 4.70,
    "f480m": 4.80,
    "f560w": 5.60,
    "f770w": 7.70,
    "f1000w": 10.0,
    "f1130w": 11.3,
    "f1280w": 12.8,
    "f1500w": 15.0,
    "f1800w": 18.0,
    "f2100w": 21.0,
    "f2550w": 25.5,
}


@dataclass(frozen=True)
class GrizliCutout:
    """Parsed grizli cutout data ready for :class:`ApertureSED`."""

    bands: dict[str, dict[str, Any]]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _FitsWCSAdapter:
    """Adapter exposing astropy FITS WCS through noobfriend's transform protocol."""

    wcs: AstropyWCS
    available_frames: tuple[str, str] = ("detector", "world")

    def get_transform(self, from_frame: str, to_frame: str):
        """Return detector/world transforms using astropy's 0-based API."""
        if (from_frame, to_frame) == ("world", "detector"):
            return self.wcs.world_to_pixel_values
        if (from_frame, to_frame) == ("detector", "world"):
            return self.wcs.pixel_to_world_values
        raise ValueError(f"Unsupported transform: {from_frame!r} -> {to_frame!r}.")


def normalize_grizli_filters(
    filters: str | tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    """Normalize an explicit grizli filter list.

    Parameters
    ----------
    filters : str or sequence of str
        Comma-separated string or sequence of filter names. Preset names are
        expanded only for static presets; dynamic presets are resolved from the
        overlap endpoint by :func:`load_grizli_cutout`.

    Returns
    -------
    tuple[str, ...]
        Lower-case filter names, preserving first occurrence order.
    """
    if isinstance(filters, str):
        key = filters.strip().lower()
        if key in STATIC_FILTER_PRESETS:
            items = list(STATIC_FILTER_PRESETS[key])
        else:
            items = [item.strip() for item in filters.split(",")]
    else:
        items = [str(item).strip() for item in filters]

    seen: set[str] = set()
    normalized: list[str] = []
    for item in items:
        lowered = item.lower()
        if lowered and lowered not in seen:
            seen.add(lowered)
            normalized.append(lowered)
    if not normalized:
        raise ValueError("At least one grizli filter must be provided.")
    return tuple(normalized)


def build_grizli_thumb_url(
    ra: float,
    dec: float,
    size_arcsec: float,
    filters: tuple[str, ...] | list[str],
    *,
    output: str = "fits_weight",
) -> str:
    """Build a grizli-cutout ``/thumb`` URL."""
    params = {
        "ra": float(ra),
        "dec": float(dec),
        "size": float(size_arcsec),
        "output": output,
        "filters": ",".join(normalize_grizli_filters(list(filters))),
    }
    return f"{GRIZLI_BASE_URL}/thumb?{urlencode(params)}"


def build_grizli_overlap_url(
    ra: float,
    dec: float,
    size_arcsec: float,
    *,
    mode: str = "count",
    filters: tuple[str, ...] | list[str] | None = None,
) -> str:
    """Build a grizli-cutout ``/overlap`` URL."""
    params: dict[str, Any] = {
        "ra": float(ra),
        "dec": float(dec),
        "size": float(size_arcsec),
        "mode": mode,
    }
    if filters is not None:
        params["filters"] = ",".join(normalize_grizli_filters(list(filters)))
    return f"{GRIZLI_BASE_URL}/overlap?{urlencode(params)}"


def cutout_cache_filename(
    ra: float,
    dec: float,
    size_arcsec: float,
    filters: tuple[str, ...] | list[str],
    *,
    output: str = "fits_weight",
) -> str:
    """Return a stable cache filename for a grizli FITS cutout."""
    filter_token = "-".join(sorted(normalize_grizli_filters(list(filters))))
    return (
        f"grizli_ra{_float_token(float(ra))}_dec{_float_token(float(dec))}"
        f"_size{_float_token(float(size_arcsec), digits=2)}"
        f"_{filter_token}_{output}.fits"
    )


async def query_grizli_overlap(
    ra: float,
    dec: float,
    *,
    size_arcsec: float,
    filters: tuple[str, ...] | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Query grizli-cutout overlap metadata."""
    url = build_grizli_overlap_url(ra, dec, size_arcsec, filters=filters)
    payload = await HTTPSession(timeout=timeout).fetch_content(url)
    return json.loads(payload.decode("utf-8"))


def overlap_filters(overlap: dict[str, Any]) -> list[str]:
    """Return filter names from a grizli overlap response."""
    filters = overlap.get("filters")
    if isinstance(filters, list):
        return [str(item).lower() for item in filters]

    metadata_keys = {"ra", "dec", "size", "filters", "mode"}
    return [
        str(key).lower()
        for key, value in overlap.items()
        if key not in metadata_keys and isinstance(value, list)
    ]


def overlap_count(overlap: dict[str, Any], filter_name: str) -> int | None:
    """Return exposure count for one filter, when present."""
    value = overlap.get(filter_name) or overlap.get(filter_name.lower())
    if isinstance(value, list) and len(value) >= 2:
        try:
            return int(value[1])
        except (TypeError, ValueError):
            return None
    return None


def select_overlap_filters(
    overlap: dict[str, Any],
    *,
    filter_set: str = "sed-default",
    min_count: int = 1,
) -> tuple[str, ...]:
    """Select science filters from overlap metadata.

    ``"sed-default"`` is intentionally greedy for JWST imaging filters and
    conservative for HST: all covered JWST imaging bands plus covered
    ``f435w`` / ``f606w`` only.
    """
    if filter_set not in FILTER_PRESETS:
        raise ValueError(
            f"Unsupported grizli filter preset {filter_set!r}; "
            f"expected one of {sorted(FILTER_PRESETS)}."
        )
    if filter_set in STATIC_FILTER_PRESETS:
        allowed = set(STATIC_FILTER_PRESETS[filter_set])
        return tuple(
            f
            for f in overlap_filters(overlap)
            if _filter_base(f) in allowed
            and (
                overlap_count(overlap, f) is None
                or overlap_count(overlap, f) >= min_count
            )
        )

    selected: list[str] = []
    for filter_name in overlap_filters(overlap):
        count = overlap_count(overlap, filter_name)
        if count is not None and count < min_count:
            continue
        if _is_grism_filter(filter_name):
            continue

        base = _filter_base(filter_name)
        is_jwst = base in JWST_IMAGING_BASES
        is_hst_blue = base in HST_BLUE_BASES

        if filter_set == "all":
            selected.append(filter_name)
        elif filter_set in {"jwst", "jwst-imaging"} and is_jwst:
            selected.append(filter_name)
        elif filter_set == "sed-default" and (is_jwst or is_hst_blue):
            selected.append(filter_name)
    return tuple(selected)


async def load_grizli_cutout(
    ra: float,
    dec: float,
    *,
    size_arcsec: float = 5.0,
    filters: str | tuple[str, ...] | list[str] = "sed-default",
    cache: bool = False,
    cache_dir: str | Path | None = None,
    overwrite: bool = False,
    allow_missing: bool = True,
    timeout: float = 120.0,
) -> GrizliCutout:
    """Fetch and parse a grizli-cutout FITS response for aperture SED use."""
    requested_filters, selected_filters, overlap, overlap_url = await _resolve_filters(
        ra,
        dec,
        size_arcsec=size_arcsec,
        filters=filters,
        timeout=timeout,
    )
    if not selected_filters:
        raise ValueError("No grizli filters selected for this position.")

    thumb_url = build_grizli_thumb_url(ra, dec, size_arcsec, selected_filters)
    payload, cache_path = await _load_thumb_payload(
        thumb_url,
        ra=ra,
        dec=dec,
        size_arcsec=size_arcsec,
        filters=selected_filters,
        cache=cache,
        cache_dir=cache_dir,
        overwrite=overwrite,
        timeout=timeout,
    )
    bands = read_grizli_cutout_bands(payload)
    available = tuple(bands)
    missing = tuple(f for f in selected_filters if _filter_base(f) not in bands)
    if missing and not allow_missing:
        raise ValueError(
            "grizli-cutout did not return all requested filters; "
            f"missing={missing}, available={available}."
        )

    metadata = {
        "source": "grizli-cutout",
        "ra": float(ra),
        "dec": float(dec),
        "size_arcsec": float(size_arcsec),
        "requested_filters": requested_filters,
        "selected_filters": selected_filters,
        "available_filters": available,
        "missing_filters": missing,
        "overlap": overlap,
        "overlap_url": overlap_url,
        "thumb_url": thumb_url,
        "cache_path": None if cache_path is None else str(cache_path),
    }
    return GrizliCutout(bands=bands, metadata=metadata)


def read_grizli_cutout_bands(payload: bytes | str | Path) -> dict[str, dict[str, Any]]:
    """Read grizli ``fits_weight`` HDUs into ApertureSED band specs."""
    source = BytesIO(payload) if isinstance(payload, bytes) else Path(payload)
    bands: dict[str, dict[str, Any]] = {}
    with fits.open(source, memmap=False) as hdul:
        index = 0
        while index < len(hdul):
            hdu = hdul[index]
            if hdu.data is None:
                index += 1
                continue
            filter_name = _hdu_filter_name(hdu)
            if filter_name is None:
                index += 1
                continue

            next_hdu = hdul[index + 1] if index + 1 < len(hdul) else None
            weight = None
            if next_hdu is not None and _same_filter_data_pair(hdu, next_hdu):
                weight = np.asarray(next_hdu.data, dtype=float)
                index += 2
            else:
                index += 1

            data = np.asarray(hdu.data, dtype=float)
            if weight is None:
                error = None
            else:
                valid_weight = np.isfinite(weight) & (weight > 0)
                data = np.where(valid_weight, data, np.nan)
                error = np.full(weight.shape, np.nan, dtype=float)
                error[valid_weight] = 1.0 / np.sqrt(weight[valid_weight])

            base = _filter_base(filter_name)
            scale_mjy, scale_source = _mjy_scale_from_header(hdu.header)
            bands[base] = {
                "data": data,
                "error": error,
                "wcs": _FitsWCSAdapter(AstropyWCS(hdu.header)),
                "wavelength": FILTER_WAVELENGTH_MICRON.get(base),
                "flux_scale_mjy": scale_mjy,
                "flux_unit": scale_source,
            }
    if not bands:
        raise ValueError("No image HDUs could be parsed from the grizli cutout.")
    return bands


async def _resolve_filters(
    ra: float,
    dec: float,
    *,
    size_arcsec: float,
    filters: str | tuple[str, ...] | list[str],
    timeout: float,
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, Any] | None, str | None]:
    """Resolve explicit or dynamic grizli filters."""
    if isinstance(filters, str) and filters.strip().lower() in DYNAMIC_FILTER_PRESETS:
        preset = filters.strip().lower()
        overlap_url = build_grizli_overlap_url(ra, dec, size_arcsec)
        overlap = await query_grizli_overlap(
            ra,
            dec,
            size_arcsec=size_arcsec,
            timeout=timeout,
        )
        selected = select_overlap_filters(overlap, filter_set=preset)
        return ((preset,), selected, overlap, overlap_url)

    selected = normalize_grizli_filters(filters)
    return (selected, selected, None, None)


async def _load_thumb_payload(
    url: str,
    *,
    ra: float,
    dec: float,
    size_arcsec: float,
    filters: tuple[str, ...],
    cache: bool,
    cache_dir: str | Path | None,
    overwrite: bool,
    timeout: float,
) -> tuple[bytes | Path, Path | None]:
    """Fetch a grizli FITS payload, optionally through an explicit cache."""
    if cache_dir is None and not cache:
        return await HTTPSession(timeout=timeout).fetch_content(url), None

    directory = (
        Path(cache_dir)
        if cache_dir is not None
        else Path.cwd() / ".noobfriend" / "grizli-cutout"
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / cutout_cache_filename(ra, dec, size_arcsec, filters)
    if path.exists() and not overwrite:
        return path, path

    payload = await HTTPSession(timeout=timeout).fetch_content(url)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return path, path


def _float_token(value: float, digits: int = 6) -> str:
    """Format a float for stable filenames."""
    prefix = "p" if value >= 0 else "m"
    return f"{prefix}{abs(value):0.{digits}f}".replace(".", "p")


def _filter_base(filter_name: str) -> str:
    """Return the canonical lower-case base filter name."""
    return str(filter_name).lower().split("-", maxsplit=1)[0]


def _is_grism_filter(filter_name: str) -> bool:
    """Return whether ``filter_name`` is a grism-like filter."""
    lowered = str(filter_name).lower()
    return "grism" in lowered or "gr150" in lowered or lowered.startswith("g")


def _hdu_filter_name(hdu: fits.hdu.base.ExtensionHDU) -> str | None:
    """Infer the science filter name from a grizli HDU."""
    candidates = [
        hdu.header.get("FILTER"),
        hdu.header.get("FILTER1"),
        hdu.header.get("FILTER2"),
        hdu.header.get("PUPIL"),
        hdu.header.get("EXTNAME"),
        hdu.name,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        lowered = str(candidate).strip().lower()
        if not lowered or lowered in {"primary", "clear"}:
            continue
        base = _filter_base(lowered)
        if base in JWST_IMAGING_BASES or base in ACS_IMAGING_BASES:
            return lowered
        if base.startswith("f") and any(char.isdigit() for char in base):
            return lowered
    return None


def _mjy_scale_from_header(header: fits.Header) -> tuple[float | None, str | None]:
    """Infer the multiplicative raw-unit to mJy scale from a FITS header."""
    bunit = header.get("BUNIT")
    if bunit is not None:
        scale = _mjy_scale_from_bunit(str(bunit))
        if scale is not None:
            return scale, str(bunit)

    photfnu = _finite_header_float(header, "PHOTFNU")
    if photfnu is not None and photfnu > 0:
        return photfnu * 1e3, "PHOTFNU"

    for key in ("ZP", "ABMAG", "MAGZERO", "ZEROPOINT"):
        zp = _finite_header_float(header, key)
        if zp is not None and zp > 0:
            return 3631.0e3 * 10 ** (-0.4 * zp), key

    photflam = _finite_header_float(header, "PHOTFLAM")
    photplam = _finite_header_float(header, "PHOTPLAM")
    if photflam is not None and photplam is not None and photflam > 0 and photplam > 0:
        jy = photflam * photplam**2 / _C_AA_PER_S / 1e-23
        return jy * 1e3, "PHOTFLAM/PHOTPLAM"

    return None, None


def _mjy_scale_from_bunit(bunit: str) -> float | None:
    """Parse simple Jansky-like BUNIT strings into an mJy scale."""
    normalized = bunit.strip().lower().replace(" ", "")
    normalized = normalized.replace("janskys", "jansky")
    match = re.fullmatch(r"(?:(?P<factor>[0-9.eE+-]+)\*)?(?P<unit>[a-zµ]+)", normalized)
    if match is None:
        return None

    factor_text = match.group("factor")
    factor = 1.0 if factor_text is None else float(factor_text)
    unit = match.group("unit")
    unit_scale = {
        "jy": 1e3,
        "jansky": 1e3,
        "mjy": 1.0,
        "millijansky": 1.0,
        "ujy": 1e-3,
        "µjy": 1e-3,
        "microjansky": 1e-3,
        "njy": 1e-6,
        "nanojansky": 1e-6,
    }.get(unit)
    if unit_scale is None:
        return None
    return factor * unit_scale


def _finite_header_float(header: fits.Header, key: str) -> float | None:
    """Return a finite float header value when available."""
    if key not in header:
        return None
    try:
        value = float(header[key])
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _same_filter_data_pair(
    left: fits.hdu.base.ExtensionHDU, right: fits.hdu.base.ExtensionHDU
) -> bool:
    """Return whether two adjacent HDUs are a grizli science/weight pair."""
    left_name = _hdu_filter_name(left)
    right_name = _hdu_filter_name(right)
    if left_name is None or right_name is None:
        return False
    if _filter_base(left_name) != _filter_base(right_name):
        return False
    if left.data is None or right.data is None:
        return False
    return tuple(left.data.shape) == tuple(right.data.shape)
