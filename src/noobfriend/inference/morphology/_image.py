"""``NoobImage``: the multi-band imaging dataset of one target.

A :class:`NoobImage` is the sole public data abstraction of the morphology module
-- the spatial analogue of
:class:`~noobfriend.inference.spectrum.NoobSpectrum`. It carries one target's
per-band cutouts (each on its own native grid, held as an internal
:class:`~noobfriend.inference.morphology._band.Band`) together with the target
identity a fit needs but a bare array cannot supply: the redshift ``z`` and the
sky ``center`` (used both to validate that merged bands really are the same
target and to project the shared sky-frame model parameters into each band).

Build one from a mapping of per-band dicts with :meth:`from_bands`; carve out a
subset of bands with :meth:`select` (the manual per-band include toggle); assemble
bands gathered from separate sources with :meth:`merge`; and override fit masks
after inspecting the auto-derived ones with :meth:`with_masks`. Every operation
returns a new image -- :class:`NoobImage` is immutable.

This module is internal; :class:`NoobImage` is re-exported from
:mod:`noobfriend.inference.morphology`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from noobfriend.inference.morphology._band import Band, normalize_band

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

#: Absolute tolerance on ``z`` when merging two images of the "same" target.
_Z_MERGE_ATOL: float = 1e-4
#: Default tolerance (arcsec) on the sky centre when merging the "same" target.
_CENTER_MERGE_TOL_ARCSEC: float = 0.5


@dataclass(frozen=True, eq=False, repr=False)
class NoobImage:
    """One target's multi-band imaging, ready to be decomposed.

    Built via :meth:`from_bands`; not constructed directly.

    Attributes
    ----------
    bands : dict[str, Band]
        Per-band records keyed by filter, in insertion order. ``Band`` is an
        internal type; reach for :meth:`filters` and :meth:`select` rather than
        indexing this directly.
    z : float
        Systemic redshift of the target (``> -1``); used for the angular-to-kpc
        conversion and the high-z default priors.
    center : tuple[float, float]
        Target sky position ``(ra, dec)`` in degrees (ICRS): the identity checked
        when merging, and the anchor the shared sky-frame parameters project from.
    """

    bands: dict[str, Band]
    z: float
    center: tuple[float, float]

    @classmethod
    def from_bands(
        cls,
        bands: Mapping[str, Any],
        *,
        z: float,
        center: tuple[float, float],
        cutout_size_arcsec: float | None = None,
    ) -> NoobImage:
        """Build a multi-band image from a mapping of per-band dicts.

        Parameters
        ----------
        bands : mapping of str to mapping
            Filter name -> :class:`~noobfriend.inference.morphology.BandSpec`-shaped
            dict (required ``"data"`` / ``"error"`` plus optional ``"wcs"`` /
            ``"psf"`` / ``"mask"``). Must be non-empty.
        z : float
            Systemic redshift (``> -1``).
        center : tuple of float
            Target ``(ra, dec)`` in degrees (ICRS).
        cutout_size_arcsec : float, optional
            Full angular size of the cutouts, used only to synthesise a
            tangent-plane WCS for any band that omits ``"wcs"`` (see
            :func:`noobfriend.extraction._wcs.tangent_plane_wcs`). Required if and
            only if some band has no WCS.

        Returns
        -------
        NoobImage

        Raises
        ------
        ValueError
            If ``bands`` is empty, ``z <= -1``, ``center`` is malformed, a band
            spec is invalid, or a band omits ``"wcs"`` with no ``cutout_size_arcsec``
            to synthesise one.
        """
        if not bands:
            raise ValueError("bands must be a non-empty mapping of filter -> spec.")
        z_out = _validated_z(z)
        center_out = _validated_center(center)
        ra, dec = center_out

        resolved: dict[str, Band] = {}
        for name, spec in bands.items():
            band = normalize_band(name, spec)
            if band.wcs is None:
                if cutout_size_arcsec is None:
                    raise ValueError(
                        f"band {name!r} has no 'wcs'; either provide one, or pass "
                        "cutout_size_arcsec so a tangent-plane WCS can be synthesised "
                        "from the target centre and the cutout size."
                    )
                from noobfriend.extraction._wcs import tangent_plane_wcs

                wcs = tangent_plane_wcs(ra, dec, cutout_size_arcsec, band.shape)
                band = replace(band, wcs=wcs)
            resolved[str(name)] = band

        return cls(bands=resolved, z=z_out, center=center_out)

    @property
    def filters(self) -> list[str]:
        """Return the band filter names, in order."""
        return list(self.bands)

    def __len__(self) -> int:
        """Return the number of bands."""
        return len(self.bands)

    def __contains__(self, filter_name: object) -> bool:
        """Return whether ``filter_name`` is one of this image's bands."""
        return filter_name in self.bands

    def select(self, filters: Sequence[str]) -> NoobImage:
        """Return a sub-image holding only the named bands, in the given order.

        This is the manual per-band include toggle: fit a subset (e.g. drop a band
        whose error calibration you distrust, or test what a band adds) without
        rebuilding the image.

        Parameters
        ----------
        filters : sequence of str
            Band names to keep; must be non-empty, present, and free of duplicates.

        Returns
        -------
        NoobImage
            A new image with the selected bands; ``z`` and ``center`` are preserved.

        Raises
        ------
        ValueError
            If ``filters`` is empty, contains a duplicate, or names a band absent
            from this image.
        """
        wanted = list(filters)
        if not wanted:
            raise ValueError("select needs at least one band.")
        if len(set(wanted)) != len(wanted):
            raise ValueError(f"select got duplicate band(s): {wanted}.")
        missing = [f for f in wanted if f not in self.bands]
        if missing:
            raise ValueError(
                f"unknown band(s) {missing}; this image has {self.filters}."
            )
        return NoobImage(
            bands={f: self.bands[f] for f in wanted}, z=self.z, center=self.center
        )

    def merge(
        self, *others: NoobImage, tol_arcsec: float = _CENTER_MERGE_TOL_ARCSEC
    ) -> NoobImage:
        """Combine this image's bands with those of other same-target images.

        Use it to assemble bands gathered from separate sources (different cutout
        extractions, pipelines, or visits) into one multi-band image. Only the
        band union and a same-target check are done -- each band keeps its own
        native grid; nothing is reprojected.

        Parameters
        ----------
        *others : NoobImage
            Images to merge in. Each must share this image's ``z`` and ``center``
            (within tolerance) and contribute only bands not already present.
        tol_arcsec : float, default 0.5
            Maximum sky separation between centres for them to count as the same
            target.

        Returns
        -------
        NoobImage
            A new image whose bands are this image's followed by each other's;
            ``z`` and ``center`` are taken from this image.

        Raises
        ------
        ValueError
            If any other image's ``z`` or ``center`` disagrees beyond tolerance, or
            a band name collides with one already present.
        """
        merged: dict[str, Band] = dict(self.bands)
        for other in others:
            if not np.isclose(self.z, other.z, rtol=0.0, atol=_Z_MERGE_ATOL):
                raise ValueError(
                    f"cannot merge: z differs ({self.z} vs {other.z}); these are "
                    "not the same target."
                )
            sep = _angular_sep_arcsec(self.center, other.center)
            if sep > tol_arcsec:
                raise ValueError(
                    f"cannot merge: centres are {sep:.3g} arcsec apart "
                    f"(tol {tol_arcsec} arcsec); these are not the same target."
                )
            clash = set(merged) & set(other.bands)
            if clash:
                raise ValueError(
                    f"cannot merge: duplicate band(s) {sorted(clash)}; each band "
                    "may come from only one image."
                )
            merged.update(other.bands)
        return NoobImage(bands=merged, z=self.z, center=self.center)

    def with_masks(self, masks: Mapping[str, np.ndarray | None]) -> NoobImage:
        """Return a copy with the fit masks of the named bands overridden.

        The companion to auto-mask derivation: inspect the masks the model picked,
        and where they are wrong, hand back corrected ones (or ``None`` to revert a
        band to auto-derivation). Bands not named keep their current mask.

        Parameters
        ----------
        masks : mapping of str to (numpy.ndarray or None)
            Filter name -> boolean mask (``True`` = excluded, same shape as that
            band's data), or ``None`` to clear that band's mask.

        Returns
        -------
        NoobImage

        Raises
        ------
        ValueError
            If a key names a band absent from this image, or a mask shape does not
            match its band's data.
        """
        missing = [f for f in masks if f not in self.bands]
        if missing:
            raise ValueError(
                f"unknown band(s) {missing}; this image has {self.filters}."
            )
        updated = {
            name: (band.with_mask(masks[name]) if name in masks else band)
            for name, band in self.bands.items()
        }
        return NoobImage(bands=updated, z=self.z, center=self.center)

    def model(self, components: Sequence[Any], **kwargs: Any) -> Any:
        """Bind profile components to this image and build a fittable model.

        A thin convenience for
        :meth:`noobfriend.inference.morphology.MorphModel.build`; imported lazily
        so the data layer carries no JAX/numpyro dependency.

        Parameters
        ----------
        components : sequence of Profile
            The declared components.
        **kwargs
            Forwarded to :meth:`MorphModel.build` (``oversample``, ``policy``,
            ``nu``, ``background``, ``auto_mask``, ``joint_mask``,
            ``fit_radius_arcsec``, ``cosmology``, ...).

        Returns
        -------
        MorphModel
        """
        from noobfriend.inference.morphology._model import MorphModel

        return MorphModel.build(self, components, **kwargs)

    def __repr__(self) -> str:
        """Concise summary of bands, redshift, and centre."""
        ra, dec = self.center
        return (
            f"NoobImage(bands={self.filters}, z={self.z:g}, center=({ra:g}, {dec:g}))"
        )


def _validated_z(z: float) -> float:
    """Validate the redshift.

    Raises
    ------
    ValueError
        If ``z <= -1``.
    """
    if z <= -1:
        raise ValueError(f"z must be > -1, got {z}.")
    return float(z)


def _validated_center(center: tuple[float, float]) -> tuple[float, float]:
    """Validate a ``(ra, dec)`` sky position in degrees.

    Raises
    ------
    ValueError
        If ``center`` is not a length-2 finite pair, or ``dec`` is out of range.
    """
    try:
        ra, dec = float(center[0]), float(center[1])
    except (TypeError, IndexError, ValueError) as exc:
        raise ValueError(
            f"center must be a (ra, dec) pair in degrees, got {center!r}."
        ) from exc
    if not (math.isfinite(ra) and math.isfinite(dec)):
        raise ValueError(f"center (ra, dec) must be finite, got {center!r}.")
    if not -90.0 <= dec <= 90.0:
        raise ValueError(f"center dec must be in [-90, 90], got {dec}.")
    return ra, dec


def _angular_sep_arcsec(c1: tuple[float, float], c2: tuple[float, float]) -> float:
    """Small-angle sky separation between two ``(ra, dec)`` degrees, in arcsec."""
    ra1, dec1 = c1
    ra2, dec2 = c2
    dec_mid = math.radians((dec1 + dec2) / 2.0)
    dra = (ra1 - ra2) * math.cos(dec_mid)
    ddec = dec1 - dec2
    return math.hypot(dra, ddec) * 3600.0
