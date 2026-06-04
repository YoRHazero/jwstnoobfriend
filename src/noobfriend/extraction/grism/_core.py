"""Rectify and combine WFSS exposures into 2-D grism spectra.

A :class:`GrismExtractor` couples a source sky position with a resolved output
grid spanning a spatial (cross-dispersion) axis and a wavelength axis. Each
covered exposure is rectified onto that common grid — by mapping every output
cell to its dispersed detector pixel through the grism WCS and resampling with
``noobase``'s exact polygon clip — so combining exposures is a plain
inverse-variance stack (no per-exposure wavelength resampling, and spatial
registration handled by the WCS).

The product is a :class:`GrismSpectrum`: an ordinary 2-D spectrum (``flux_2d`` +
``error_2d`` on a ``(spatial, wavelength)`` grid) ready to be displayed, with
convenience 1-D accessors that collapse along the spatial axis. Contamination
handling is deliberately a separate step; :class:`GrismSpectrum` carries an
optional ``contam_2d`` layer for it to fill in later.

This module is internal; :class:`GrismExtractor` and :class:`GrismSpectrum` are
re-exported from ``noobfriend.extraction.grism``.
"""

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np
from noobase.axis import Grid
from noobase.image import reproject_exact
from noobase.spectroscopy import Spectrum

from noobfriend.extraction._wcs import grism_trace_transform
from noobfriend.extraction.grism._array import _native
from noobfriend.extraction.grism._coverage import FrameCoverage


def _trace_centerline(
    trace: Callable[..., tuple[np.ndarray, np.ndarray]],
    x0: float,
    y0: float,
    wavelength: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a grism trace at one source position over a wavelength array.

    Evaluates the dispersed detector pixel of a single undispersed source
    position ``(x0, y0)`` at each wavelength. Passing scalar ``x0``/``y0`` with a
    1-D ``wavelength`` keeps the dispersion model's evaluation element-wise (a
    clean 1-D result), avoiding the quadratic outer product it produces when all
    three inputs are arrays. The spatial (cross-dispersion) axis of a rectified
    beam is then reconstructed analytically from this centerline, exploiting
    grism separability — see :meth:`GrismExtractor.rectify`.

    Parameters
    ----------
    trace : collections.abc.Callable
        ``trace(x0, y0, wavelength) -> (x_grism, y_grism)`` for order 1, from
        :func:`~noobfriend.extraction._wcs.grism_trace_transform`.
    x0, y0 : float
        Undispersed source pixel position.
    wavelength : numpy.ndarray
        1-D array of wavelengths in microns.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(x_grism, y_grism)``, each 1-D with the shape of ``wavelength``.
    """
    xg, yg = trace(float(x0), float(y0), np.asarray(wavelength, dtype=float))
    return np.asarray(xg, dtype=float), np.asarray(yg, dtype=float)


@dataclass(frozen=True)
class GrismSpectrum:
    """A rectified 2-D grism spectrum on a ``(spatial, wavelength)`` grid.

    The 2-D arrays are the primary product; :attr:`flux_1d` / :attr:`error_1d`
    and :meth:`collapse` derive a 1-D spectrum from them on demand.

    Attributes
    ----------
    flux_2d : numpy.ndarray
        Surface brightness, shape ``(n_spatial, n_wave)``; empty cells ``NaN``.
    error_2d : numpy.ndarray
        Propagated 1-sigma error, same shape as ``flux_2d``.
    weight_2d : numpy.ndarray
        Valid-overlap fraction in ``[0, 1]`` per cell (coverage map).
    wavelength : noobase.axis.Grid
        The wavelength axis (length ``n_wave``), in microns.
    spatial_offset : noobase.axis.Grid
        The cross-dispersion axis (length ``n_spatial``), in detector-y pixels,
        zero at the source's trace centre.
    n_frames : int
        Number of exposures combined into this product.
    group : collections.abc.Hashable | None, optional
        The user grouping key shared by the combined exposures (from
        ``FrameMeta.group``); ``None`` when no grouping was used.
    contam_2d : numpy.ndarray | None, optional
        Optional contamination model on the same grid, subtracted before a 1-D
        collapse. Filled by the separate contamination step; ``None`` here.
    """

    flux_2d: np.ndarray
    error_2d: np.ndarray
    weight_2d: np.ndarray
    wavelength: Grid
    spatial_offset: Grid
    n_frames: int
    group: Hashable | None = None
    contam_2d: np.ndarray | None = None

    @property
    def flux_1d(self) -> np.ndarray:
        """1-D flux from the default spatial collapse, on the native grid.

        Equivalent to ``self.collapse()[0]``.

        Returns
        -------
        numpy.ndarray
            Flux of shape ``(n_wave,)``.
        """
        return self.collapse()[0]

    @property
    def error_1d(self) -> np.ndarray:
        """1-D error from the default spatial collapse, on the native grid.

        Equivalent to ``self.collapse()[1]``.

        Returns
        -------
        numpy.ndarray
            1-sigma error of shape ``(n_wave,)``.
        """
        return self.collapse()[1]

    def collapse(
        self, *, wavelength: Grid | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Collapse the 2-D spectrum to 1-D along the spatial axis.

        The default collapse is a boxcar sum over the full spatial extent, with
        ``contam_2d`` subtracted first when present, error propagated in
        quadrature over valid (finite, non-zero-weight) cells, and output bins
        with no valid contribution set to ``NaN``. (Profile-weighted / optimal
        extraction is a separate concern and may be layered on later.)

        Parameters
        ----------
        wavelength : noobase.axis.Grid | None, optional
            Target wavelength axis. ``None`` (default) returns the 1-D spectrum
            on this object's native :attr:`wavelength`. When given, the collapsed
            spectrum is flux-conservingly rebinned onto it via ``noobase``.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            ``(flux_1d, error_1d)``, each of shape ``(n_wave,)`` on the native
            grid, or matching ``wavelength`` when supplied.
        """
        base = self.flux_2d if self.contam_2d is None else self.flux_2d - self.contam_2d
        valid = (
            np.isfinite(base)
            & np.isfinite(self.error_2d)
            & (self.error_2d > 0)
            & (self.weight_2d > 0)
        )
        n_valid = valid.sum(axis=0)
        flux_1d = np.where(valid, base, 0.0).sum(axis=0)
        var_1d = np.where(valid, self.error_2d**2, 0.0).sum(axis=0)
        with np.errstate(invalid="ignore"):
            flux_1d = np.where(n_valid > 0, flux_1d, np.nan)
            error_1d = np.where(n_valid > 0, np.sqrt(var_1d), np.nan)
        if wavelength is None:
            return (flux_1d, error_1d)
        spec = Spectrum(wavelength=self.wavelength, flux=flux_1d, error=error_1d)
        r = spec.rebin(wavelength)
        return (np.asarray(r.flux), np.asarray(r.error))


@dataclass(frozen=True)
class GrismExtractor:
    """A source's grism extraction, resolved against a set of covered exposures.

    Construct with :meth:`from_world`, which resolves the common output grid at
    construction time. Then call :meth:`rectify` to turn each covered exposure's
    pixels into a per-exposure 2-D spectrum, and :meth:`combine` to stack those
    into one :class:`GrismSpectrum` per group.

    Attributes
    ----------
    ra, dec : float
        Source world coordinates in degrees.
    wavelength : noobase.axis.Grid
        The resolved common wavelength axis, in microns.
    spatial_offset : noobase.axis.Grid
        The resolved common cross-dispersion axis, in detector-y pixels.
    coverage : tuple[FrameCoverage, ...]
        The covered exposures backing this extraction.
    """

    ra: float
    dec: float
    wavelength: Grid
    spatial_offset: Grid
    coverage: tuple[FrameCoverage, ...]

    @classmethod
    def from_world(
        cls,
        ra: float,
        dec: float,
        frames: Sequence[FrameCoverage],
        *,
        spatial_half: int,
        oversample: int = 1,
        wavelength: Grid | None = None,
    ) -> Self:
        """Resolve the common output grid for an extraction.

        Builds the ``(spatial_offset, wavelength)`` grid from the covered
        exposures' coverage; does no resampling and needs no pixel data — that
        is deferred to :meth:`rectify` / :meth:`combine`.

        Parameters
        ----------
        ra, dec : float
            Source world coordinates in degrees.
        frames : Sequence[FrameCoverage]
            Covered exposures, as returned (and filtered on ``covered``) by
            :func:`~noobfriend.extraction.grism.find_coverage`.
        spatial_half : int
            Half-height of the cross-dispersion axis in detector-y pixels; the
            spatial axis spans ``[-spatial_half, +spatial_half]`` about the
            source, giving ``2 * spatial_half * oversample + 1`` rows.
        oversample : int, optional
            Sub-pixel sampling factor of the spatial axis, by default ``1``.
        wavelength : noobase.axis.Grid | None, optional
            Explicit wavelength axis. ``None`` (default) builds it automatically
            from the union of the exposures' covered ranges, sampled at the
            median native dispersion.

        Returns
        -------
        GrismExtractor
        """
        n_spatial = 2 * spatial_half * oversample + 1
        spatial_offset = Grid.linspace(
            -spatial_half, spatial_half, n_spatial, kind="centers"
        )

        if wavelength is None:
            ranged = [f for f in frames if f.wavelength_range is not None]
            if not ranged:
                raise ValueError(
                    "Cannot build a wavelength axis: no covered frame has a "
                    "wavelength_range. Pass an explicit `wavelength` grid."
                )
            lo = min(f.wavelength_range[0] for f in ranged)
            hi = max(f.wavelength_range[1] for f in ranged)

            dispersions: list[float] = []
            for f in ranged:
                trace = grism_trace_transform(f.meta.wcs)
                x0, y0 = f.source_xy
                flo, fhi = f.wavelength_range
                xa, ya = trace(x0, y0, flo)
                xb, yb = trace(x0, y0, fhi)
                span_px = float(np.hypot(float(xb) - float(xa), float(yb) - float(ya)))
                if span_px > 0:
                    dispersions.append(abs(fhi - flo) / span_px)
            dlam_dpix = float(np.median(dispersions))
            n_wave = max(2, int(round((hi - lo) / dlam_dpix)) + 1)
            wavelength = Grid.linspace(lo, hi, n_wave, kind="centers")

        return cls(
            ra=ra,
            dec=dec,
            wavelength=wavelength,
            spatial_offset=spatial_offset,
            coverage=tuple(frames),
        )

    def rectify(
        self, load: Callable[[str], tuple[np.ndarray, np.ndarray]]
    ) -> dict[str, GrismSpectrum]:
        """Rectify each covered exposure onto the common grid, separately.

        Each exposure is mapped onto the ``(spatial_offset, wavelength)`` grid
        by computing, per output cell, its dispersed detector pixel and
        resampling with :func:`noobase.image.reproject_exact` (carrying the
        error through). Useful for QA and display before combining, including
        per-exposure or per-module inspection (the ``id`` keys identify each).
        Each beam carries its ``group`` (from ``FrameMeta.group``) for
        :meth:`combine`.

        Parameters
        ----------
        load : collections.abc.Callable
            ``load(id) -> (data, error)``, called once per covered exposure to
            fetch its pixels lazily. A dict satisfies this via ``__getitem__``.

        Returns
        -------
        dict[str, GrismSpectrum]
            One single-exposure :class:`GrismSpectrum` (``n_frames == 1``) per
            exposure, keyed by ``FrameMeta.id``.
        """
        se = self.spatial_offset.to_edges().values
        we = self.wavelength.to_edges().values
        spectra: dict[str, GrismSpectrum] = {}
        for cov in self.coverage:
            data, error = load(cov.meta.id)
            data = _native(data)
            error = _native(error)

            x0, y0 = cov.source_xy
            trace = grism_trace_transform(cov.meta.wcs)

            # Grism separability: a cross-dispersion (detector-y) offset of the
            # source shifts the dispersed trace almost purely in grism-y (slope
            # ~1; grism-x coupling < 0.02 px across the on-array band), so the
            # trace is sampled once per wavelength at the source and the spatial
            # axis is reconstructed analytically -- O(n_wave), not
            # O(n_spatial * n_wave), avoiding the model's quadratic outer product.
            xc, yc = _trace_centerline(trace, x0, y0, we)
            xg = np.broadcast_to(xc, (se.size, we.size))
            yg = yc[None, :] + se[:, None]
            corners = np.stack(
                [np.ascontiguousarray(xg), np.ascontiguousarray(yg)], axis=-1
            )

            res = reproject_exact(data, corners, error=error)
            spectra[cov.meta.id] = GrismSpectrum(
                flux_2d=res.image,
                error_2d=res.error,
                weight_2d=res.weight,
                wavelength=self.wavelength,
                spatial_offset=self.spatial_offset,
                n_frames=1,
                group=cov.meta.group,
                contam_2d=None,
            )
        return spectra

    def combine(self, spec_map: Mapping[str, GrismSpectrum]) -> list[GrismSpectrum]:
        """Combine rectified per-exposure beams into one spectrum per group.

        The downstream of :meth:`rectify`: it does no loading and no resampling,
        only stacks the already-rectified, co-gridded beams onto this
        extractor's grid by inverse-variance weighting. Beams are partitioned
        solely by their ``group`` (from ``FrameMeta.group``); nothing else is
        imposed — exposures of different dispersion (GRISMR + GRISMC) are
        combined too when they share a group, so keeping them apart is the
        caller's choice via ``group``, not a rule enforced here.

        Parameters
        ----------
        spec_map : Mapping[str, GrismSpectrum]
            Rectified per-exposure spectra, as returned by :meth:`rectify`.

        Returns
        -------
        list[GrismSpectrum]
            One combined :class:`GrismSpectrum` per distinct ``group`` present —
            a single element when all beams share one group (the default).
        """
        groups: dict[Hashable | None, list[GrismSpectrum]] = {}
        for spec in spec_map.values():
            groups.setdefault(spec.group, []).append(spec)

        products: list[GrismSpectrum] = []
        for group_key, specs in groups.items():
            flux = np.stack([s.flux_2d for s in specs])
            err = np.stack([s.error_2d for s in specs])
            wt = np.stack([s.weight_2d for s in specs])

            valid = np.isfinite(flux) & np.isfinite(err) & (err > 0)
            w = np.where(valid, 1.0 / np.where(valid, err, 1.0) ** 2, 0.0)
            wsum = w.sum(axis=0)

            with np.errstate(invalid="ignore", divide="ignore"):
                flux_c = np.where(
                    wsum > 0,
                    (np.where(valid, flux, 0.0) * w).sum(axis=0) / wsum,
                    np.nan,
                )
                err_c = np.where(wsum > 0, np.sqrt(1.0 / wsum), np.nan)
                wt_c = np.nanmean(np.where(np.isfinite(wt), wt, np.nan), axis=0)

            products.append(
                GrismSpectrum(
                    flux_2d=flux_c,
                    error_2d=err_c,
                    weight_2d=wt_c,
                    wavelength=specs[0].wavelength,
                    spatial_offset=specs[0].spatial_offset,
                    n_frames=len(specs),
                    group=group_key,
                    contam_2d=None,
                )
            )
        return products
