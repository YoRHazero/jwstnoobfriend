"""The head of the fitting chain: one source observed in one or more frames."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from noobfriend.inference.spectrum._wavelength import (
    WaveUnit,
    check_unit,
    in_intervals,
    parse_intervals,
)
from noobfriend.inference.spectrum._xp import readonly
from noobfriend.inference.spectrum.data._frame import (
    Source2D,
    SpectrumFrame,
    build_frame,
    check_frame_id,
)
from noobfriend.inference.spectrum.noise import Noise, estimated_noise

if TYPE_CHECKING:
    from noobfriend.inference.spectrum.workspace import NoobFitWorkspace

_DEFAULT_FRAME = "frame_0"


@dataclass(frozen=True, slots=True, eq=False, init=False)
class NoobSpectrum:
    """One source's spectrum in one or more frames: the head of the fit chain.

    A single 1-D spectrum is one frame (id ``"frame_0"``); :meth:`join` puts
    several single-frame spectra of the same source together. Every
    transformation returns a new object.

    Parameters
    ----------
    obs
        Observed-frame wavelengths, strictly increasing.
    flux, error
        Flux density and its 1-sigma error per pixel. Non-finite values or
        non-positive errors mark a pixel invalid; it stays in place and is
        skipped by fits.
    unit
        Wavelength unit of ``obs``: ``"angstrom"``, ``"nm"`` or ``"um"``.
    resolving_power
        ``R`` of the gaussian line-spread function: a number, a
        ``(wavelength, R)`` table interpolated linearly at each line's
        observed wavelength, or ``None`` for no LSF. Required, so leaving the
        LSF out is always a decision.
    z
        Source redshift; needed for rest-frame lines and rest-frame ranges.
    mask
        ``True`` on pixels to exclude from fitting. Pixel masks are given only
        here; later exclusions go through :meth:`exclude`.

    Attributes
    ----------
    frames : Mapping[str, SpectrumFrame]
        The frames by id, in order.
    z : float or None
        Source redshift.
    unit : str
        Wavelength unit shared by every frame.
    """

    frames: Mapping[str, SpectrumFrame]
    z: float | None
    unit: WaveUnit

    def __init__(
        self,
        obs: ArrayLike,
        flux: ArrayLike,
        error: ArrayLike,
        *,
        unit: WaveUnit,
        resolving_power: float | tuple[ArrayLike, ArrayLike] | None,
        z: float | None = None,
        mask: ArrayLike | None = None,
    ) -> None:
        """Validate the arrays and build a single-frame spectrum."""
        frame = build_frame(
            _DEFAULT_FRAME, obs, flux, error, mask=mask, resolving_power=resolving_power
        )
        _init(self, {frame.id: frame}, _check_z(z), check_unit(unit))

    @classmethod
    def from_2d(
        cls,
        flux_2d: ArrayLike,
        error_2d: ArrayLike,
        *,
        obs: ArrayLike,
        collapse_window: tuple[int, int],
        unit: WaveUnit,
        resolving_power: float | tuple[ArrayLike, ArrayLike] | None,
        dispersion_axis: int = 1,
        spatial: ArrayLike | None = None,
        z: float | None = None,
        mask: ArrayLike | None = None,
    ) -> NoobSpectrum:
        """Collapse a 2-D spectrum into a single-frame spectrum and keep the 2-D.

        The 1-D flux sums ``collapse_window`` rows; its error is the naive
        quadrature sum. Correlated noise is described afterwards with
        :meth:`noise_from_background` (or ``Noise(scale=k)`` by hand).

        Parameters
        ----------
        flux_2d, error_2d
            The 2-D spectrum and its per-pixel error.
        obs
            Observed wavelength of each dispersion pixel.
        collapse_window
            ``(lo, hi)`` half-open index range on the cross-dispersion axis.
        unit, resolving_power, z
            As for the constructor.
        dispersion_axis
            Axis of ``flux_2d`` running along wavelength.
        spatial
            Cross-dispersion coordinate of each row; pixel index by default.
        mask
            1-D mask over dispersion pixels, ``True`` to exclude.

        Returns
        -------
        NoobSpectrum
            One frame carrying a :class:`Source2D`.

        Raises
        ------
        ValueError
            If the shapes, axis or window are inconsistent.
        """
        from noobfriend.core.specutils import collapse

        if dispersion_axis not in (0, 1):
            raise ValueError(
                f"dispersion_axis must be 0 or 1; got {dispersion_axis!r}."
            )
        flux = np.asarray(flux_2d, dtype=float)
        error = np.asarray(error_2d, dtype=float)
        if flux.ndim != 2 or flux.shape != error.shape:
            raise ValueError(
                f"flux_2d {flux.shape} and error_2d {error.shape} must be equal 2-D arrays."
            )
        canonical_flux = flux.T if dispersion_axis == 0 else flux
        canonical_error = error.T if dispersion_axis == 0 else error
        n_rows = canonical_flux.shape[0]
        lo, hi = (int(collapse_window[0]), int(collapse_window[1]))
        if not 0 <= lo < hi <= n_rows:
            raise ValueError(
                f"collapse_window {collapse_window} is outside the {n_rows} cross-dispersion rows."
            )
        coordinates = np.arange(n_rows, dtype=float) if spatial is None else spatial
        coordinates = readonly(coordinates)
        if coordinates.shape != (n_rows,):
            raise ValueError(f"spatial must have one value per row ({n_rows}).")
        source = Source2D(
            flux=readonly(canonical_flux),
            error=readonly(canonical_error),
            spatial=coordinates,
            collapse_window=(lo, hi),
            dispersion_axis=dispersion_axis,
        )
        flux_1d, error_1d = collapse(
            canonical_flux, canonical_error, window=(lo, hi), dispersion_axis=1
        )
        frame = build_frame(
            _DEFAULT_FRAME,
            obs,
            flux_1d,
            error_1d,
            mask=mask,
            resolving_power=resolving_power,
            source_2d=source,
        )
        return _new(cls, {frame.id: frame}, _check_z(z), check_unit(unit))

    @classmethod
    def join(
        cls, spectra: Mapping[str, NoobSpectrum] | Sequence[NoobSpectrum]
    ) -> NoobSpectrum:
        """Combine single-frame spectra of one source into a multi-frame spectrum.

        Frames are kept side by side, never co-added.

        Parameters
        ----------
        spectra
            A mapping from frame id to single-frame spectrum, or a sequence
            (ids ``frame_0``, ``frame_1``, ...).

        Returns
        -------
        NoobSpectrum
            One frame per input, in order. Each frame keeps its own resolving
            power and noise.

        Raises
        ------
        ValueError
            If an input has several frames, the inputs disagree on ``z`` or
            ``unit``, or an id is invalid.
        """
        if isinstance(spectra, Mapping):
            items = list(spectra.items())
        elif isinstance(spectra, Sequence):
            items = [
                (f"frame_{index}", spectrum) for index, spectrum in enumerate(spectra)
            ]
        else:
            raise TypeError("join expects a mapping or a sequence of NoobSpectrum.")
        if not items:
            raise ValueError("join needs at least one spectrum.")
        frames: dict[str, SpectrumFrame] = {}
        for frame_id, spectrum in items:
            if not isinstance(spectrum, NoobSpectrum):
                raise TypeError(
                    f"join inputs must be NoobSpectrum; got {type(spectrum).__name__}."
                )
            if len(spectrum.frames) != 1:
                raise ValueError(
                    f"join takes single-frame spectra; {frame_id!r} has "
                    f"{len(spectrum.frames)} frames."
                )
            (frame,) = spectrum.frames.values()
            frames[check_frame_id(frame_id)] = replace(frame, id=frame_id)
        first = items[0][1]
        for frame_id, spectrum in items[1:]:
            if spectrum.unit != first.unit:
                raise ValueError(
                    f"{frame_id!r} has unit {spectrum.unit!r}, not {first.unit!r}."
                )
            if spectrum.z != first.z:
                raise ValueError(f"{frame_id!r} has z={spectrum.z!r}, not {first.z!r}.")
        return _new(cls, frames, first.z, first.unit)

    @property
    def frame_ids(self) -> tuple[str, ...]:
        """Frame ids in order."""
        return tuple(self.frames)

    def select(self, *frame_ids: str) -> NoobSpectrum:
        """Keep only the named frames (in the given order).

        Raises
        ------
        KeyError
            If an id is unknown.
        ValueError
            If no id is given.
        """
        if not frame_ids:
            raise ValueError("select needs at least one frame id.")
        return self._with_frames(
            {frame_id: self._frame(frame_id) for frame_id in frame_ids}
        )

    def drop(self, *frame_ids: str) -> NoobSpectrum:
        """Remove the named frames.

        Raises
        ------
        KeyError
            If an id is unknown.
        ValueError
            If no frame would remain.
        """
        for frame_id in frame_ids:
            self._frame(frame_id)
        kept = {
            key: frame for key, frame in self.frames.items() if key not in frame_ids
        }
        if not kept:
            raise ValueError("drop would remove every frame.")
        return self._with_frames(kept)

    def exclude(self, *, obs: object = None, rest: object = None) -> NoobSpectrum:
        """Mask wavelength ranges in every frame.

        Parameters
        ----------
        obs
            ``(lo, hi)`` or a sequence of ranges, observed frame, in
            :attr:`unit`.
        rest
            The same in the rest frame; needs :attr:`z`.

        Returns
        -------
        NoobSpectrum
            A copy whose frame masks also cover the ranges (closed intervals).
        """
        if obs is None and rest is None:
            raise ValueError("exclude needs obs=... and/or rest=... ranges.")
        intervals = self._observed_intervals(obs, rest, "obs", "rest")
        frames = {}
        for frame_id, frame in self.frames.items():
            mask = readonly(frame.mask | in_intervals(frame.obs, intervals), dtype=bool)
            frames[frame_id] = replace(frame, mask=mask)
        return self._with_frames(frames)

    def with_noise(self, noise: Noise | Mapping[str, Noise]) -> NoobSpectrum:
        """Attach noise models, replacing the frames' previous ones.

        Parameters
        ----------
        noise
            One :class:`~noobfriend.inference.spectrum.noise.Noise` for every
            frame (each frame still gets its own parameters), or a mapping
            from frame id to ``Noise`` for some frames.

        Returns
        -------
        NoobSpectrum
            A copy with the new noise models. Error arrays are never changed.
        """
        if isinstance(noise, Noise):
            updates = dict.fromkeys(self.frames, noise)
        elif isinstance(noise, Mapping):
            updates = dict(noise)
            for frame_id, value in updates.items():
                self._frame(frame_id)
                if not isinstance(value, Noise):
                    raise TypeError(f"noise for {frame_id!r} must be a Noise.")
        else:
            raise TypeError(
                "with_noise expects a Noise or a {frame_id: Noise} mapping."
            )
        frames = {
            frame_id: replace(frame, noise=updates.get(frame_id, frame.noise))
            for frame_id, frame in self.frames.items()
        }
        return self._with_frames(frames)

    def noise_from_continuum(
        self,
        *,
        lags: int = 0,
        exclude_obs: object = None,
        exclude_rest: object = None,
    ) -> dict[str, Noise]:
        """Estimate each frame's noise from its line-free 1-D scatter.

        Uses each frame's usable pixels minus the extra ranges, fits a
        sigma-clipped cubic continuum and measures the error-normalized
        residual (see :func:`noobfriend.core.specutils.continuum_noise`). Use
        it only without a 2-D source: the polynomial absorbs low-frequency
        noise, so it underestimates correlated noise.

        Parameters
        ----------
        lags
            Estimate ``rho(1 ... lags)`` too; ``0`` estimates the scale only.
        exclude_obs, exclude_rest
            Extra ``(lo, hi)`` ranges to leave out (lines), observed or rest
            frame.

        Returns
        -------
        dict of str to Noise
            Fixed estimates per frame, to pass to :meth:`with_noise`.

        Raises
        ------
        ValueError
            If an estimate fails (too few pixels, invalid correlation); the
            message names the frame and the reason.
        """
        from noobfriend.core.specutils import continuum_noise

        intervals = ()
        if exclude_obs is not None or exclude_rest is not None:
            intervals = self._observed_intervals(
                exclude_obs, exclude_rest, "exclude_obs", "exclude_rest"
            )
        estimates = {}
        for frame_id, frame in self.frames.items():
            excluded = ~frame.usable | in_intervals(frame.obs, intervals)
            estimates[frame_id] = _estimate(
                frame_id,
                lambda frame=frame, excluded=excluded: continuum_noise(
                    frame.flux,
                    frame.error,
                    wavelength=frame.obs,
                    exclude=excluded,
                    lags=lags,
                ),
            )
        return estimates

    def noise_from_background(
        self,
        *,
        source_mask: ArrayLike | Mapping[str, ArrayLike] | None = None,
        lags: int = 0,
    ) -> dict[str, Noise]:
        """Estimate each frame's noise from the sky pixels of its 2-D source.

        See :func:`noobfriend.core.specutils.background_noise`: the 2-D
        autocovariance of the error-normalized sky, summed over the collapse
        window, gives the collapsed spectrum's scale and dispersion-axis
        correlation (including per-pixel error mis-calibration).

        Parameters
        ----------
        source_mask
            ``True`` where the 2-D cutout holds source flux, in the
            orientation passed to :meth:`from_2d`; one array for every frame
            or a ``{frame_id: array}`` mapping. Defaults to a segmentation.
        lags
            Estimate ``rho(1 ... lags)`` too; ``0`` estimates the scale only.

        Returns
        -------
        dict of str to Noise
            Fixed estimates per frame, to pass to :meth:`with_noise`.

        Raises
        ------
        ValueError
            If a frame has no 2-D source or an estimate fails.
        """
        from noobfriend.core.specutils import background_noise

        estimates = {}
        for frame_id, frame in self.frames.items():
            source = frame.source_2d
            if source is None:
                raise ValueError(
                    f"frame {frame_id!r} has no 2-D source; build it with "
                    "NoobSpectrum.from_2d or use noise_from_continuum."
                )
            mask = (
                source_mask.get(frame_id)
                if isinstance(source_mask, Mapping)
                else source_mask
            )
            canonical_mask = None if mask is None else source.canonical(mask)
            estimates[frame_id] = _estimate(
                frame_id,
                lambda source=source, canonical_mask=canonical_mask: background_noise(
                    source.flux,
                    source.error,
                    window=source.collapse_window,
                    dispersion_axis=1,
                    source_mask=canonical_mask,
                    lags=lags,
                ),
            )
        return estimates

    def rest(self, frame_id: str | None = None) -> np.ndarray:
        """Return a frame's rest-frame wavelengths, ``obs / (1 + z)``.

        Parameters
        ----------
        frame_id
            The frame; may be omitted for a single-frame spectrum.

        Returns
        -------
        numpy.ndarray
            Read-only rest wavelengths in :attr:`unit`.
        """
        if self.z is None:
            raise ValueError("rest wavelengths need the spectrum's z.")
        if frame_id is None:
            if len(self.frames) != 1:
                raise ValueError("name the frame: this spectrum has several.")
            (frame,) = self.frames.values()
        else:
            frame = self._frame(frame_id)
        return readonly(frame.obs / (1.0 + self.z))

    def prepare(
        self,
        lines: Sequence[Any],
        *,
        continuum: Any,
        window: object = None,
    ) -> NoobFitWorkspace:
        """Resolve lines, continuum and noise into a fit workspace.

        Parameters
        ----------
        lines
            ``NoobLine`` declarations (may be empty); names must be unique.
        continuum
            A ``NoobContinuum`` declaration (required).
        window
            Observed-frame ``(lo, hi)`` range(s) in :attr:`unit` to fit,
            applied to every frame; the whole spectrum by default.

        Returns
        -------
        NoobFitWorkspace
            The fully resolved, backend-independent fit plan.
        """
        from noobfriend.inference.spectrum.workspace import prepare

        return prepare(self, lines, continuum=continuum, window=window)

    def __repr__(self) -> str:
        """Summarize frames, unit and redshift."""
        z = "None" if self.z is None else f"{self.z:g}"
        return f"NoobSpectrum(frames={list(self.frames)}, unit={self.unit!r}, z={z})"

    def _frame(self, frame_id: str) -> SpectrumFrame:
        try:
            return self.frames[frame_id]
        except KeyError:
            raise KeyError(
                f"unknown frame {frame_id!r}; frames are {list(self.frames)}."
            ) from None

    def _with_frames(self, frames: Mapping[str, SpectrumFrame]) -> NoobSpectrum:
        return _new(type(self), frames, self.z, self.unit)

    def _observed_intervals(
        self, obs: object, rest: object, obs_name: str, rest_name: str
    ) -> tuple[tuple[float, float], ...]:
        intervals = ()
        if obs is not None:
            intervals += parse_intervals(obs, name=obs_name)
        if rest is not None:
            if self.z is None:
                raise ValueError(f"{rest_name} ranges need the spectrum's z.")
            factor = 1.0 + self.z
            intervals += tuple(
                (lo * factor, hi * factor)
                for lo, hi in parse_intervals(rest, name=rest_name)
            )
        return intervals


def _estimate(frame_id: str, run: Any) -> Noise:
    try:
        scale, correlations = run()
        return estimated_noise(scale, correlations)
    except ValueError as error:
        raise ValueError(
            f"noise estimate failed for frame {frame_id!r}: {error}"
        ) from error


def _check_z(z: object) -> float | None:
    if z is None:
        return None
    if (
        not isinstance(z, Real)
        or isinstance(z, bool)
        or not isfinite(float(z))
        or z < 0
    ):
        raise ValueError(f"z must be a finite, non-negative number; got {z!r}.")
    return float(z)


def _init(
    spectrum: NoobSpectrum,
    frames: Mapping[str, SpectrumFrame],
    z: float | None,
    unit: WaveUnit,
) -> None:
    object.__setattr__(spectrum, "frames", MappingProxyType(dict(frames)))
    object.__setattr__(spectrum, "z", z)
    object.__setattr__(spectrum, "unit", unit)


def _new(
    cls: type[NoobSpectrum],
    frames: Mapping[str, SpectrumFrame],
    z: float | None,
    unit: WaveUnit,
) -> NoobSpectrum:
    spectrum = object.__new__(cls)
    _init(spectrum, frames, z, unit)
    return spectrum
