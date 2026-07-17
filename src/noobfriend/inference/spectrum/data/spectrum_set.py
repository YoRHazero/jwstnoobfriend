"""Container for several same-source spectra fitted under one joint model."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from math import isclose
from typing import TYPE_CHECKING

from noobfriend.inference.spectrum.data.spectrum import NoobSpectrum

if TYPE_CHECKING:
    from noobfriend.inference.spectrum.line import NoobLine
    from noobfriend.inference.spectrum.types import WaveUnit
    from noobfriend.inference.spectrum.workspace import NoobFitWorkspace
    from noobfriend.inference.spectrum.workspace.continuum import ContinuumSharing


@dataclass(frozen=True, slots=True, eq=False, init=False)
class NoobSpectrumSet:
    """Several exposures of one source prepared for a single joint fit.

    A set holds two or more :class:`NoobSpectrum` frames that share a source,
    a redshift, and a wavelength unit. Line parameters are shared across the
    frames while continuum, spectral resolution, and other instrumental
    nuisances stay per-frame. Frames keep their native wavelength grids; they
    are never resampled onto a common grid.

    Parameters
    ----------
    spectra
        The frames to join. A sequence assigns positional ids
        ``"frame_0"``, ``"frame_1"``, .... A mapping uses its keys as frame
        ids, which is the natural place for exposure names.

    Raises
    ------
    TypeError
        If ``spectra`` is neither a sequence nor a mapping, or a frame is not
        a :class:`NoobSpectrum`.
    ValueError
        If no frame is given, frame ids are not unique, or the frames disagree
        on wavelength unit or redshift.
    """

    frames: tuple[NoobSpectrum, ...]
    frame_ids: tuple[str, ...]

    def __init__(
        self, spectra: Sequence[NoobSpectrum] | Mapping[str, NoobSpectrum]
    ) -> None:
        """Normalize frame entries and validate joint-fit consistency."""
        frames, frame_ids = _normalize_frame_entries(spectra)
        _validate_frames(frames, frame_ids)
        object.__setattr__(self, "frames", frames)
        object.__setattr__(self, "frame_ids", frame_ids)

    @property
    def n_frames(self) -> int:
        """Number of frames in the set."""
        return len(self.frames)

    @property
    def unit(self) -> WaveUnit:
        """Shared wavelength unit of every frame."""
        return self.frames[0].unit

    @property
    def z(self) -> float | None:
        """Shared source redshift, or ``None`` when unset on every frame."""
        return self.frames[0].z

    @property
    def resolving_powers(self) -> tuple[float | None, ...]:
        """Per-frame spectral resolving power in frame order."""
        return tuple(frame.resolving_power for frame in self.frames)

    def prepare(
        self,
        lines: Sequence[NoobLine] | Mapping[str, NoobLine],
        *,
        continuum_order: int = 1,
        continuum_lambda_0: float | None = None,
        continuum_sharing: ContinuumSharing = "pooled",
    ) -> NoobFitWorkspace:
        """Resolve a line combination jointly against every frame.

        Parameters
        ----------
        lines
            Line declarations, identical in meaning to
            :meth:`NoobSpectrum.prepare`.
        continuum_order
            Polynomial order of each frame's continuum.
        continuum_lambda_0
            Continuum anchor wavelength; defaults to the first line's center.
        continuum_sharing
            How continuum coefficients are shared across frames. ``"pooled"``
            uses non-centered partial pooling, ``"independent"`` fits each
            frame freely, and ``"shared"`` ties all frames to one continuum.
        """
        from noobfriend.inference.spectrum.workspace import NoobFitWorkspace

        return NoobFitWorkspace.build(
            self,
            lines,
            continuum_order=continuum_order,
            continuum_lambda_0=continuum_lambda_0,
            continuum_sharing=continuum_sharing,
        )

    def __len__(self) -> int:
        """Return the number of frames."""
        return len(self.frames)

    def __iter__(self) -> Iterator[NoobSpectrum]:
        """Iterate over frames in frame order."""
        return iter(self.frames)

    def __repr__(self) -> str:
        """Concise multi-frame summary."""
        z = "None" if self.z is None else f"{self.z:g}"
        return (
            f"NoobSpectrumSet(n_frames={self.n_frames}, unit={self.unit!r}, "
            f"z={z}, frame_ids={self.frame_ids!r})"
        )


def _normalize_frame_entries(
    spectra: Sequence[NoobSpectrum] | Mapping[str, NoobSpectrum],
) -> tuple[tuple[NoobSpectrum, ...], tuple[str, ...]]:
    if isinstance(spectra, Mapping):
        frames: list[NoobSpectrum] = []
        frame_ids: list[str] = []
        for key, frame in spectra.items():
            if not isinstance(key, str) or not key:
                raise ValueError("manual frame ids must be non-empty strings.")
            frame_ids.append(key)
            frames.append(frame)
        if not frames:
            raise ValueError("NoobSpectrumSet requires at least one frame.")
        return tuple(frames), tuple(frame_ids)

    if isinstance(spectra, (str, bytes)) or not isinstance(spectra, Sequence):
        raise TypeError(
            "NoobSpectrumSet expects a sequence of NoobSpectrum "
            "or a mapping of id to NoobSpectrum."
        )
    frames = tuple(spectra)
    if not frames:
        raise ValueError("NoobSpectrumSet requires at least one frame.")
    frame_ids = tuple(f"frame_{index}" for index in range(len(frames)))
    return frames, frame_ids


def _validate_frames(
    frames: tuple[NoobSpectrum, ...], frame_ids: tuple[str, ...]
) -> None:
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("frame ids must be unique.")
    for frame in frames:
        if not isinstance(frame, NoobSpectrum):
            raise TypeError("NoobSpectrumSet entries must be NoobSpectrum instances.")

    units = {frame.unit for frame in frames}
    if len(units) > 1:
        raise ValueError(
            f"all frames must share one wavelength unit; got {sorted(units)}."
        )

    reference_z = frames[0].z
    for frame in frames[1:]:
        if (reference_z is None) != (frame.z is None):
            raise ValueError(
                "all frames must share the same redshift (either all set or all unset)."
            )
        if (
            reference_z is not None
            and frame.z is not None
            and not isclose(reference_z, frame.z, rel_tol=1e-10, abs_tol=1e-12)
        ):
            raise ValueError("all frames must share one source redshift.")
