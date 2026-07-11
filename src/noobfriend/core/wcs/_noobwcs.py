"""``NoobWCS``: a gwcs pipeline compiled onto the noobase evaluator.

Why this exists: gwcs coordinate evaluation pays the astropy ``Model``
call machinery on every invocation -- ~1 ms per scalar call and an
interpreted model tree even for full-frame grids. ``NoobWCS`` compiles each
pipeline stage once (see :mod:`._compile`) into flat op-list programs run by
:class:`noobase.image.wcs.WcsProgram`: scalar calls drop to microseconds
and full-grid evaluation is multithreaded Rust.

The surface mirrors the subset of gwcs that noobfriend actually consumes
(the :class:`~noobfriend.core.wcs.TransformWCS` protocol):
``available_frames`` plus ``get_transform(from_frame, to_frame)`` returning
a callable of plain arrays / floats. Bounding boxes, units, and coordinate
objects are deliberately absent.

Programs are JSON-able: ``to_spec`` / ``from_spec`` round-trip a compiled
``NoobWCS`` without touching gwcs (or the ASDF it came from), which makes
per-file spec caching possible later.
"""

from typing import Any, Callable

import numpy as np
from astropy.wcs import WCS as AstropyWCS
from gwcs import WCS as GwcsWCS
from noobase.image.wcs import WcsProgram

from ._compile import Spec, compile_fits_tan, compile_transform, concat_specs
from ._correction import TangentCorrection

#: One pipeline stage: forward / backward program specs (backward is None
#: when the source transform defines no analytic inverse).
_Stage = dict[str, Any]

_SPEC_VERSION = 1


def _normalized_call(program: WcsProgram) -> Callable[..., Any]:
    """Wrap a compiled program to accept gwcs-style call forms.

    :class:`noobase.image.wcs.WcsProgram` only takes same-shape ``float64``
    arrays or all-scalar inputs, while gwcs transforms additionally allow
    mixed scalar/array calls, broadcasting, integer dtypes, and lists. The
    wrapper closes that gap so a compiled transform is a drop-in for its
    gwcs counterpart; the normalisation is a no-op (and costs ~1 us) when
    the inputs are already in program form.
    """

    def transform(*values: Any) -> Any:
        if any(np.ndim(value) for value in values):
            arrays = np.broadcast_arrays(
                *(np.asarray(value, dtype=np.float64) for value in values)
            )
            return program(*(np.ascontiguousarray(array) for array in arrays))
        return program(*(float(value) for value in values))

    return transform


class NoobWCS:
    """A compiled multi-frame WCS.

    Construct via :func:`from_gwcs`, :func:`from_fits_wcs`, or
    :meth:`from_spec`; the constructor takes already-compiled stages.

    Parameters
    ----------
    frames : sequence of str
        Coordinate-frame names in pipeline order (e.g. ``["detector",
        "v2v3", "v2v3vacorr", "world"]``).
    stages : sequence of dict
        One ``{"forward": spec, "backward": spec | None}`` per adjacent
        frame pair, in the same order.
    """

    def __init__(self, frames: tuple[str, ...], stages: tuple[_Stage, ...]) -> None:
        if len(stages) != len(frames) - 1:
            raise ValueError(
                f"{len(frames)} frames require {len(frames) - 1} stages, "
                f"got {len(stages)}."
            )
        self._frames = frames
        self._stages = stages
        self._programs: dict[tuple[str, str], Callable[..., Any]] = {}

    @property
    def available_frames(self) -> list[str]:
        """Frame names in pipeline order (gwcs-compatible)."""
        return list(self._frames)

    def get_transform(self, from_frame: str, to_frame: str) -> Callable[..., Any]:
        """Return the compiled transform between two frames.

        Parameters
        ----------
        from_frame, to_frame : str
            Frame names from :attr:`available_frames`; either direction.

        Returns
        -------
        Callable
            A callable evaluating the chained stage programs. Accepts the
            same call forms as a gwcs transform -- scalars (returns floats)
            or broadcastable arrays of any numeric dtype (returns
            ``float64`` arrays).

        Raises
        ------
        ValueError
            If a frame name is unknown, the frames are equal, or a
            backward leg is needed where the source transform had no
            analytic inverse.
        """
        key = (from_frame, to_frame)
        cached = self._programs.get(key)
        if cached is not None:
            return cached

        try:
            start, stop = self._frames.index(from_frame), self._frames.index(to_frame)
        except ValueError:
            raise ValueError(
                f"Unknown frame in {key!r}; available: {self._frames}."
            ) from None
        if start == stop:
            raise ValueError(f"from_frame and to_frame are both {from_frame!r}.")

        if start < stop:
            specs = [self._stages[i]["forward"] for i in range(start, stop)]
        else:
            specs = [self._stages[i - 1]["backward"] for i in range(start, stop, -1)]
            if any(spec is None for spec in specs):
                raise ValueError(
                    f"No inverse available along {from_frame!r} -> {to_frame!r}."
                )
        transform = _normalized_call(WcsProgram(concat_specs(specs)))
        self._programs[key] = transform
        return transform

    def __call__(self, *values: Any) -> Any:
        """Evaluate the full forward pipeline (first frame -> last), like gwcs.

        Unlike :meth:`gwcs.WCS.__call__` no bounding box is applied --
        out-of-array inputs extrapolate instead of returning NaN.
        """
        return self.get_transform(self._frames[0], self._frames[-1])(*values)

    def __getstate__(self) -> dict[str, Any]:
        """Pickle the specs only -- built programs hold Rust objects."""
        return {"frames": self._frames, "stages": self._stages}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._frames = state["frames"]
        self._stages = state["stages"]
        self._programs = {}

    def with_tangent_correction(self, correction: "TangentCorrection") -> "NoobWCS":
        """Return a copy with an alignment correction composed in.

        The correction (see
        :class:`noobfriend.core.wcs.TangentCorrection`) is composed into
        the final pipeline stage on the world side -- frame names are
        untouched, and both directions stay exact (the backward leg uses
        the analytically inverted affine). Extra world coordinates beyond
        ``(ra, dec)`` (WFSS wavelength / order) pass through unchanged.

        Parameters
        ----------
        correction : TangentCorrection
            The tangent-plane affine correction, e.g. from the stage-3
            alignment fit.

        Returns
        -------
        NoobWCS
            A new instance; ``self`` is not modified.
        """
        forward_corr, backward_corr = correction.to_specs()
        last = self._stages[-1]
        n_extra = len(last["forward"]["outputs"]) - 2
        if n_extra > 0:
            forward_corr = _pad_passthrough(forward_corr, n_extra)
            backward_corr = _pad_passthrough(backward_corr, n_extra)
        stage: _Stage = {
            "forward": concat_specs([last["forward"], forward_corr]),
            "backward": None
            if last["backward"] is None
            else concat_specs([backward_corr, last["backward"]]),
        }
        return NoobWCS(self._frames, (*self._stages[:-1], stage))

    def to_spec(self) -> dict[str, Any]:
        """JSON-able representation (see :meth:`from_spec`)."""
        return {
            "version": _SPEC_VERSION,
            "frames": list(self._frames),
            "stages": [dict(stage) for stage in self._stages],
        }

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> "NoobWCS":
        """Rebuild from :meth:`to_spec` output without any gwcs object."""
        version = spec.get("version")
        if version != _SPEC_VERSION:
            raise ValueError(f"Unsupported NoobWCS spec version: {version!r}.")
        return cls(tuple(spec["frames"]), tuple(spec["stages"]))

    def __repr__(self) -> str:
        return f"NoobWCS(frames={list(self._frames)!r})"


def _pad_passthrough(spec: Spec, n_extra: int) -> Spec:
    """Widen a program with pass-through inputs/outputs (pure wiring)."""
    padded = dict(spec)
    extra = list(range(spec["n_regs"], spec["n_regs"] + n_extra))
    padded["n_regs"] = spec["n_regs"] + n_extra
    padded["inputs"] = [*spec["inputs"], *extra]
    padded["outputs"] = [*spec["outputs"], *extra]
    return padded


def from_gwcs(wcs: GwcsWCS) -> NoobWCS:
    """Compile a JWST gwcs object into a :class:`NoobWCS`.

    Parameters
    ----------
    wcs : gwcs.WCS
        A gwcs whose pipeline transforms are built from the supported
        model set: JWST CLEAR imaging chains, NIRCam WFSS grism chains,
        and NIRSpec IFU chains (grating modes; PRISM's ``Snell`` model is
        not compiled yet). See :mod:`._compile`.

    Returns
    -------
    NoobWCS
        The compiled WCS. Frame names and transform semantics (including
        the WFSS extra ``wavelength`` / ``order`` coordinates) match the
        source gwcs.

    Raises
    ------
    noobfriend.core.wcs.UnsupportedTransformError
        If a pipeline stage contains a model the compiler cannot express.
    """
    frames: list[str] = []
    stages: list[_Stage] = []
    for step in wcs.pipeline:
        frames.append(getattr(step.frame, "name", str(step.frame)))
        if step.transform is None:  # the terminal frame
            continue
        forward = compile_transform(step.transform)
        try:
            inverse = step.transform.inverse
        except NotImplementedError:
            backward = None
        else:
            backward = compile_transform(inverse)
        stages.append({"forward": forward, "backward": backward})
    return NoobWCS(tuple(frames), tuple(stages))


def from_fits_wcs(wcs: AstropyWCS) -> NoobWCS:
    """Compile a plain FITS TAN header WCS into a :class:`NoobWCS`.

    Covers mosaic-tile grids and synthesised cutout headers. The two
    frames are named ``detector`` and ``world`` and both directions are
    exact, so ``get_transform("world", "detector")`` replaces
    ``astropy.wcs.WCS.world_to_pixel_values`` (0-based convention).

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        A 2-axis ``RA---TAN`` / ``DEC--TAN`` WCS without distortion terms.

    Returns
    -------
    NoobWCS
        The compiled two-frame WCS.
    """
    forward, backward = compile_fits_tan(wcs)
    return NoobWCS(("detector", "world"), ({"forward": forward, "backward": backward},))
