"""NoobFitWorkspace: the resolved, backend-independent fit plan."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from noobfriend.inference.spectrum._wavelength import Interval, parse_intervals
from noobfriend.inference.spectrum.noise import FrameNoise
from noobfriend.inference.spectrum.workspace._fingerprint import (
    data_fingerprint,
    frame_digest,
    model_fingerprint,
)
from noobfriend.inference.spectrum.workspace._handles import ContinuumHandle, LineHandle
from noobfriend.inference.spectrum.workspace._model import (
    FrameModel,
    ModelView,
    frame_model,
    physical,
)
from noobfriend.inference.spectrum.workspace._protocols import SpectrumLike
from noobfriend.inference.spectrum.workspace._resolve import DataScales, resolve
from noobfriend.inference.spectrum.workspace._summary import workspace_summary
from noobfriend.inference.spectrum.workspace._table import JointBlock, ParameterTable

if TYPE_CHECKING:
    from pathlib import Path

    from noobfriend.inference.spectrum.data import NoobSpectrum
    from noobfriend.inference.spectrum.mcmc import NoobMCMCResult, NoobSpectrumModel
    from noobfriend.inference.spectrum.mle import NoobMLEResult


@dataclass(frozen=True, eq=False)
class NoobFitWorkspace:
    """A fully resolved fit plan, independent of the fit backend.

    Built by ``NoobSpectrum.prepare``. Immutable: to change the plan, change
    the declarations and prepare again (it is cheap).

    Attributes
    ----------
    spectrum : NoobSpectrum
        The data, unchanged.
    window : tuple of (float, float) or None
        Fit window (observed frame, data unit), ``None`` for everything.
    fitted : Mapping[str, numpy.ndarray]
        Each frame's fitted-pixel mask (usable and inside the window).
    lines : tuple of LineHandle
        Lines in prepare order.
    continuum : ContinuumHandle
        The continuum.
    parameters : ParameterTable
        The single parameter table: every row with its prior or value, plus
        the physical line quantities.
    joint_priors : tuple of JointBlock
        Joint priors gathered from ``as_prior`` declarations.
    frame_noise : Mapping[str, FrameNoise]
        Each frame's noise structure on its fitted pixels.
    scales : DataScales
        Data scales behind the data-derived default priors.
    data_fingerprint, model_fingerprint : str
        Digests checked when a saved fit is loaded.
    frame_digests : Mapping[str, str]
        Each frame's array digest (identifies the frame for joint priors).
    """

    spectrum: NoobSpectrum
    window: tuple[Interval, ...] | None
    fitted: Mapping[str, np.ndarray]
    lines: tuple[LineHandle, ...]
    continuum: ContinuumHandle
    parameters: ParameterTable
    joint_priors: tuple[JointBlock, ...]
    frame_noise: Mapping[str, FrameNoise]
    scales: DataScales
    data_fingerprint: str
    model_fingerprint: str
    frame_digests: Mapping[str, str]

    def __getitem__(self, name: str) -> LineHandle | ContinuumHandle:
        """Return a component handle by name (``"continuum"`` for the continuum)."""
        if name == "continuum":
            return self.continuum
        for line in self.lines:
            if line.name == name:
                return line
        names = [line.name for line in self.lines] + ["continuum"]
        raise KeyError(f"no component {name!r}; components are {names}.")

    def __iter__(self) -> Iterator[LineHandle | ContinuumHandle]:
        """Iterate over components in prepare order, continuum last."""
        yield from self.lines
        yield self.continuum

    def __len__(self) -> int:
        """Return the number of components."""
        return len(self.lines) + 1

    @property
    def frame_ids(self) -> tuple[str, ...]:
        """Frame ids in order."""
        return tuple(self.spectrum.frames)

    def summary(self) -> str:
        """Return the complete fit plan as plain text."""
        return workspace_summary(self)

    def physical(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """Return every parameter and line quantity from free-parameter values.

        Parameters
        ----------
        values
            Values of all free and pooled rows by flat name (NumPy or JAX,
            optionally with a leading draw axis).

        Returns
        -------
        dict
            Values by flat name, fixed rows included.
        """
        return physical(self.parameters, values)

    def frame_model(
        self,
        xp: Any,
        values: Mapping[str, Any],
        frame_id: str,
        wavelength: Any | None = None,
        *,
        view: ModelView = "observed",
    ) -> FrameModel:
        """Evaluate one frame's mean model.

        Parameters
        ----------
        xp
            Array namespace (``_xp.numpy_namespace()`` or ``jax_namespace()``).
        values
            Output of :meth:`physical`.
        frame_id
            The frame.
        wavelength
            Where to evaluate; the frame's fitted pixels by default.
        view
            Line components as ``"observed"``, ``"emergent"`` or
            ``"intrinsic"``; the total is always observed.

        Returns
        -------
        FrameModel
            Continuum, components and total.
        """
        if wavelength is None:
            frame = self.spectrum.frames[frame_id]
            wavelength = frame.obs[self.fitted[frame_id]]
        return frame_model(
            xp, self.lines, self.continuum, values, frame_id, wavelength, view=view
        )

    def model(self) -> NoobSpectrumModel:
        """Build the NumPyro sampling model (see ``NoobSpectrumModel``).

        Returns
        -------
        NoobSpectrumModel
            Call ``.sample()`` on it, or take ``.numpyro_model`` to customize.
        """
        from noobfriend.inference.spectrum.mcmc import NoobSpectrumModel

        return NoobSpectrumModel(self)

    def mle(
        self,
        absorption_bound_multipliers: Sequence[float] = (0.5, 1.0, 2.0, 5.0),
        cancellation_threshold: float | None = 0.37,
        relative_likelihood_min: float = 0.5,
        random_seed: int = 1729,
    ) -> NoobMLEResult:
        """Quick maximum-likelihood fit of a single-frame workspace.

        Reading rules: the frame's raw error with white noise (the ``Noise``
        model is ignored); joint priors are ignored; every other prior only
        bounds its parameter (an omitted ``Uniform()`` / ``LogUniform()``
        bound leaves it unbounded); fixed values and bindings apply. Multi-start
        bounded least squares, then among solutions within the relative
        likelihood region the one with the least emission/absorption
        cancellation, refined further when it still cancels more than
        ``cancellation_threshold``.

        Parameters
        ----------
        absorption_bound_multipliers
            Internal upper bounds tried for free absorption fluxes, in units
            of the flux that reaches the data range at the widest width.
        cancellation_threshold
            Cancellation fraction above which the refinement runs; ``None``
            disables it.
        relative_likelihood_min
            Smallest relative likelihood a solution may have to be selected.
        random_seed
            Seed of the random starts.

        Returns
        -------
        NoobMLEResult
            Selected solution, likelihood optimum, Jacobian error bars.

        Raises
        ------
        ValueError
            If the workspace has more than one frame (select one with
            ``spec.select(frame_id)``).
        """
        from noobfriend.inference.spectrum.mle import fit

        return fit(
            self,
            absorption_bound_multipliers=absorption_bound_multipliers,
            cancellation_threshold=cancellation_threshold,
            relative_likelihood_min=relative_likelihood_min,
            random_seed=random_seed,
        )

    def load(self, path: str | Path) -> NoobMCMCResult:
        """Reattach a saved MCMC result to this workspace.

        Rebuild the workspace with the script that produced the fit (prepare
        is cheap), then load. Data and model fingerprints are checked
        separately; a mismatch names which part changed.

        Parameters
        ----------
        path
            File written by ``NoobMCMCResult.save``.

        Returns
        -------
        NoobMCMCResult
            The result with every method available.
        """
        from noobfriend.inference.spectrum.mcmc import load

        return load(self, path)

    def __repr__(self) -> str:
        """Summarize the workspace in one line."""
        return (
            f"NoobFitWorkspace(frames={list(self.frame_ids)}, "
            f"lines={[line.name for line in self.lines]}, "
            f"free parameters={len(self.parameters.free())})"
        )


def prepare(
    spectrum: SpectrumLike,
    lines: Sequence[Any],
    *,
    continuum: Any,
    window: object = None,
) -> NoobFitWorkspace:
    """Resolve declarations into a :class:`NoobFitWorkspace`.

    Parameters
    ----------
    spectrum
        The data.
    lines
        Line declarations (a sequence; may be empty). Names must be unique
        and every derived line's base must be included.
    continuum
        The continuum declaration (required).
    window
        Observed-frame ``(lo, hi)`` range(s) in the data unit, applied to
        every frame. Every line's nominal observed wavelength must fall
        inside one of them.

    Returns
    -------
    NoobFitWorkspace
        The fit plan.

    Raises
    ------
    TypeError
        If an argument has the wrong type.
    ValueError
        If the declarations do not fit the data (duplicate names, a missing
        base, a line outside the window, a joint prior learned from the same
        pixels, ...).
    """
    if not isinstance(spectrum, SpectrumLike):
        raise TypeError("prepare needs a NoobSpectrum.")
    if isinstance(lines, (str, bytes, Mapping)) or not isinstance(lines, Sequence):
        raise TypeError("lines must be a sequence of NoobLine (a list or tuple).")
    intervals = None if window is None else parse_intervals(window, name="window")
    resolution = resolve(spectrum, lines, continuum, intervals)
    fitted = resolution.fitted
    frame_noise = {}
    for frame_id, frame in spectrum.frames.items():
        mask = fitted[frame_id]
        frame_noise[frame_id] = FrameNoise.build(
            frame.noise, np.asarray(frame.error)[mask], np.flatnonzero(mask)
        )
    _check_prior_reuse(resolution.joint_blocks, spectrum, fitted)
    return NoobFitWorkspace(
        spectrum=spectrum,  # type: ignore[arg-type]
        window=intervals,
        fitted=fitted,
        lines=resolution.lines,
        continuum=resolution.continuum,
        parameters=resolution.table,
        joint_priors=resolution.joint_blocks,
        frame_noise=MappingProxyType(frame_noise),
        scales=resolution.scales,
        data_fingerprint=data_fingerprint(spectrum, fitted, intervals),
        model_fingerprint=model_fingerprint(
            resolution.table,
            resolution.lines,
            resolution.continuum,
            resolution.joint_blocks,
        ),
        frame_digests=MappingProxyType(
            {
                frame_id: frame_digest(frame)
                for frame_id, frame in spectrum.frames.items()
            }
        ),
    )


def _check_prior_reuse(
    blocks: tuple[JointBlock, ...],
    spectrum: SpectrumLike,
    fitted: Mapping[str, np.ndarray],
) -> None:
    """Refuse a joint prior learned from pixels this fit uses again (T7.6)."""
    for block in blocks:
        for source in block.joint.sources:
            for frame_id, frame in spectrum.frames.items():
                if frame_digest(frame) != source.digest:
                    continue
                if np.any(fitted[frame_id] & source.fitted):
                    raise ValueError(
                        f"a joint prior was learned from frame {frame_id!r} and this fit "
                        "uses some of the same pixels; that would count the data twice."
                    )
