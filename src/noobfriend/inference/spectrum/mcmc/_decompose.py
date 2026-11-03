"""Posterior model curves per frame: total, continuum and line components.

This is the posterior of the library's *mean model* (no observation noise),
not a posterior predictive. Draws are taken at evenly spaced positions of
every chain, so the result is deterministic. Bands are pointwise ArviZ HDIs.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from noobfriend.inference.spectrum._xp import numpy_namespace, readonly

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from noobfriend.inference.spectrum.data import SpectrumFrame
    from noobfriend.inference.spectrum.mcmc._result import NoobMCMCResult
    from noobfriend.inference.spectrum.workspace import ModelView


@dataclass(frozen=True, slots=True)
class Band:
    """Pointwise posterior band of a curve.

    Attributes
    ----------
    median, lower, upper
        Median and HDI limits.
    """

    median: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True, slots=True, eq=False)
class FrameDecomposition:
    """One frame's decomposed posterior model.

    Attributes
    ----------
    frame
        The data (``obs``, ``flux``, ``error``, ``mask``, ``source_2d``).
    fitted
        Fitted-pixel mask on the native grid.
    grid
        Oversampled wavelengths over the fitted span.
    total, continuum
        Bands on ``grid``; ``total`` is always the observed model.
    components
        Each line's band on ``grid`` in the chosen view. With ``"emergent"``
        or ``"intrinsic"`` views they do not add up to ``total``.
    model
        Observed-model band on the native pixels.
    residual
        ``(flux - model.median) / sigma`` on fitted pixels (``NaN``
        elsewhere), with ``sigma`` the error times the median noise scale.
    chi2
        Whitened chi-square of ``model.median`` on the fitted pixels.
    """

    frame: SpectrumFrame
    fitted: np.ndarray
    grid: np.ndarray
    total: Band
    continuum: Band
    components: Mapping[str, Band]
    model: Band
    residual: np.ndarray
    chi2: float

    @property
    def n_fitted(self) -> int:
        """Number of fitted pixels."""
        return int(self.fitted.sum())


class NoobSpectrumDecomposition(Mapping[str, FrameDecomposition]):
    """Frame id to :class:`FrameDecomposition`.

    Attributes
    ----------
    view : str
        Which line components were computed (``"observed"``, ``"emergent"``,
        ``"intrinsic"``).
    hdi_prob : float
        Credible mass of the bands.
    unit : str
        The spectrum's wavelength unit.
    """

    def __init__(
        self,
        frames: Mapping[str, FrameDecomposition],
        *,
        view: ModelView,
        hdi_prob: float,
        unit: str,
    ) -> None:
        self._frames = dict(frames)
        self.view = view
        self.hdi_prob = hdi_prob
        self.unit = unit

    def __getitem__(self, frame_id: str) -> FrameDecomposition:
        """Return one frame."""
        return self._frames[frame_id]

    def __iter__(self) -> Iterator[str]:
        """Iterate over frame ids."""
        return iter(self._frames)

    def __len__(self) -> int:
        """Return the number of frames."""
        return len(self._frames)

    def plot(
        self,
        *,
        continuum: str = "include",
        normalize: bool = False,
        offset: float = 0.0,
        ylog: bool = False,
        xlim: tuple[float, float] | None = None,
    ) -> Figure:
        """Plot every frame: data against its own model, components, residual.

        Frames are stacked vertically on a shared wavelength range. For each
        frame, in order, the figure's axes are: the 2-D image (only when the
        frame has a 2-D source), the spectrum panel, the residual panel. The
        first panel of each frame is titled with the frame id and its
        chi-square. Masked pixels inside the range are drawn grey.

        Parameters
        ----------
        continuum
            ``"include"`` draws components on top of the continuum;
            ``"subtract"`` removes the continuum median from data, total and
            bands and draws components on their own.
        normalize
            Divide each frame's displayed values by the peak (largest
            absolute value) of its displayed total median.
        offset
            Added after normalizing.
        ylog
            Logarithmic flux axis; values ``<= 0`` are not drawn.
        xlim
            Wavelength range; the span of the fitted pixels by default.

        Returns
        -------
        matplotlib.figure.Figure
            Restyle it through its axes (colours, sizes and titles are not
            parameters). The residual and 2-D panels ignore the display
            options.
        """
        from noobfriend.inference.spectrum.mcmc._plot import plot_decomposition

        return plot_decomposition(
            self,
            continuum=continuum,
            normalize=normalize,
            offset=offset,
            ylog=ylog,
            xlim=xlim,
        )


def select_draws(chains: int, draws: int, wanted: int) -> tuple[np.ndarray, np.ndarray]:
    """Evenly spaced ``(chain, draw)`` indices, the same from every chain.

    Parameters
    ----------
    chains, draws
        Posterior shape.
    wanted
        Number of draws to keep (all when it exceeds the posterior).

    Returns
    -------
    chain_index, draw_index : numpy.ndarray
        Flat index arrays.
    """
    per_chain = min(draws, max(1, -(-wanted // chains)))
    positions = np.unique(np.linspace(0, draws - 1, per_chain).round().astype(int))
    chain_index = np.repeat(np.arange(chains), positions.size)
    draw_index = np.tile(positions, chains)
    return chain_index[:wanted], draw_index[:wanted]


def decompose(
    result: NoobMCMCResult,
    *,
    view: ModelView = "observed",
    frames: Sequence[str] | None = None,
    hdi_prob: float = 0.94,
    draws: int = 1000,
    oversample: int = 8,
) -> NoobSpectrumDecomposition:
    """Compute :class:`NoobSpectrumDecomposition` (see ``result.decompose``)."""
    workspace = result.workspace
    frame_ids = tuple(workspace.frame_ids if frames is None else frames)
    for frame_id in frame_ids:
        if frame_id not in workspace.fitted:
            raise KeyError(f"unknown frame {frame_id!r}.")
    if oversample < 1 or draws < 1:
        raise ValueError("draws and oversample must be positive.")
    values = posterior_values(result, draws)
    out = {}
    for frame_id in frame_ids:
        out[frame_id] = _decompose_frame(
            result, values, frame_id, view, hdi_prob, oversample
        )
    return NoobSpectrumDecomposition(
        out, view=view, hdi_prob=hdi_prob, unit=workspace.spectrum.unit
    )


def posterior_values(result: NoobMCMCResult, draws: int) -> dict[str, np.ndarray]:
    """Return selected posterior draws of every physical name, shaped ``(S, 1)``."""
    posterior = result.idata["posterior"].dataset
    chains, total = posterior.sizes["chain"], posterior.sizes["draw"]
    chain_index, draw_index = select_draws(chains, total, draws)
    return {
        name: np.asarray(array.transpose("chain", "draw").values)[
            chain_index, draw_index
        ][:, None]
        for name, array in posterior.data_vars.items()
        if set(array.dims) == {"chain", "draw"}
    }


def median_noise(
    values: Mapping[str, np.ndarray], frame_id: str
) -> tuple[float, float | None]:
    """Posterior-median noise scale and AR(1) rho of one frame."""
    scale = float(np.median(values[f"frame.{frame_id}.noise.scale"]))
    rho_name = f"frame.{frame_id}.noise.rho"
    rho = float(np.median(values[rho_name])) if rho_name in values else None
    return scale, rho


def _decompose_frame(
    result: NoobMCMCResult,
    values: Mapping[str, np.ndarray],
    frame_id: str,
    view: ModelView,
    hdi_prob: float,
    oversample: int,
) -> FrameDecomposition:
    xp = numpy_namespace()
    workspace = result.workspace
    frame = workspace.spectrum.frames[frame_id]
    fitted = workspace.fitted[frame_id]
    obs = np.asarray(frame.obs)
    fitted_obs = obs[fitted]
    span = (obs >= fitted_obs[0]) & (obs <= fitted_obs[-1])
    grid = np.linspace(fitted_obs[0], fitted_obs[-1], oversample * int(span.sum()))
    on_grid = workspace.frame_model(xp, values, frame_id, grid, view=view)
    native = workspace.frame_model(xp, values, frame_id, obs).total
    model = _band(native, hdi_prob)
    scale, rho = median_noise(values, frame_id)
    flux = np.asarray(frame.flux)
    residual = np.full(obs.size, np.nan)
    residual[fitted] = (flux[fitted] - model.median[fitted]) / (
        np.asarray(frame.error)[fitted] * scale
    )
    whitened = workspace.frame_noise[frame_id].whiten(
        xp, flux[fitted] - model.median[fitted], scale, rho
    )
    return FrameDecomposition(
        frame=frame,
        fitted=fitted,
        grid=readonly(grid),
        total=_band(on_grid.total, hdi_prob),
        continuum=_band(on_grid.continuum + 0.0 * on_grid.total, hdi_prob),
        components={
            name: _band(curve, hdi_prob) for name, curve in on_grid.components.items()
        },
        model=model,
        residual=readonly(residual),
        chi2=float(np.sum(np.asarray(whitened) ** 2)),
    )


def _band(curves: Any, prob: float) -> Band:
    """Median and pointwise ArviZ HDI over the leading draw axis."""
    import arviz as az
    import xarray as xr

    curves = np.asarray(curves, dtype=float)
    if curves.ndim == 1:
        curves = curves[None, :]
    data = xr.DataArray(curves[None], dims=("chain", "draw", "x"))
    interval = az.hdi(data, prob=prob).transpose("x", "ci_bound").values
    return Band(
        median=readonly(np.median(curves, axis=0)),
        lower=readonly(interval[:, 0]),
        upper=readonly(interval[:, 1]),
    )
