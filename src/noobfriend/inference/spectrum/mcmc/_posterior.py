"""Posterior access that mirrors the declarations.

``result["Ha_broad"].fwhm``, ``result["Ha_broad"].laplace.fwhm``,
``result["continuum"].coefficient(1)`` and ``result.frames["v1"].noise.rho``
all return a :class:`ParameterPosterior`: the draws of one parameter with
their summary statistics. Posteriors combine with ``+ - * / **`` draw by
draw, and ``map`` applies any function.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from noobfriend.inference.spectrum._xp import readonly

if TYPE_CHECKING:
    from noobfriend.inference.spectrum.workspace import LineHandle, NoobFitWorkspace

#: Default credible mass of highest-density intervals.
HDI_PROB = 0.94


class ParameterPosterior:
    """Posterior draws of one (possibly derived) quantity.

    Parameters
    ----------
    name
        Flat name or an expression describing the quantity.
    samples
        Draws, shape ``(chain, draw)``.
    """

    __slots__ = ("name", "samples")

    def __init__(self, name: str, samples: np.ndarray) -> None:
        samples = readonly(samples)
        if samples.ndim != 2:
            raise ValueError(
                f"samples must be (chain, draw); got shape {samples.shape}."
            )
        self.name = name
        self.samples = samples

    @property
    def median(self) -> float:
        """Posterior median."""
        return float(np.median(self.samples))

    @property
    def mean(self) -> float:
        """Posterior mean."""
        return float(np.mean(self.samples))

    @property
    def std(self) -> float:
        """Posterior standard deviation."""
        return float(np.std(self.samples))

    def hdi(self, prob: float = HDI_PROB) -> tuple[float, float]:
        """Highest-density interval (ArviZ) holding ``prob`` of the draws."""
        import arviz as az

        lower, upper = np.asarray(
            az.hdi(self.samples.reshape(-1), prob=prob), dtype=float
        )
        return float(lower), float(upper)

    def map(
        self, function: Callable[[np.ndarray], np.ndarray], name: str | None = None
    ) -> ParameterPosterior:
        """Apply ``function`` to the draws (vectorized) and wrap the result.

        Parameters
        ----------
        function
            Elementwise function of the ``(chain, draw)`` array.
        name
            Name of the result; ``"f(<name>)"`` by default.

        Returns
        -------
        ParameterPosterior
            The transformed draws.
        """
        values = np.asarray(function(np.asarray(self.samples)), dtype=float)
        return ParameterPosterior(name or f"f({self.name})", values)

    def __repr__(self) -> str:
        """Show the median and HDI."""
        lower, upper = self.hdi()
        return (
            f"ParameterPosterior({self.name!r}, median={self.median:.6g}, "
            f"{HDI_PROB:.0%} HDI=[{lower:.6g}, {upper:.6g}])"
        )

    def _combine(
        self,
        other: object,
        symbol: str,
        operation: Callable[[Any, Any], Any],
        swap: bool = False,
    ) -> ParameterPosterior:
        if isinstance(other, ParameterPosterior):
            if other.samples.shape != self.samples.shape:
                raise ValueError(
                    f"cannot combine {self.name!r} {self.samples.shape} with "
                    f"{other.name!r} {other.samples.shape}: draws do not align."
                )
            right, right_name = other.samples, other.name
        elif isinstance(other, (int, float, np.integer, np.floating)):
            right, right_name = float(other), f"{float(other):g}"
        else:
            return NotImplemented  # type: ignore[return-value]
        left, left_name = self.samples, self.name
        if swap:
            left, right = right, left
            left_name, right_name = right_name, left_name
        return ParameterPosterior(
            f"({left_name} {symbol} {right_name})", operation(left, right)
        )

    def __add__(self, other: object) -> ParameterPosterior:
        """Add draw by draw."""
        return self._combine(other, "+", np.add)

    def __radd__(self, other: object) -> ParameterPosterior:
        """Add draw by draw."""
        return self._combine(other, "+", np.add, swap=True)

    def __sub__(self, other: object) -> ParameterPosterior:
        """Subtract draw by draw."""
        return self._combine(other, "-", np.subtract)

    def __rsub__(self, other: object) -> ParameterPosterior:
        """Subtract draw by draw."""
        return self._combine(other, "-", np.subtract, swap=True)

    def __mul__(self, other: object) -> ParameterPosterior:
        """Multiply draw by draw."""
        return self._combine(other, "*", np.multiply)

    def __rmul__(self, other: object) -> ParameterPosterior:
        """Multiply draw by draw."""
        return self._combine(other, "*", np.multiply, swap=True)

    def __truediv__(self, other: object) -> ParameterPosterior:
        """Divide draw by draw."""
        return self._combine(other, "/", np.divide)

    def __rtruediv__(self, other: object) -> ParameterPosterior:
        """Divide draw by draw."""
        return self._combine(other, "/", np.divide, swap=True)

    def __pow__(self, other: object) -> ParameterPosterior:
        """Raise to a power draw by draw."""
        return self._combine(other, "**", np.power)

    def __rpow__(self, other: object) -> ParameterPosterior:
        """Raise to a power draw by draw."""
        return self._combine(other, "**", np.power, swap=True)

    def __neg__(self) -> ParameterPosterior:
        """Negate."""
        return ParameterPosterior(f"-{self.name}", -self.samples)


class PosteriorSource:
    """Draws by flat name, read from the ``posterior`` group."""

    def __init__(self, posterior: Any) -> None:
        self._posterior = posterior

    def __contains__(self, name: str) -> bool:
        """Whether a parameter was sampled or reported."""
        return name in self._posterior.data_vars

    def get(self, name: str) -> ParameterPosterior:
        """Return one parameter's posterior.

        Raises
        ------
        KeyError
            If the name is not in the posterior.
        """
        if name not in self._posterior.data_vars:
            raise KeyError(f"{name!r} is not in the posterior.")
        values = self._posterior[name].transpose("chain", "draw", ...).values
        return ParameterPosterior(name, np.asarray(values, dtype=float))


class KernelPosterior:
    """A line's convolution kernel: ``fwhm`` and ``fraction``."""

    def __init__(self, source: PosteriorSource, prefix: str) -> None:
        self._source = source
        self._prefix = prefix

    @property
    def fwhm(self) -> ParameterPosterior:
        """Kernel FWHM (km/s)."""
        return self._source.get(f"{self._prefix}.fwhm")

    @property
    def fraction(self) -> ParameterPosterior:
        """Convolved fraction."""
        return self._source.get(f"{self._prefix}.fraction")


class ComponentPosterior:
    """Posterior of one line, read like its declaration.

    ``center`` (data unit), ``delta_v`` (km/s), ``fwhm`` (km/s) and ``flux``
    always exist, whether free, bound or fixed. ``delta_v_from_base``,
    ``fwhm_ratio_to_base`` and ``flux_ratio_to_base`` exist only when free.
    Kernels are attributes by kind (``.laplace.fwhm``).
    """

    _RELATIVE = ("delta_v_from_base", "fwhm_ratio_to_base", "flux_ratio_to_base")

    def __init__(self, source: PosteriorSource, handle: LineHandle) -> None:
        self._source = source
        self._handle = handle

    @property
    def name(self) -> str:
        """The line's name."""
        return self._handle.name

    @property
    def center(self) -> ParameterPosterior:
        """Line centre in the spectrum's wavelength unit."""
        return self._source.get(f"{self.name}.center")

    @property
    def delta_v(self) -> ParameterPosterior:
        """Velocity offset from the nominal wavelength (km/s)."""
        return self._source.get(f"{self.name}.delta_v")

    @property
    def fwhm(self) -> ParameterPosterior:
        """Intrinsic FWHM (km/s)."""
        return self._source.get(f"{self.name}.fwhm")

    @property
    def flux(self) -> ParameterPosterior:
        """Integrated flux."""
        return self._source.get(f"{self.name}.flux")

    def __getattr__(self, attribute: str) -> Any:
        """Free relative parameters and kernels by kind."""
        if attribute.startswith("_"):
            raise AttributeError(attribute)
        if attribute in self._RELATIVE:
            name = f"{self.name}.{attribute}"
            if name not in self._source:
                raise AttributeError(
                    f"{name} is not a free parameter of this fit (it is fixed or not declared)."
                )
            return self._source.get(name)
        if attribute in self._handle.kernels:
            return KernelPosterior(self._source, f"{self.name}.{attribute}")
        raise AttributeError(f"line {self.name!r} has no {attribute!r}.")

    def __repr__(self) -> str:
        """Show the four physical quantities."""
        parts = [
            f"{key}={getattr(self, key).median:.6g}"
            for key in ("center", "fwhm", "flux")
        ]
        return f"ComponentPosterior({self.name!r}, {', '.join(parts)})"


class ContinuumPosterior:
    """Posterior of continuum parameters shared by frames (or pooled means)."""

    def __init__(
        self, source: PosteriorSource, names: tuple[str, ...], where: str
    ) -> None:
        self._source = source
        self._names = names
        self._where = where

    def coefficient(self, degree: int) -> ParameterPosterior:
        """Polynomial coefficient ``c_degree``."""
        return self._get(f"c{degree}")

    @property
    def level(self) -> ParameterPosterior:
        """Power-law flux at ``lambda_0``."""
        return self._get("level")

    @property
    def index(self) -> ParameterPosterior:
        """Power-law index."""
        return self._get("index")

    def _get(self, parameter: str) -> ParameterPosterior:
        for name in self._names:
            if name.endswith(f"continuum.{parameter}"):
                return self._source.get(name)
        raise KeyError(
            f"continuum parameter {parameter!r} is not available {self._where}; per-frame "
            "values are in result.frames[frame_id].continuum."
        )


class NoisePosterior:
    """A frame's noise parameters."""

    def __init__(self, source: PosteriorSource, frame_id: str) -> None:
        self._source = source
        self._prefix = f"frame.{frame_id}.noise"

    @property
    def scale(self) -> ParameterPosterior:
        """Error scale."""
        return self._source.get(f"{self._prefix}.scale")

    @property
    def rho(self) -> ParameterPosterior:
        """AR(1) lag-one correlation."""
        name = f"{self._prefix}.rho"
        if name not in self._source:
            raise AttributeError("this frame has no AR(1) correlation.")
        return self._source.get(name)


class FramePosterior:
    """One frame of a joint fit: its own parameters and goodness of fit.

    Attributes
    ----------
    id
        Frame id.
    noise
        :class:`NoisePosterior` of the frame.
    continuum
        :class:`ContinuumPosterior` with the frame's own continuum values.
    chi2
        Whitened chi-square of the posterior-median model on the fitted
        pixels (correlated noise is whitened first).
    n_pixels
        Number of fitted pixels.
    """

    def __init__(
        self,
        source: PosteriorSource,
        workspace: NoobFitWorkspace,
        frame_id: str,
        *,
        chi2: float,
    ) -> None:
        self.id = frame_id
        self.noise = NoisePosterior(source, frame_id)
        self.continuum = ContinuumPosterior(
            source,
            workspace.continuum.frame_parameters[frame_id],
            f"for frame {frame_id!r}",
        )
        self.chi2 = chi2
        self.n_pixels = int(workspace.fitted[frame_id].sum())

    def __repr__(self) -> str:
        """Show the frame's fit statistic."""
        return f"FramePosterior({self.id!r}, chi2={self.chi2:.1f}/{self.n_pixels})"


class FramePosteriors(Mapping[str, FramePosterior]):
    """Frames by id."""

    def __init__(self, frames: Mapping[str, FramePosterior]) -> None:
        self._frames = dict(frames)

    def __repr__(self) -> str:
        """List the frames."""
        return (
            "FramePosteriors(" + ", ".join(repr(f) for f in self._frames.values()) + ")"
        )

    def __getitem__(self, frame_id: str) -> FramePosterior:
        """Return one frame."""
        try:
            return self._frames[frame_id]
        except KeyError:
            raise KeyError(
                f"unknown frame {frame_id!r}; frames are {list(self._frames)}."
            ) from None

    def __iter__(self) -> Iterator[str]:
        """Iterate over frame ids."""
        return iter(self._frames)

    def __len__(self) -> int:
        """Return the number of frames."""
        return len(self._frames)
