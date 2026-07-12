"""Shared engine for the matplotlib spectrum plotters.

This module holds the dimension-agnostic spectrum machinery reused by both
:func:`~noobfriend.core.display.plot._spectrum1d.plot_spectrum1d` and (in
future) ``plot_spectrum2d``: the :class:`ModelSpec` mapping and its validated
:class:`_ModelCurve` normal form, the data-line / error / colour normalisation
helpers, and :func:`draw_spectrum`, which draws one or several *data* spectra
(histogram-style steps with an optional uncertainty band) plus *model* overlays
(smooth lines, optionally banded) onto a caller-supplied
:class:`~matplotlib.axes.Axes`. Drawing onto a passed Axes is exactly what lets
the stacked 2-D-over-1-D panel reuse the 1-D line drawer on a ``sharex`` subplot.

matplotlib (not Bokeh) is used for spectra because they are static publication
line plots with no interactive need and trivial ``savefig`` export. The plotter
is deliberately unit-agnostic: ``wavelength`` and ``flux`` are bare numeric
arrays and units are purely a labelling concern handled by ``x_label`` /
``y_label``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from matplotlib.axes import Axes

#: Number of samples used to evaluate a model ``flux_func`` over its range.
_MODEL_GRID_N: int = 512

#: Recognised keys of a :class:`ModelSpec`; anything else is rejected.
_MODEL_KEYS: frozenset[str] = frozenset(
    {
        "flux_func",
        "wavelength",
        "flux",
        "error",
        "wavelength_range",
        "label",
        "color",
        "linestyle",
    }
)


class ModelSpec(TypedDict, total=False):
    """A single model curve overlaid on a spectrum.

    A model is given in one of two mutually exclusive forms: *functional*
    (``flux_func``, evaluated on an automatically generated dense grid) or
    *sampled* (explicit ``wavelength`` + ``flux`` arrays). All keys are
    optional, but exactly one of the two forms must be supplied; the validation
    (with precise error messages) lives in :meth:`_ModelCurve.from_input`.

    Because this is a :class:`~typing.TypedDict`, a plain dict literal at a call
    site is statically checked for key names and value types -- no import or
    construction is required to use it.

    Keys
    ----
    flux_func : callable
        ``f(wavelength) -> flux`` returning either a single array (no band) or
        ``(low, mid, high)`` (the band edges plus the plotted curve). Evaluated
        on a dense grid over ``wavelength_range`` if given, else over the span
        of the data spectra.
    wavelength, flux : numpy.ndarray
        Pre-sampled curve, given together. Mutually exclusive with ``flux_func``.
    error : numpy.ndarray or tuple of numpy.ndarray
        Uncertainty band for the sampled form only: a 1-D ``sigma`` array
        (symmetric ``flux +/- sigma``) or a ``(low, high)`` pair of arrays
        (asymmetric). Invalid with ``flux_func`` (which carries its own band).
    wavelength_range : tuple of float
        ``(min, max)`` grid extent for ``flux_func``. Valid only with
        ``flux_func``; redundant with an explicit ``wavelength``.
    label : str
        Legend label.
    color : str
        Matplotlib color; defaults to the next color in the cycle.
    linestyle : str
        Matplotlib linestyle; defaults to ``"-"``.
    """

    flux_func: Callable[
        [np.ndarray], np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]
    ]
    wavelength: np.ndarray
    flux: np.ndarray
    error: np.ndarray | tuple[np.ndarray, np.ndarray]
    wavelength_range: tuple[float, float]
    label: str
    color: str
    linestyle: str


def _check_range(wavelength_range: Any, where: str) -> tuple[float, float]:
    """Validate a ``(min, max)`` model grid range."""
    seq = tuple(wavelength_range)
    if len(seq) != 2:
        raise ValueError(f"{where}: 'wavelength_range' must be (min, max)")
    low, high = float(seq[0]), float(seq[1])
    if not low < high:
        raise ValueError(f"{where}: 'wavelength_range' must have min < max, got {seq}")
    return low, high


def _normalize_model_error(
    error: Any, flux: np.ndarray, where: str
) -> np.ndarray | tuple[np.ndarray, np.ndarray] | None:
    """Normalise a sampled model's ``error`` into ``sigma`` or ``(low, high)``.

    A length-2 container whose *elements are arrays* is read as an asymmetric
    ``(low, high)`` band; anything else is read as a symmetric ``sigma`` array
    (so a plain ``[0.1, 0.2]`` of scalars is sigma-per-point, not a band).
    """
    if error is None:
        return None
    if (
        isinstance(error, (tuple, list))
        and len(error) == 2
        and np.ndim(error[0]) >= 1
        and np.ndim(error[1]) >= 1
    ):
        low = np.asarray(error[0], dtype=float)
        high = np.asarray(error[1], dtype=float)
        if low.shape != flux.shape or high.shape != flux.shape:
            raise ValueError(
                f"{where}: 'error' band arrays must match 'flux' length {flux.shape}"
            )
        return low, high
    sigma = np.asarray(error, dtype=float)
    if sigma.shape != flux.shape:
        raise ValueError(
            f"{where}: 'error' length {sigma.shape} != 'flux' length {flux.shape}"
        )
    return sigma


def _unpack_func_output(
    out: Any, where: str
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Turn a ``flux_func`` return into ``(mid, low, high)``.

    One array (or 1-row 2-D) is the plotted curve with no band; three arrays
    (a 3-tuple or a 3-row 2-D) are ``(low, mid, high)``. Any other arity is a
    user error.
    """
    if isinstance(out, (tuple, list)):
        parts = list(out)
    elif isinstance(out, np.ndarray) and out.ndim == 2:
        parts = list(out)
    else:
        parts = [out]
    parts = [np.asarray(p, dtype=float) for p in parts]
    if len(parts) == 1:
        return parts[0], None, None
    if len(parts) == 3:
        low, mid, high = parts
        return mid, low, high
    raise ValueError(
        f"{where}: flux_func must return 1 array (flux) or 3 (low, mid, high), "
        f"got {len(parts)}"
    )


@dataclass(frozen=True)
class _ModelCurve:
    """A validated, draw-ready model curve (internal normalised form).

    Built via :meth:`from_input` from a :class:`ModelSpec`, a bare callable, or
    a ``(wavelength, flux)`` tuple; :meth:`sample` then produces the arrays to
    plot, generating the dense grid for the functional form on demand.
    """

    flux_func: Callable[[np.ndarray], Any] | None
    wavelength: np.ndarray | None
    flux: np.ndarray | None
    error: np.ndarray | tuple[np.ndarray, np.ndarray] | None
    wavelength_range: tuple[float, float] | None
    label: str | None
    color: str | None
    linestyle: str

    @classmethod
    def from_input(cls, obj: ModelSpec | Callable | tuple, index: int) -> _ModelCurve:
        """Validate and normalise one model input.

        Parameters
        ----------
        obj : ModelSpec or callable or tuple
            A model dict, a bare ``flux_func`` callable, or a
            ``(wavelength, flux)`` 2-tuple.
        index : int
            Position in the ``models`` list, used only in error messages.

        Returns
        -------
        _ModelCurve
            The normalised curve.

        Raises
        ------
        ValueError
            On an unknown key, a missing or doubly specified curve, an ``error``
            given with a ``flux_func``, a ``wavelength_range`` given without one,
            or a length mismatch.
        """
        where = f"models[{index}]"
        if isinstance(obj, dict):
            spec: dict = dict(obj)
        elif callable(obj):
            spec = {"flux_func": obj}
        elif isinstance(obj, tuple) and len(obj) == 2:
            if np.ndim(obj[0]) < 1 or np.ndim(obj[1]) < 1:
                raise ValueError(
                    f"{where}: a 2-tuple is read as (wavelength, flux) arrays; "
                    "pass a list for multiple models"
                )
            spec = {"wavelength": obj[0], "flux": obj[1]}
        else:
            raise ValueError(
                f"{where} must be a dict, a callable, or a (wavelength, flux) "
                f"tuple, got {type(obj).__name__}"
            )

        unknown = set(spec) - _MODEL_KEYS
        if unknown:
            raise ValueError(
                f"{where} has unknown key(s) {sorted(unknown)}; "
                f"valid keys are {sorted(_MODEL_KEYS)}"
            )

        flux_func = spec.get("flux_func")
        wavelength = spec.get("wavelength")
        flux = spec.get("flux")
        error = spec.get("error")
        wavelength_range = spec.get("wavelength_range")

        has_func = flux_func is not None
        has_wave = wavelength is not None
        has_flux = flux is not None
        if has_wave != has_flux:
            raise ValueError(f"{where}: 'wavelength' and 'flux' must be given together")
        has_arrays = has_wave and has_flux
        if has_func and has_arrays:
            raise ValueError(
                f"{where}: give either 'flux_func' or ('wavelength', 'flux'), not both"
            )
        if not has_func and not has_arrays:
            raise ValueError(
                f"{where}: provide a curve via 'flux_func' or ('wavelength', 'flux')"
            )
        if has_func and not callable(flux_func):
            raise ValueError(f"{where}: 'flux_func' must be callable")
        if error is not None and has_func:
            raise ValueError(
                f"{where}: 'error' applies only to the ('wavelength', 'flux') "
                "form; a 'flux_func' carries its band by returning (low, mid, high)"
            )
        if wavelength_range is not None and not has_func:
            raise ValueError(
                f"{where}: 'wavelength_range' applies only to 'flux_func' (it "
                "sets the grid the function is evaluated on); with explicit "
                "'wavelength' it is redundant"
            )

        wl = None if wavelength is None else np.asarray(wavelength, dtype=float)
        fl = None if flux is None else np.asarray(flux, dtype=float)
        if has_arrays and wl.shape != fl.shape:
            raise ValueError(
                f"{where}: 'wavelength' length {wl.shape} != 'flux' length {fl.shape}"
            )
        err = None if fl is None else _normalize_model_error(error, fl, where)
        wr = None if wavelength_range is None else _check_range(wavelength_range, where)
        return cls(
            flux_func=flux_func if has_func else None,
            wavelength=wl,
            flux=fl,
            error=err,
            wavelength_range=wr,
            label=spec.get("label"),
            color=spec.get("color"),
            linestyle=str(spec.get("linestyle", "-")),
        )

    def sample(
        self, default_range: tuple[float, float], where: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Produce ``(x, mid, low, high)`` arrays to plot.

        For the functional form, evaluates ``flux_func`` on a dense grid over
        ``wavelength_range`` (or ``default_range`` if unset). For the sampled
        form, returns the stored arrays and derives the band from ``error``.
        ``low``/``high`` are ``None`` when there is no band.
        """
        if self.flux_func is not None:
            low, high = self.wavelength_range or default_range
            x = np.linspace(low, high, _MODEL_GRID_N)
            mid, lo, hi = _unpack_func_output(self.flux_func(x), where)
            return x, mid, lo, hi
        x = self.wavelength
        mid = self.flux
        assert x is not None and mid is not None  # noqa: S101  (arrays form)
        if self.error is None:
            return x, mid, None, None
        if isinstance(self.error, tuple):
            lo, hi = self.error
            return x, mid, lo, hi
        return x, mid, mid - self.error, mid + self.error


def _to_lines(x: ArrayLike | Sequence[ArrayLike], name: str) -> list[np.ndarray]:
    """Normalise a single 1-D array or a sequence of 1-D arrays to a list.

    A 2-D array is read row-by-row (each row a spectrum); a sequence whose first
    element is itself array-like is read as one spectrum per element; anything
    else is a single 1-D spectrum.
    """
    if isinstance(x, np.ndarray):
        if x.ndim == 1:
            return [x.astype(float, copy=False)]
        if x.ndim == 2:
            return [np.asarray(row, dtype=float) for row in x]
        raise ValueError(f"{name} must be 1-D or 2-D, got {x.ndim}-D")
    seq = list(x)
    if len(seq) == 0:
        raise ValueError(f"{name} is empty")
    if np.ndim(seq[0]) >= 1:
        return [np.asarray(a, dtype=float) for a in seq]
    return [np.asarray(seq, dtype=float)]


def _resolve_wavelengths(
    wavelength: ArrayLike | Sequence[ArrayLike], flux_lines: list[np.ndarray]
) -> list[np.ndarray]:
    """Resolve ``wavelength`` against the flux lines: shared array or one each."""
    n = len(flux_lines)
    waves = _to_lines(wavelength, "wavelength")
    if len(waves) == 1 and n > 1:
        waves = waves * n
    if len(waves) != n:
        raise ValueError(
            f"wavelength has {len(waves)} array(s), expected 1 (shared) or {n}"
        )
    for i, (w, f) in enumerate(zip(waves, flux_lines)):
        if w.shape != f.shape:
            raise ValueError(
                f"wavelength[{i}] length {w.shape} != flux length {f.shape}"
            )
    return waves


def _resolve_errors(
    error: ArrayLike | Sequence[ArrayLike | None] | None, flux_lines: list[np.ndarray]
) -> list[np.ndarray | None]:
    """Resolve per-line ``error``: ``None``, one sigma array, or one per line.

    A sequence whose first element is ``None`` or array-like is read as one
    (optional) sigma array per line; otherwise the input is a single sigma array
    for a single spectrum.
    """
    n = len(flux_lines)
    if error is None:
        return [None] * n
    if isinstance(error, (list, tuple)) and (
        len(error) == 0 or error[0] is None or np.ndim(error[0]) >= 1
    ):
        seq = list(error)
        if len(seq) != n:
            raise ValueError(
                f"error has length {len(seq)}, expected {n} (one per line)"
            )
        out: list[np.ndarray | None] = []
        for i, (e, f) in enumerate(zip(seq, flux_lines)):
            if e is None:
                out.append(None)
                continue
            ea = np.asarray(e, dtype=float)
            if ea.shape != f.shape:
                raise ValueError(
                    f"error[{i}] length {ea.shape} != flux length {f.shape}"
                )
            out.append(ea)
        return out
    if n != 1:
        raise ValueError(
            f"error must be a list of {n} arrays (one per line) when multiple "
            "fluxes are given"
        )
    ea = np.asarray(error, dtype=float)
    if ea.shape != flux_lines[0].shape:
        raise ValueError(
            f"error length {ea.shape} != flux length {flux_lines[0].shape}"
        )
    return [ea]


def _broadcast_labels(labels: str | Sequence[str] | None, n: int) -> list[str | None]:
    """Resolve ``labels`` into one (optional) label per line."""
    if labels is None:
        return [None] * n
    if isinstance(labels, str):
        if n != 1:
            raise ValueError(
                f"labels must be a list of {n} strings when multiple fluxes are given"
            )
        return [labels]
    seq = [str(s) for s in labels]
    if len(seq) != n:
        raise ValueError(f"labels has length {len(seq)}, expected {n}")
    return seq  # type: ignore[return-value]


def _broadcast_colors(colors: str | Sequence[str] | None, n: int) -> list[str | None]:
    """Resolve ``colors`` into one (optional) color per line.

    ``None`` lets matplotlib's color cycle choose; a single string is broadcast
    to every line; a sequence is used verbatim and must have length ``n``.
    """
    if colors is None:
        return [None] * n
    if isinstance(colors, str):
        return [colors] * n
    seq = list(colors)
    if len(seq) != n:
        raise ValueError(f"colors length {len(seq)} != {n}")
    return seq


def _normalize_models(
    models: ModelSpec | Callable | tuple | Sequence | None,
) -> list[_ModelCurve]:
    """Normalise the ``models`` argument into a list of :class:`_ModelCurve`.

    A ``list`` is read as several models; a dict, callable, or
    ``(wavelength, flux)`` tuple is a single model.
    """
    if models is None:
        return []
    items = models if isinstance(models, list) else [models]
    return [_ModelCurve.from_input(m, i) for i, m in enumerate(items)]


def _data_range(waves: list[np.ndarray]) -> tuple[float, float]:
    """Return the finite ``(min, max)`` wavelength span across all data lines."""
    allw = np.concatenate([w.ravel() for w in waves])
    finite = allw[np.isfinite(allw)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(finite.min()), float(finite.max())


def draw_spectrum(
    ax: Axes,
    wavelength: ArrayLike | Sequence[ArrayLike],
    flux: ArrayLike | Sequence[ArrayLike],
    *,
    error: ArrayLike | Sequence[ArrayLike | None] | None = None,
    labels: str | Sequence[str] | None = None,
    colors: str | Sequence[str] | None = None,
    models: ModelSpec | Callable | tuple | Sequence | None = None,
    drawstyle: Literal["steps", "line"] = "steps",
    error_style: Literal["band", "line", "none"] = "band",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    ylog: bool = False,
    x_label: str = "Wavelength",
    y_label: str = "Flux",
    title: str | None = None,
) -> None:
    """Draw data spectra and model overlays onto a caller-supplied ``ax``.

    The shared drawing engine behind :func:`plot_spectrum1d` and the stacked
    :func:`plot_spectrum2d` panel, exposed so callers can compose their own
    multi-panel figures (e.g. a spectrum over a shared-x residual axis). It only
    draws onto ``ax`` -- it creates and saves no figure. See
    :func:`plot_spectrum1d` for the parameter semantics.
    """
    if drawstyle not in ("steps", "line"):
        raise ValueError(f"drawstyle must be 'steps' or 'line', got {drawstyle!r}")
    if error_style not in ("band", "line", "none"):
        raise ValueError(
            f"error_style must be 'band', 'line', or 'none', got {error_style!r}"
        )

    flux_lines = _to_lines(flux, "flux")
    n = len(flux_lines)
    waves = _resolve_wavelengths(wavelength, flux_lines)
    errs = _resolve_errors(error, flux_lines)
    label_list = _broadcast_labels(labels, n)
    color_list = _broadcast_colors(colors, n)

    has_legend = False
    for w, f, e, lab, col in zip(waves, flux_lines, errs, label_list, color_list):
        line_kw: dict[str, Any] = {}
        if col is not None:
            line_kw["color"] = col
        if lab is not None:
            line_kw["label"] = lab
            has_legend = True
        if drawstyle == "steps":
            (line,) = ax.step(w, f, where="mid", **line_kw)
        else:
            (line,) = ax.plot(w, f, **line_kw)
        c = line.get_color()
        if e is not None and error_style != "none":
            if error_style == "band":
                step_kw = {"step": "mid"} if drawstyle == "steps" else {}
                ax.fill_between(
                    w, f - e, f + e, color=c, alpha=0.25, linewidth=0, **step_kw
                )
            elif drawstyle == "steps":
                ax.step(w, e, where="mid", color=c, linewidth=0.8, alpha=0.7)
            else:
                ax.plot(w, e, color=c, linewidth=0.8, alpha=0.7)

    default_range = _data_range(waves)
    for i, mc in enumerate(_normalize_models(models)):
        x, mid, low, high = mc.sample(default_range, f"models[{i}]")
        model_kw: dict[str, Any] = {"linestyle": mc.linestyle, "zorder": 3}
        if mc.color is not None:
            model_kw["color"] = mc.color
        if mc.label is not None:
            model_kw["label"] = mc.label
            has_legend = True
        (line,) = ax.plot(x, mid, **model_kw)
        if low is not None:
            ax.fill_between(
                x, low, high, color=line.get_color(), alpha=0.2, linewidth=0, zorder=2
            )

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if ylog:
        ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    if has_legend:
        ax.legend()


def draw_residual(
    ax: Axes,
    wavelength: ArrayLike,
    residual: ArrayLike,
    *,
    error: ArrayLike | None = None,
    color: str = "#1A1A1A",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    x_label: str = "Wavelength",
    y_label: str = "Residual",
) -> None:
    """Draw residual samples, their optional uncertainties, and a zero line."""
    wave = np.asarray(wavelength, dtype=float)
    values = np.asarray(residual, dtype=float)
    if wave.ndim != 1 or values.ndim != 1:
        raise ValueError("residual wavelength and values must be 1-D arrays")
    if wave.shape != values.shape:
        raise ValueError(
            f"residual length {values.shape} must match wavelength {wave.shape}"
        )
    errors = None if error is None else np.asarray(error, dtype=float)
    if errors is not None:
        if errors.ndim != 1 or errors.shape != values.shape:
            raise ValueError(
                f"residual error length {errors.shape} must match residual "
                f"{values.shape}"
            )
        finite_errors = errors[np.isfinite(errors)]
        if np.any(finite_errors < 0.0):
            raise ValueError("residual errors must be nonnegative")

    ax.axhline(0.0, color="#7F7F7F", linewidth=1.0, zorder=1)
    if errors is None:
        ax.plot(
            wave,
            values,
            linestyle="none",
            marker="o",
            markersize=3.2,
            color=color,
            zorder=2,
        )
    else:
        ax.errorbar(
            wave,
            values,
            yerr=errors,
            linestyle="none",
            marker="o",
            markersize=3.2,
            color=color,
            ecolor=color,
            elinewidth=0.8,
            capsize=0,
            alpha=0.8,
            zorder=2,
        )
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is None:
        extent = np.abs(values)
        if errors is not None:
            extent = extent + np.where(np.isfinite(errors), errors, 0.0)
        finite = extent[np.isfinite(extent)]
        if finite.size:
            limit = 1.1 * float(np.max(finite))
            if limit > 0.0:
                ax.set_ylim(-limit, limit)
    else:
        ax.set_ylim(*ylim)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
