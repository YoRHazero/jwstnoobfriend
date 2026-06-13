"""``LineFitSetup``: a list of ``NoobLine`` compiled against a spectrum.

:meth:`NoobSpectrum.setup` calls :func:`build_setup`, which (1) generates the
minimal unique id of every component, (2) validates the derive graph (every
parent present, no cycles, one wavelength unit), and (3) resolves each
component's three axes -- centre velocity, velocity FWHM, flux -- into a
:class:`ResolvedAxis` describing whether that axis is fixed, free (with bounds),
or tied to a parent. The result, :class:`LineFitSetup`, is the inspectable
hand-off to the (not-yet-wired) PyMC layer; :meth:`LineFitSetup.summary` prints
the whole graph so it can be eyeballed before fitting.

Ids follow a minimal scheme: a line with a single component is just its
``linename``; a line with several components is ``linename.component``; repeated
``(linename, component)`` pairs get a 1-based numeric suffix. A
``NoobLine.custom_id`` overrides the generated id. Ties are by object identity
(the derive parent), so id reassignment never affects the model.

This module is internal; :class:`LineFitSetup` is re-exported from
:mod:`noobfriend.inference.spectrum`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from noobfriend.inference.spectrum._line import classify_spec
from noobfriend.inference.spectrum._template import ComponentTemplate, get_template
from noobfriend.inference.spectrum._units import convert

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.figure import Figure

    from noobfriend.inference.spectrum._line import NoobLine
    from noobfriend.inference.spectrum._result import LineFitResult
    from noobfriend.inference.spectrum._spectrum import NoobSpectrum

__all__ = ["LineFitSetup", "ResolvedAxis", "ResolvedComponent", "build_setup"]

#: Default free centre-velocity prior support (km/s) for a floating root line.
DEFAULT_DV_BOUNDS: tuple[float, float] = (-300.0, 300.0)


@dataclass(frozen=True)
class ResolvedAxis:
    """The resolved treatment of one parameter axis of one component.

    Attributes
    ----------
    kind : {"fixed", "free", "tied"}
        ``"fixed"`` a constant, ``"free"`` a sampled parameter, ``"tied"`` equal
        to a parent's same axis.
    value : float or None
        The constant for ``"fixed"`` (an offset, a width, a flux, or a ratio).
    bounds : tuple of float or None
        The ``(lower, upper)`` prior support for ``"free"``; ``None`` means a
        data-driven default decided later (flux only).
    base_id : str or None
        The parent component id this axis is relative to; ``None`` if absolute.
    relation : {"equal", "ratio", "offset"} or None
        How ``base_id`` enters: an additive velocity ``"offset"``, a multiplicative
        flux ``"ratio"``, or width ``"equal"``.
    unit : str
        Unit label for display (``"km/s"`` for centre/width; empty for flux).
    """

    kind: Literal["fixed", "free", "tied"]
    value: float | None = None
    bounds: tuple[float, float] | None = None
    base_id: str | None = None
    relation: Literal["equal", "ratio", "offset"] | None = None
    unit: str = ""

    def render(self) -> str:
        """Return a one-line human-readable description of the resolution."""
        u = f" {self.unit}".rstrip()
        if self.base_id is None:
            if self.kind == "fixed":
                return f"{self.value:g}{u}"
            if self.bounds is None:
                return "free (data-driven)"
            return f"free [{self.bounds[0]:g}, {self.bounds[1]:g}]{u}"
        if self.relation == "equal":
            return f"= {self.base_id}"
        if self.relation == "ratio":
            if self.kind == "fixed":
                return f"= {self.value:g} × {self.base_id}"
            assert self.bounds is not None  # noqa: S101
            return f"= [{self.bounds[0]:g}, {self.bounds[1]:g}] × {self.base_id}"
        # offset
        if self.kind == "fixed":
            if self.value == 0:
                return f"= {self.base_id} (co-moving)"
            sign = "+" if (self.value or 0) > 0 else "−"
            return f"= {self.base_id} {sign} {abs(self.value or 0):g}{u}"
        assert self.bounds is not None  # noqa: S101
        return f"= {self.base_id} + free [{self.bounds[0]:g}, {self.bounds[1]:g}]{u}"


@dataclass(frozen=True)
class ResolvedComponent:
    """One compiled component: its id, source line, template, and resolved axes.

    Attributes
    ----------
    id : str
        The generated (or custom) component id.
    line : NoobLine
        The source declaration.
    rest_wavelength : float
        The line's rest wavelength converted into the spectrum's ``wave_unit``
        (so all downstream code works in one frame).
    template : ComponentTemplate
        The component's shape template.
    centre, width, flux : ResolvedAxis
        The resolved velocity offset, velocity FWHM, and flux axes.
    """

    id: str
    line: NoobLine
    rest_wavelength: float
    template: ComponentTemplate
    centre: ResolvedAxis
    width: ResolvedAxis
    flux: ResolvedAxis


@dataclass(frozen=True)
class LineFitSetup:
    """A compiled, inspectable line-fit model (pre-sampling).

    Returned by :meth:`NoobSpectrum.setup`; not constructed directly. The PyMC
    model, window, and initial values are wired in a later step.

    Attributes
    ----------
    spectrum : NoobSpectrum
        The spectrum the components were compiled against.
    components : tuple of ResolvedComponent
        The compiled components, in input order.
    model_name : str
        A label for this model, carried onto the result for
        :meth:`~noobfriend.inference.spectrum.LineFitResult.compare`.
    mask_excluded : numpy.ndarray or None
        The effective fit-time bad-pixel mask (``True`` = excluded): the
        spectrum's ``mask_excluded`` combined with whatever :meth:`with_mask`
        last applied. ``None`` is no mask.
    """

    spectrum: NoobSpectrum
    components: tuple[ResolvedComponent, ...]
    model_name: str
    mask_excluded: np.ndarray | None = field(default=None, compare=False, repr=False)

    def summary(self) -> str:
        """Return a table of every component's id and resolved axes."""
        header = ("id", "component", "centre (km/s)", "fwhm (km/s)", "flux")
        rows = [
            (
                c.id,
                c.template.name,
                c.centre.render(),
                c.width.render(),
                c.flux.render(),
            )
            for c in self.components
        ]
        widths = [max(len(r[i]) for r in (header, *rows)) for i in range(len(header))]
        fmt = "  ".join(f"{{:<{w}}}" for w in widths)
        lines = [fmt.format(*header), fmt.format(*("-" * w for w in widths))]
        lines += [fmt.format(*r) for r in rows]
        return "\n".join(lines)

    def with_mask(self, mask: Any = None, **auto_kw: Any) -> LineFitSetup:
        """Return a copy of this setup with fit-time pixels masked.

        Immutable (it does not mutate ``self``): bind the result and reuse it for
        both preview and fit, so they see the identical mask::

            masked = setup.with_mask("auto")
            masked.plot()   # previews the mask (hatched)
            masked.run()    # fits with it

        The mask layers on top of the spectrum's own ``mask_excluded`` (a
        data-quality flag, always honoured); each call *replaces* the fit-time
        layer rather than accumulating.

        Parameters
        ----------
        mask : None, "auto", list of [lo, hi], or array_like of bool
            ``None`` clears the fit-time layer (back to the spectrum's mask).
            ``"auto"`` detects artifacts / unrelated emission via
            :func:`~noobfriend.inference.spectrum._mask.compute_auto_mask`. A
            list of ``[lo, hi]`` wavelength intervals (in the spectrum's unit)
            masks those ranges; a length-N boolean array is used directly.
        **auto_kw
            Tuning forwarded to ``compute_auto_mask`` (e.g. ``tau``,
            ``sigma_isolated``, ``sigma_run``, ``window``); only valid with
            ``mask="auto"``.

        Returns
        -------
        LineFitSetup
            A copy whose :attr:`mask_excluded` is the combined effective mask.
        """
        from noobfriend.inference.spectrum._mask import resolve_mask

        layer = resolve_mask(self, mask, **auto_kw)
        dq = self.spectrum.mask_excluded
        if layer is None:
            effective = dq
        elif dq is None:
            effective = layer
        else:
            effective = dq | layer
        return replace(self, mask_excluded=effective)

    def plot(
        self,
        *,
        window: tuple[float, float] | None = None,
        pad_kms: float = 5000.0,
        continuum_degree: int = 1,
        **kwargs: object,
    ) -> Figure:
        """Preview the fitting window and initial-guess model over the data.

        Drawn before sampling (and without needing the ``mcmc`` extra), so a bad
        window or starting point is caught early. ``window`` / ``pad_kms`` /
        ``continuum_degree`` mirror :meth:`run` so the preview matches the fit.

        Parameters
        ----------
        window : tuple of float, optional
            Explicit ``(lo, hi)`` window; else auto from the line span.
        pad_kms : float, default 5000.0
            Velocity padding for the automatic window.
        continuum_degree : int, default 1
            Local continuum polynomial degree.
        **kwargs
            Forwarded to
            :func:`~noobfriend.inference.spectrum._plot.preview_setup` (``size``,
            ``title``, ``save``).

        Returns
        -------
        matplotlib.figure.Figure
        """
        from noobfriend.inference.spectrum._plot import preview_setup

        return preview_setup(
            self,
            window=window,
            pad_kms=pad_kms,
            continuum_degree=continuum_degree,
            **kwargs,
        )

    def run(
        self,
        *,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        nuts_sampler: str = "pymc",
        mp_ctx: str | None = "forkserver",
        window: tuple[float, float] | None = None,
        pad_kms: float = 5000.0,
        continuum_degree: int = 1,
        jitter: bool = False,
        random_seed: int | None = None,
        progressbar: bool = False,
        compute_log_likelihood: bool = True,
        **sample_kwargs: object,
    ) -> LineFitResult:
        """Build the PyMC model, sample it, and return the result.

        Parameters
        ----------
        draws, tune, chains : int
            NUTS draw count, tuning steps, and chains.
        target_accept : float, default 0.9
            NUTS target acceptance (raise toward 0.95 if divergences appear).
        nuts_sampler : str, default "pymc"
            Sampling backend: ``"pymc"`` (default) or ``"numpyro"`` / ``"blackjax"``
            / ``"nutpie"`` (the same model, compiled to JAX etc.).
        mp_ctx : str or None, default "forkserver"
            Multiprocessing start method for the ``"pymc"`` backend's parallel
            chains. The default ``"forkserver"`` forks workers from a clean
            single-threaded server, avoiding the "fork() in a multi-threaded
            process" deadlock risk (and its Python 3.14 ``DeprecationWarning``).
            ``None`` defers to PyMC; ignored by non-``"pymc"`` backends.
        window : tuple of float, optional
            Explicit ``(lo, hi)`` fitting window; else auto from the line span.
        pad_kms : float, default 5000.0
            Velocity padding for the automatic window.
        continuum_degree : int, default 1
            Local continuum polynomial degree.
        jitter : bool, default False
            Add a fractional error-inflation term to the likelihood.
        random_seed : int, optional
            Seed for reproducibility.
        progressbar : bool, default False
            Show the sampler progress bar.
        compute_log_likelihood : bool, default True
            Compute the pointwise log likelihood (needed for
            :meth:`~noobfriend.inference.spectrum.LineFitResult.compare`).
        **sample_kwargs
            Extra keyword arguments forwarded to ``pymc.sample`` (e.g.
            ``nuts_sampler_kwargs``).

        Returns
        -------
        LineFitResult

        Raises
        ------
        ImportError
            If the optional ``mcmc`` extra (PyMC) is not installed.
        """
        try:
            import pymc as pm
        except (
            ImportError
        ) as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "Sampling needs PyMC (the optional 'mcmc' extra). Install it with: "
                "uv add 'noobfriend[mcmc]'."
            ) from exc

        from noobfriend.inference.spectrum._pymc_model import build_model
        from noobfriend.inference.spectrum._result import LineFitResult

        bundle = build_model(
            self,
            window=window,
            pad_kms=pad_kms,
            continuum_degree=continuum_degree,
            jitter=jitter,
        )
        # Only the default PyMC backend forks Python workers; pick a safe start
        # method there. JAX-based backends manage their own chains.
        if mp_ctx is not None and nuts_sampler == "pymc":
            sample_kwargs.setdefault("mp_ctx", mp_ctx)
        with bundle.model:
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                nuts_sampler=nuts_sampler,
                random_seed=random_seed,
                progressbar=progressbar,
                **sample_kwargs,
            )
            if compute_log_likelihood:
                pm.compute_log_likelihood(idata, progressbar=False)
        return LineFitResult(
            idata=idata,
            model_name=self.model_name,
            components=bundle.components,
            bounded=bundle.bounded,
            continuum_degree=bundle.continuum_degree,
            lambda0=bundle.lambda0,
            window_wl=bundle.window_wl,
            window_flux=bundle.window_flux,
            window_error=bundle.window_error,
            spectrum=replace(self.spectrum, mask_excluded=self.mask_excluded),
        )

    def __repr__(self) -> str:
        """Concise representation: component count and ids (and masked pixels)."""
        ids = ", ".join(c.id for c in self.components)
        masked = (
            ""
            if self.mask_excluded is None
            else f", {int(self.mask_excluded.sum())} masked"
        )
        return f"LineFitSetup({len(self.components)} components: {ids}{masked})"


def _default_model_name(lines: list[NoobLine]) -> str:
    """Auto-name a model by its first line and that line's component count."""
    first = lines[0].linename
    n = sum(1 for line in lines if line.linename == first)
    return f"{first} {n}-component model"


def build_setup(
    spectrum: NoobSpectrum,
    lines: Sequence[NoobLine],
    *,
    model_name: str | None = None,
) -> LineFitSetup:
    """Compile ``lines`` against ``spectrum`` into a :class:`LineFitSetup`.

    Parameters
    ----------
    spectrum : NoobSpectrum
        The target spectrum.
    lines : sequence of NoobLine
        The model declarations (root lines and derived companions).
    model_name : str, optional
        A label for the model; defaults to ``"<first line> N-component model"``
        (N = that line's component count), which makes the common single-vs-double
        comparison self-naming.

    Returns
    -------
    LineFitSetup

    Raises
    ------
    ValueError
        If ``lines`` is empty, a derived line's parent is absent, the derive
        graph has a cycle, or generated/custom ids collide.
    """
    items = list(lines)
    if not items:
        raise ValueError("setup needs at least one NoobLine.")

    _validate_graph(items)
    id_by_obj = _generate_ids(items)

    resolved = []
    for line in items:
        template = get_template(line.component)
        parent = line.parent
        parent_id = None if parent is None else id_by_obj[id(parent)]
        resolved.append(
            ResolvedComponent(
                id=id_by_obj[id(line)],
                line=line,
                rest_wavelength=convert(
                    line.rest_wavelength, line.unit, spectrum.wave_unit
                ),
                template=template,
                centre=_resolve_centre(line, parent_id, spectrum.wave_unit),
                width=_resolve_width(line, parent, parent_id, template),
                flux=_resolve_flux(line, parent_id),
            )
        )
    return LineFitSetup(
        spectrum=spectrum,
        components=tuple(resolved),
        model_name=model_name if model_name is not None else _default_model_name(items),
        mask_excluded=spectrum.mask_excluded,
    )


def _validate_graph(lines: list[NoobLine]) -> None:
    """Check derived parents are present and the graph is acyclic."""
    present = {id(line) for line in lines}
    for line in lines:
        if line.parent is not None and id(line.parent) not in present:
            raise ValueError(
                f"line {line.linename!r}'s parent "
                f"({line.parent.linename}.{line.parent.component}) is not in the "
                "lines list; include it so the tie resolves."
            )
        seen: set[int] = set()
        cur: NoobLine | None = line
        while cur is not None:
            if id(cur) in seen:
                raise ValueError(
                    f"cycle in the derive chain involving {line.linename!r}."
                )
            seen.add(id(cur))
            cur = cur.parent


def _generate_ids(lines: list[NoobLine]) -> dict[int, str]:
    """Map each line's ``id()`` to its component id (custom_id wins).

    Lines carrying a ``custom_id`` are excluded from the auto-id grouping, so
    naming one component yourself does not push numeric suffixes onto its
    siblings (a lone auto narrow stays ``Halpha.narrow``, not ``Halpha.narrow.1``).
    """
    by_name: dict[str, list[int]] = defaultdict(list)
    for i, line in enumerate(lines):
        if line.custom_id is None:
            by_name[line.linename].append(i)

    auto: dict[int, str] = {}
    for name, idxs in by_name.items():
        if len(idxs) == 1:
            auto[idxs[0]] = name
            continue
        by_comp: dict[str, list[int]] = defaultdict(list)
        for i in idxs:
            by_comp[lines[i].component].append(i)
        for comp, cidxs in by_comp.items():
            if len(cidxs) == 1:
                auto[cidxs[0]] = f"{name}.{comp}"
            else:
                for n, i in enumerate(cidxs, start=1):
                    auto[i] = f"{name}.{comp}.{n}"

    final: dict[int, str] = {}
    seen: dict[str, int] = {}
    for i, line in enumerate(lines):
        cid = line.custom_id if line.custom_id is not None else auto[i]
        if cid in seen:
            raise ValueError(
                f"duplicate component id {cid!r} (positions {seen[cid]} and {i}); "
                "set a unique custom_id."
            )
        seen[cid] = i
        final[id(line)] = cid
    return final


def _resolve_centre(
    line: NoobLine, parent_id: str | None, wave_unit: str
) -> ResolvedAxis:
    """Resolve the centre-velocity axis (offset from parent or systemic).

    A ``delta_wavelength`` is converted into the spectrum's ``wave_unit`` so the
    later km/s conversion divides it by a same-frame line centre.
    """
    if line.delta_v_kms is not None:
        kind, payload = classify_spec(line.delta_v_kms, "delta_v_kms")
        unit = "km/s"
    elif line.delta_wavelength is not None:
        kind, payload = classify_spec(line.delta_wavelength, "delta_wavelength")
        assert line.unit is not None  # noqa: S101  (validated)
        if kind == "fixed":
            payload = convert(payload, line.unit, wave_unit)
        elif kind == "bounded":
            payload = (
                convert(payload[0], line.unit, wave_unit),
                convert(payload[1], line.unit, wave_unit),
            )
        unit = wave_unit
    else:
        kind, payload, unit = "none", None, "km/s"

    if kind == "none":
        if parent_id is None:  # root → floats with the default prior
            return ResolvedAxis(
                "free", bounds=DEFAULT_DV_BOUNDS, relation="offset", unit="km/s"
            )
        return ResolvedAxis(
            "fixed", value=0.0, base_id=parent_id, relation="offset", unit="km/s"
        )  # derived → locked co-moving
    if kind == "fixed":
        assert isinstance(payload, float)  # noqa: S101
        return ResolvedAxis(
            "fixed", value=payload, base_id=parent_id, relation="offset", unit=unit
        )
    assert isinstance(payload, tuple)  # noqa: S101
    return ResolvedAxis(
        "free", bounds=payload, base_id=parent_id, relation="offset", unit=unit
    )


def _resolve_width(
    line: NoobLine,
    parent: NoobLine | None,
    parent_id: str | None,
    template: ComponentTemplate,
) -> ResolvedAxis:
    """Resolve the velocity-FWHM axis (abs override > lock > auto)."""
    kind, payload = classify_spec(line.abs_fwhm, "abs_fwhm")
    if kind == "fixed":
        assert isinstance(payload, float)  # noqa: S101
        return ResolvedAxis("fixed", value=payload, unit="km/s")
    if kind == "bounded":
        assert isinstance(payload, tuple)  # noqa: S101
        return ResolvedAxis("free", bounds=payload, unit="km/s")

    if line.lock_fwhm is True:
        return ResolvedAxis("tied", base_id=parent_id, relation="equal", unit="km/s")
    if (
        line.lock_fwhm is None
        and parent is not None
        and line.component == parent.component
    ):
        return ResolvedAxis("tied", base_id=parent_id, relation="equal", unit="km/s")
    return ResolvedAxis("free", bounds=template.fwhm_kms_bounds, unit="km/s")


def _resolve_flux(line: NoobLine, parent_id: str | None) -> ResolvedAxis:
    """Resolve the flux axis (abs override > ratio to parent > free data-driven)."""
    kind, payload = classify_spec(line.abs_flux, "abs_flux")
    if kind == "fixed":
        assert isinstance(payload, float)  # noqa: S101
        return ResolvedAxis("fixed", value=payload)
    if kind == "bounded":
        assert isinstance(payload, tuple)  # noqa: S101
        return ResolvedAxis("free", bounds=payload)

    kind, payload = classify_spec(line.flux_ratio, "flux_ratio")
    if kind == "fixed":
        assert isinstance(payload, float)  # noqa: S101
        return ResolvedAxis("fixed", value=payload, base_id=parent_id, relation="ratio")
    if kind == "bounded":
        assert isinstance(payload, tuple)  # noqa: S101
        return ResolvedAxis("free", bounds=payload, base_id=parent_id, relation="ratio")
    return ResolvedAxis("free", bounds=None)
