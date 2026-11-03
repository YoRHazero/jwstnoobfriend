"""NoobLine: one emission or absorption component to fit."""

from __future__ import annotations

from dataclasses import dataclass, fields
from math import isfinite
from numbers import Real
from typing import Literal

from noobfriend.inference.spectrum._wavelength import WaveUnit, check_unit
from noobfriend.inference.spectrum.line._rules import LineKernel, LineRule
from noobfriend.inference.spectrum.priors import (
    FRACTION,
    NONNEGATIVE,
    POSITIVE,
    REAL,
    ParameterValue,
    check_value,
)
from noobfriend.inference.spectrum.profiles import KERNELS, check_kernels

type ComponentName = Literal["narrow", "broad"]
type ContributionName = Literal["emission", "absorption"]
type ProfileName = Literal["gaussian", "lorentzian", "exponential"]

#: Names reserved for the flat parameter paths of the continuum and frames.
RESERVED_NAMES: frozenset[str] = frozenset({"continuum", "frame"})
_COMPONENTS = ("narrow", "broad")
_CONTRIBUTIONS = ("emission", "absorption")
_DEFAULT = LineRule("default")


@dataclass(frozen=True, slots=True, init=False)
class NoobLine:
    """A line to fit: where it sits and how its parameters are set.

    The constructor takes only what the line *is*; parameters are set by
    :meth:`center`, :meth:`fwhm`, :meth:`flux` and :meth:`convolve`, and
    relations to another line only through :meth:`derive`. Every method
    returns a new line. A parameter value is a ``float`` (fixed) or a prior
    (free); parameters left alone get explicit default priors at ``prepare``.

    Parameters
    ----------
    name
        Unique id of the component in the fit (results are read as
        ``result[name]``). No ``.``; ``"continuum"`` and ``"frame"`` are
        reserved.
    rest, obs
        Nominal wavelength, rest frame or observed frame (exactly one). Rest
        wavelengths are shifted with the spectrum's ``z`` at ``prepare``.
    unit
        Unit of ``rest`` / ``obs``; converted to the spectrum's unit at
        ``prepare``.
    component
        ``"narrow"`` or ``"broad"``: only selects the default width prior and
        nominal width.
    contribution
        ``"emission"`` adds the line, ``"absorption"`` subtracts it.
    profile
        ``"gaussian"``, ``"lorentzian"`` or ``"exponential"``.

    Attributes
    ----------
    base : NoobLine or None
        The line this one was derived from.
    center_rule, fwhm_rule, flux_rule : LineRule
        The parameter rules.
    kernels : tuple of LineKernel
        Convolution kernels.
    """

    name: str
    rest: float | None
    obs: float | None
    unit: WaveUnit
    component: ComponentName
    contribution: ContributionName
    profile: ProfileName
    base: NoobLine | None
    center_rule: LineRule
    fwhm_rule: LineRule
    flux_rule: LineRule
    kernels: tuple[LineKernel, ...]

    def __init__(
        self,
        name: str,
        *,
        rest: float | None = None,
        obs: float | None = None,
        unit: WaveUnit,
        component: ComponentName = "narrow",
        contribution: ContributionName = "emission",
        profile: ProfileName = "gaussian",
    ) -> None:
        """Validate the declaration."""
        _set(self, "name", check_name(name))
        rest_value, obs_value = _wavelengths(rest, obs)
        _set(self, "rest", rest_value)
        _set(self, "obs", obs_value)
        _set(self, "unit", check_unit(unit))
        _set(self, "component", _choice(component, _COMPONENTS, "component"))
        _set(
            self, "contribution", _choice(contribution, _CONTRIBUTIONS, "contribution")
        )
        check_kernels(profile, ())
        _set(self, "profile", profile)
        _set(self, "base", None)
        _set(self, "center_rule", _DEFAULT)
        _set(self, "fwhm_rule", _DEFAULT)
        _set(self, "flux_rule", _DEFAULT)
        _set(self, "kernels", ())

    def derive(
        self,
        name: str,
        *,
        rest: float | None = None,
        obs: float | None = None,
        unit: WaveUnit | None = None,
        component: ComponentName | None = None,
        contribution: ContributionName | None = None,
        profile: ProfileName | None = None,
    ) -> NoobLine:
        """Create a line tied to this one as its base.

        Defaults: the centre moves with the base
        (``delta_v_from_base_kms=0``); the FWHM is ``ratio_to_base=1`` when
        the component matches and an independent default prior otherwise;
        the flux is independent. Kernels are inherited, each parameter locked
        to the base when the component matches and copied as an independent
        declaration otherwise. Change any of this with the parameter methods.

        Parameters
        ----------
        name
            The new line's name (required; names are never generated).
        rest, obs, unit
            New nominal wavelength; the base's when omitted.
        component, contribution, profile
            Overrides of the base's values.

        Returns
        -------
        NoobLine
            The derived line.
        """
        if rest is None and obs is None:
            if unit is not None:
                raise ValueError(
                    "derive(unit=...) needs a new rest= or obs= wavelength."
                )
            rest, obs, unit = self.rest, self.obs, self.unit
        derived_component = self.component if component is None else component
        line = NoobLine(
            name,
            rest=rest,
            obs=obs,
            unit=self.unit if unit is None else unit,
            component=derived_component,
            contribution=self.contribution if contribution is None else contribution,
            profile=self.profile if profile is None else profile,
        )
        same_component = derived_component == self.component
        kernels = tuple(
            LineKernel(kernel.kind, None, None)
            if same_component
            else LineKernel(kernel.kind, *self._effective_kernel(kernel.kind))
            for kernel in self.kernels
        )
        check_kernels(line.profile, [kernel.kind for kernel in kernels])
        fwhm_rule = LineRule("ratio_to_base", 1.0) if same_component else _DEFAULT
        return _evolve(
            line,
            base=self,
            center_rule=LineRule("delta_v_from_base_kms", 0.0),
            fwhm_rule=fwhm_rule,
            kernels=kernels,
        )

    def center(
        self,
        *,
        delta_v_kms: ParameterValue | None = None,
        delta_wavelength: ParameterValue | None = None,
        delta_v_from_base_kms: ParameterValue | None = None,
    ) -> NoobLine:
        """Set the line centre (exactly one argument).

        Parameters
        ----------
        delta_v_kms
            Velocity offset from the line's own nominal wavelength, km/s.
        delta_wavelength
            Observed-frame wavelength offset from the nominal wavelength, in
            the line's unit.
        delta_v_from_base_kms
            Velocity offset minus the base's velocity offset, km/s (derived
            lines only). ``0`` locks the two; a prior such as
            ``Uniform(high=0)`` orders them (this line bluer).

        Returns
        -------
        NoobLine
            A copy with the new centre rule.
        """
        given = {
            key: value
            for key, value in (
                ("delta_v_kms", delta_v_kms),
                ("delta_wavelength", delta_wavelength),
                ("delta_v_from_base_kms", delta_v_from_base_kms),
            )
            if value is not None
        }
        if len(given) != 1:
            raise ValueError(
                "center takes exactly one of delta_v_kms, delta_wavelength, "
                "delta_v_from_base_kms."
            )
        ((kind, value),) = given.items()
        if kind == "delta_v_from_base_kms":
            self._require_base(kind)
        rule = LineRule(kind, check_value(value, REAL, f"{self.name}.{kind}"))
        return _evolve(self, center_rule=rule)

    def fwhm(
        self,
        spec: ParameterValue | None = None,
        *,
        ratio_to_base: ParameterValue | None = None,
    ) -> NoobLine:
        """Set the intrinsic FWHM (exactly one argument).

        Parameters
        ----------
        spec
            The FWHM in km/s, fixed or a prior.
        ratio_to_base
            FWHM divided by the base's FWHM (derived lines only); ``1.0``
            locks them.

        Returns
        -------
        NoobLine
            A copy with the new width rule.
        """
        return _evolve(
            self, fwhm_rule=self._scale_rule("fwhm", spec, ratio_to_base, POSITIVE)
        )

    def flux(
        self,
        spec: ParameterValue | None = None,
        *,
        ratio_to_base: ParameterValue | None = None,
    ) -> NoobLine:
        """Set the integrated flux (exactly one argument).

        Parameters
        ----------
        spec
            The flux (non-negative; ``contribution`` sets the sign), fixed or
            a prior.
        ratio_to_base
            Flux divided by the base's flux (derived lines only), fixed or a
            prior.

        Returns
        -------
        NoobLine
            A copy with the new flux rule.
        """
        return _evolve(
            self, flux_rule=self._scale_rule("flux", spec, ratio_to_base, NONNEGATIVE)
        )

    def convolve(
        self,
        kind: str,
        fwhm: ParameterValue,
        *,
        fraction: ParameterValue = 1.0,
    ) -> NoobLine:
        """Convolve the line with a kernel.

        The profile becomes ``(1 - fraction) * line + fraction * (line *
        kernel)``. A line has at most one kernel of each kind: convolving
        with a kind it already has (for example one inherited by
        :meth:`derive`) replaces it with this independent declaration.

        Parameters
        ----------
        kind
            ``"gaussian"`` or ``"laplace"``.
        fwhm
            Kernel FWHM in km/s, fixed or a prior.
        fraction
            Convolved fraction in ``(0, 1]``, fixed or a prior.

        Returns
        -------
        NoobLine
            A copy with the kernel.

        Raises
        ------
        NotImplementedError
            If the base profile is not gaussian or a second non-gaussian
            kernel would be added (closed forms only).
        """
        if not isinstance(kind, str):
            raise TypeError(f"kernel kind must be one of {KERNELS} as a string.")
        kernel = LineKernel(
            kind,
            check_value(fwhm, POSITIVE, f"{self.name}.{kind}.fwhm"),
            check_value(fraction, FRACTION, f"{self.name}.{kind}.fraction"),
        )
        kernels = tuple(k for k in self.kernels if k.kind != kind) + (kernel,)
        check_kernels(self.profile, [k.kind for k in kernels])
        return _evolve(self, kernels=kernels)

    def __repr__(self) -> str:
        """Summarize the declaration."""
        where = f"rest={self.rest:g}" if self.rest is not None else f"obs={self.obs:g}"
        base = "" if self.base is None else f", base={self.base.name!r}"
        return (
            f"NoobLine({self.name!r}, {where} {self.unit}, {self.component}, "
            f"{self.contribution}, {self.profile}{base})"
        )

    def _scale_rule(
        self,
        quantity: str,
        spec: ParameterValue | None,
        ratio_to_base: ParameterValue | None,
        domain: object,
    ) -> LineRule:
        if (spec is None) == (ratio_to_base is None):
            raise ValueError(
                f"{quantity} takes exactly one of a value or ratio_to_base=."
            )
        if spec is not None:
            return LineRule(
                "independent", check_value(spec, domain, f"{self.name}.{quantity}")
            )
        self._require_base(f"{quantity}(ratio_to_base=...)")
        ratio_domain = POSITIVE if quantity == "fwhm" else NONNEGATIVE
        name = f"{self.name}.{quantity}_ratio_to_base"
        return LineRule("ratio_to_base", check_value(ratio_to_base, ratio_domain, name))

    def _require_base(self, what: str) -> None:
        if self.base is None:
            raise ValueError(f"{what} needs a derived line; {self.name!r} has no base.")

    def _effective_kernel(self, kind: str) -> tuple[ParameterValue, ParameterValue]:
        """Return the kernel declaration, following locks up the base chain."""
        line: NoobLine | None = self
        fwhm: ParameterValue | None = None
        fraction: ParameterValue | None = None
        while line is not None and (fwhm is None or fraction is None):
            kernel = next(k for k in line.kernels if k.kind == kind)
            fwhm = kernel.fwhm if fwhm is None else fwhm
            fraction = kernel.fraction if fraction is None else fraction
            line = line.base
        if fwhm is None or fraction is None:
            raise RuntimeError(
                f"kernel {kind!r} is locked all the way up the base chain."
            )
        return fwhm, fraction


def check_name(name: object) -> str:
    """Validate a component name.

    Raises
    ------
    ValueError
        If ``name`` is empty, contains ``.``, or is reserved.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"line name must be a non-empty string; got {name!r}.")
    if "." in name:
        raise ValueError(f"line name {name!r} must not contain '.'.")
    if name in RESERVED_NAMES:
        raise ValueError(f"line name {name!r} is reserved.")
    return name


def _wavelengths(rest: object, obs: object) -> tuple[float | None, float | None]:
    if (rest is None) == (obs is None):
        raise ValueError("give exactly one of rest= or obs=.")
    value = rest if rest is not None else obs
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"line wavelength must be a positive number; got {value!r}.")
    return (float(value), None) if rest is not None else (None, float(value))


def _choice(value: object, choices: tuple[str, ...], name: str) -> str:
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}; got {value!r}.")
    return value  # type: ignore[return-value]


def _set(line: NoobLine, name: str, value: object) -> None:
    object.__setattr__(line, name, value)


def _evolve(line: NoobLine, **changes: object) -> NoobLine:
    """Copy a line with some fields changed (bypassing the constructor)."""
    copy = object.__new__(NoobLine)
    for entry in fields(NoobLine):
        _set(copy, entry.name, changes.get(entry.name, getattr(line, entry.name)))
    return copy
