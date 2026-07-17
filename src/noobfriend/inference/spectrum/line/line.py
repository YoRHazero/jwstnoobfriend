"""NoobLine public contract."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from inspect import signature

from noobfriend.inference.spectrum.line.kernels import KERNEL_KINDS, LineKernel
from noobfriend.inference.spectrum.line.rules import (
    _ParameterRule,
    _required_float,
    _rule_from_fixed_or_bounded,
)
from noobfriend.inference.spectrum.line.types import (
    DEFAULT_CENTER_RANGE_KMS,
    DEFAULT_FWHM_RANGES,
    VALID_COMPONENTS,
    VALID_CONTRIBUTIONS,
    VALID_PROFILES,
    ComponentName,
    ContributionName,
    FixedOrBounded,
    ProfileName,
    WaveUnit,
)
from noobfriend.inference.spectrum.line.wavelength import Wavelength


@dataclass(frozen=True, slots=True)
class NoobLine:
    """A fitting line normalized to the observed frame.

    Parameters
    ----------
    linename
        Human-readable line name, such as ``"OIII5007"``.
    z
        Redshift used to convert ``rest`` to observed wavelength. When
        provided it must be strictly positive.
    rest
        Rest-frame wavelength in ``unit``.
    obs
        Observed-frame wavelength in ``unit``.
    unit
        Wavelength unit. Internally the line center is always represented in
        observed frame but the numeric unit is kept unchanged.
    component
        Width/kinematic component this line belongs to.
    contribution
        Whether this line contributes emission or absorption relative to the
        continuum.
    profile
        Profile family used by this line's component model.
    base
        Parent line this line is derived from. Derived lines default to a
        locked center against their base. FWHM is locked to the base only when
        the derived line shares the base's component; deriving a different
        component frees the FWHM to that component's default range.
    """

    linename: str | None = None
    z: float | None = None
    rest: float | None = None
    obs: float | None = None
    unit: WaveUnit = "angstrom"
    component: ComponentName = "narrow"
    contribution: ContributionName = "emission"
    profile: ProfileName = "gaussian"
    base: NoobLine | None = None
    kernels: tuple[LineKernel, ...] = ()
    _center_rule: _ParameterRule | None = field(default=None, repr=False, compare=False)
    _fwhm_rule: _ParameterRule | None = field(default=None, repr=False, compare=False)
    _flux_rule: _ParameterRule | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate and normalize constructor inputs."""
        if self.component not in VALID_COMPONENTS:
            raise ValueError(f"Unsupported component: {self.component!r}.")
        if self.contribution not in VALID_CONTRIBUTIONS:
            raise ValueError(f"Unsupported contribution: {self.contribution!r}.")
        if self.profile not in VALID_PROFILES:
            raise ValueError(f"Unsupported profile: {self.profile!r}.")
        if self.kernels:
            if self.profile != "gaussian":
                raise NotImplementedError(
                    "convolution kernels currently require a gaussian base profile."
                )
            names = [k.name for k in self.kernels]
            if len(set(names)) != len(names):
                raise ValueError("convolution kernel names must be unique per line.")
            non_gaussian = [k for k in self.kernels if k.kind != "gaussian"]
            if len(non_gaussian) > 1:
                raise NotImplementedError(
                    "at most one non-gaussian convolution kernel is supported."
                )

        wavelength = Wavelength.normalize(
            z=self.z, rest=self.rest, obs=self.obs, unit=self.unit
        )
        object.__setattr__(self, "z", wavelength.z)
        object.__setattr__(self, "rest", wavelength.rest)
        object.__setattr__(self, "obs", wavelength.obs)

        if self._center_rule is None:
            center_rule = (
                _ParameterRule.locked(self.base)
                if self.base is not None
                else _default_center_rule()
            )
            object.__setattr__(self, "_center_rule", center_rule)
        if self._fwhm_rule is None:
            # A derived line locks its FWHM to the base only when it shares the
            # base's component; deriving a different component (e.g. narrow ->
            # broad) frees the FWHM to that component's default range instead.
            if self.base is not None and self.base.component == self.component:
                fwhm_rule = _ParameterRule.locked(self.base)
            else:
                fwhm_rule = _default_fwhm_rule(self.component)
            object.__setattr__(self, "_fwhm_rule", fwhm_rule)
        if self._flux_rule is None:
            object.__setattr__(self, "_flux_rule", _ParameterRule.free())

    @property
    def observed_wavelength(self) -> float:
        """Observed-frame wavelength in ``unit``."""
        return Wavelength(
            z=self.z, rest=self.rest, obs=self.obs, unit=self.unit
        ).observed

    @property
    def center_rule(self) -> _ParameterRule:
        """Center rule for the line."""
        if self._center_rule is None:
            raise RuntimeError("NoobLine invariant violated: missing center rule.")
        return self._center_rule

    @property
    def fwhm_rule(self) -> _ParameterRule:
        """FWHM rule for the line."""
        if self._fwhm_rule is None:
            raise RuntimeError("NoobLine invariant violated: missing FWHM rule.")
        return self._fwhm_rule

    @property
    def flux_rule(self) -> _ParameterRule:
        """Flux rule for the line."""
        if self._flux_rule is None:
            raise RuntimeError("NoobLine invariant violated: missing flux rule.")
        return self._flux_rule

    @property
    def is_derived(self) -> bool:
        """Whether this line was derived from another line."""
        return self.base is not None

    def derive(
        self,
        linename: str | None = None,
        *,
        z: float | None = None,
        rest: float | None = None,
        obs: float | None = None,
        component: ComponentName | None = None,
        contribution: ContributionName | None = None,
        profile: ProfileName | None = None,
    ) -> NoobLine:
        """Create a line strongly bound to this line as its base.

        The derived line defaults to a **locked center** against the base. Its
        **FWHM locks to the base only when the component matches**; deriving a
        different component (e.g. a broad component off a narrow base) instead
        frees the FWHM to that component's default range. Flux defaults to
        free. Loosen or tighten any axis explicitly with :meth:`center`,
        :meth:`fwhm`, and :meth:`flux` — e.g. give a broad component its own
        velocity with ``center(delta_v_kms=...)``.
        """
        return NoobLine(
            linename=linename if linename is not None else self.linename,
            z=self.z if z is None else z,
            rest=self.rest if rest is None and obs is None else rest,
            obs=self.obs if rest is None and obs is None else obs,
            unit=self.unit,
            component=self.component if component is None else component,
            contribution=self.contribution if contribution is None else contribution,
            profile=self.profile if profile is None else profile,
            base=self,
            kernels=self.kernels,
        )

    def center(
        self,
        *,
        delta_v_kms: FixedOrBounded | None = None,
        delta_wavelength: FixedOrBounded | None = None,
    ) -> NoobLine:
        """Return a copy with an explicit center offset rule.

        ``delta_v_kms`` is a velocity offset in km/s. ``delta_wavelength`` is
        an observed-frame wavelength offset in ``unit``.
        """
        if (delta_v_kms is None) == (delta_wavelength is None):
            raise ValueError("Provide exactly one of delta_v_kms or delta_wavelength.")

        if delta_v_kms is not None:
            rule = _rule_from_fixed_or_bounded(
                delta_v_kms, "delta_v_kms", offset_unit="km/s"
            )
        else:
            rule = _rule_from_fixed_or_bounded(
                delta_wavelength, "delta_wavelength", offset_unit="wavelength"
            )
        return replace(self, _center_rule=rule)

    def fwhm(
        self,
        *,
        override: FixedOrBounded | None = None,
        locked: bool | NoobLine | None = None,
    ) -> NoobLine:
        """Return a copy with an explicit FWHM rule.

        ``override`` is interpreted as FWHM in km/s. ``locked=True`` locks to
        ``base``. Passing another ``NoobLine`` locks to that line explicitly.
        ``locked=False`` restores this line's component default range.
        """
        if (override is None) == (locked is None):
            raise ValueError("Provide exactly one of override or locked.")
        if override is not None:
            rule = _rule_from_fixed_or_bounded(override, "fwhm.override", positive=True)
            return replace(self, _fwhm_rule=rule)
        if isinstance(locked, NoobLine):
            return replace(self, _fwhm_rule=_ParameterRule.locked(locked))
        if locked is True:
            if self.base is None:
                raise ValueError("locked=True requires a derived line with base.")
            return replace(self, _fwhm_rule=_ParameterRule.locked(self.base))
        return replace(self, _fwhm_rule=_default_fwhm_rule(self.component))

    def flux(
        self, *, override: FixedOrBounded | None = None, ratio: float | None = None
    ) -> NoobLine:
        """Return a copy with an explicit flux rule.

        ``override`` is an absolute integrated flux. ``ratio`` is relative to
        ``base`` and therefore only valid on a derived line. ``ratio`` must be
        a nonnegative scalar constant.
        """
        if (override is None) == (ratio is None):
            raise ValueError("Provide exactly one of override or ratio.")
        if override is not None:
            rule = _rule_from_fixed_or_bounded(
                override, "flux.override", nonnegative=True
            )
            return replace(self, _flux_rule=rule)
        if self.base is None:
            raise ValueError("ratio requires a derived line with base.")
        value = _required_float(ratio, name="flux.ratio")
        if value < 0:
            raise ValueError("flux.ratio must be nonnegative.")
        return replace(self, _flux_rule=_ParameterRule.ratio(value))

    def convolve(self, kernel: object, **shape_rules: FixedOrBounded) -> NoobLine:
        """Return a copy convolved with a built-in kernel.

        The kernel is one of the functions in
        :mod:`noobfriend.inference.spectrum.line.kernels`, defined in
        zero-centred velocity space (km/s). Its signature after the first
        (``x``) argument names the kernel's shape parameters; each must be
        given here as a fixed value or ``(lower, upper)`` bounds in km/s.

        Parameters
        ----------
        kernel
            A registered kernel function, e.g. ``kernels.laplace``.
        **shape_rules
            One fixed-or-bounded rule per kernel shape parameter. The
            reserved ``fraction`` key (default: fixed ``1.0``) sets the
            convolved fraction: the profile becomes
            ``(1 - fraction) * base + fraction * base (x) kernel``. Fixed
            values and bounds must lie in ``(0, 1]``.

        Returns
        -------
        NoobLine
            A copy with the kernel appended to :attr:`kernels`.
        """
        fraction = shape_rules.pop("fraction", 1.0)
        kind = KERNEL_KINDS.get(kernel)
        if kind is None:
            raise TypeError(
                "convolve expects a built-in kernel from "
                "noobfriend.inference.spectrum.line.kernels; arbitrary "
                "functions require the numerical convolution backend, "
                "which is not implemented yet."
            )
        parameters = tuple(signature(kernel).parameters)[1:]
        unknown = set(shape_rules) - set(parameters)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown kernel shape parameters: {names}.")
        missing = [name for name in parameters if name not in shape_rules]
        if missing:
            names = ", ".join(missing)
            raise ValueError(f"kernel shape parameters missing rules: {names}.")
        rules = tuple(
            (
                name,
                _rule_from_fixed_or_bounded(
                    shape_rules[name], f"convolve.{name}", positive=True
                ),
            )
            for name in parameters
        )
        fraction_rule = _rule_from_fixed_or_bounded(
            fraction, "convolve.fraction", positive=True
        )
        if fraction_rule.is_fixed:
            if fraction_rule.value is None or not 0.0 < fraction_rule.value <= 1.0:
                raise ValueError("convolve.fraction must lie in (0, 1].")
        elif fraction_rule.bounds is not None:
            lower, upper = fraction_rule.bounds
            if not 0.0 < lower < upper <= 1.0:
                raise ValueError("convolve.fraction bounds must lie in (0, 1].")
        appended = LineKernel(
            kind=kind, name=kind, rules=(*rules, ("fraction", fraction_rule))
        )
        return replace(self, kernels=(*self.kernels, appended))


def _default_center_rule() -> _ParameterRule:
    return _ParameterRule.bounded(DEFAULT_CENTER_RANGE_KMS, offset_unit="km/s")


def _default_fwhm_rule(component: ComponentName) -> _ParameterRule:
    return _ParameterRule.bounded(DEFAULT_FWHM_RANGES[component])
