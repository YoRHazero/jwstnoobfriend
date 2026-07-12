"""NoobLine public contract."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from noobfriend.inference.spectrum.line.rules import _ParameterRule, _required_float, _rule_from_fixed_or_bounded
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
        Parent line this line is derived from. Derived lines default to locked
        center and FWHM rules against their base.
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

        wavelength = Wavelength.normalize(z=self.z, rest=self.rest, obs=self.obs, unit=self.unit)
        object.__setattr__(self, "z", wavelength.z)
        object.__setattr__(self, "rest", wavelength.rest)
        object.__setattr__(self, "obs", wavelength.obs)

        if self._center_rule is None:
            center_rule = _ParameterRule.locked(self.base) if self.base is not None else _default_center_rule()
            object.__setattr__(self, "_center_rule", center_rule)
        if self._fwhm_rule is None:
            fwhm_rule = _ParameterRule.locked(self.base) if self.base is not None else _default_fwhm_rule(self.component)
            object.__setattr__(self, "_fwhm_rule", fwhm_rule)
        if self._flux_rule is None:
            object.__setattr__(self, "_flux_rule", _ParameterRule.free())

    @property
    def observed_wavelength(self) -> float:
        """Observed-frame wavelength in ``unit``."""
        return Wavelength(z=self.z, rest=self.rest, obs=self.obs, unit=self.unit).observed

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
        """Create a line that is strongly bound to this line as its base."""
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
            rule = _rule_from_fixed_or_bounded(delta_v_kms, "delta_v_kms", offset_unit="km/s")
        else:
            rule = _rule_from_fixed_or_bounded(delta_wavelength, "delta_wavelength", offset_unit="wavelength")
        return replace(self, _center_rule=rule)

    def fwhm(self, *, override: FixedOrBounded | None = None, locked: bool | NoobLine | None = None) -> NoobLine:
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

    def flux(self, *, override: FixedOrBounded | None = None, ratio: float | None = None) -> NoobLine:
        """Return a copy with an explicit flux rule.

        ``override`` is an absolute integrated flux. ``ratio`` is relative to
        ``base`` and therefore only valid on a derived line. ``ratio`` must be
        a nonnegative scalar constant.
        """
        if (override is None) == (ratio is None):
            raise ValueError("Provide exactly one of override or ratio.")
        if override is not None:
            rule = _rule_from_fixed_or_bounded(override, "flux.override", nonnegative=True)
            return replace(self, _flux_rule=rule)
        if self.base is None:
            raise ValueError("ratio requires a derived line with base.")
        value = _required_float(ratio, name="flux.ratio")
        if value < 0:
            raise ValueError("flux.ratio must be nonnegative.")
        return replace(self, _flux_rule=_ParameterRule.ratio(value))


def _default_center_rule() -> _ParameterRule:
    return _ParameterRule.bounded(DEFAULT_CENTER_RANGE_KMS, offset_unit="km/s")


def _default_fwhm_rule(component: ComponentName) -> _ParameterRule:
    return _ParameterRule.bounded(DEFAULT_FWHM_RANGES[component])
