"""Declarative priors for spectrum-fit parameters.

Every parameter of a line, kernel, continuum or noise model takes either a
``float`` (fixed) or one of these priors (free). They are plain value objects:
how a prior becomes an optimizer bound or a sampling site is up to the fit
backend.

An omitted ``Uniform`` / ``LogUniform`` / ``TruncatedNormal`` bound means the
edge of the parameter's own domain (a width's ``0``, a velocity's ``inf``). A
uniform prior with an infinite end is improper. Bounds are filled in by
:meth:`Prior.resolve`, which ``prepare`` calls with each parameter's domain.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import exp, inf, isfinite, log, log1p, pi
from numbers import Real
from typing import Any

from noobfriend.inference.spectrum.priors._domain import Domain

_HALF_LOG_TWO_PI = 0.5 * log(2.0 * pi)


@dataclass(frozen=True, slots=True)
class Prior:
    """Base class for declarative parameter priors."""

    def resolve(self, domain: Domain) -> Prior:
        """Return a copy with omitted bounds filled from ``domain``.

        Parameters
        ----------
        domain
            The parameter's domain.

        Returns
        -------
        Prior
            The same prior with every bound explicit.

        Raises
        ------
        ValueError
            If the prior puts mass outside ``domain``.
        """
        raise NotImplementedError

    def bounds(self) -> tuple[float, float]:
        """Return the support ``(low, high)`` of a resolved prior."""
        raise NotImplementedError

    def scaled(self, factor: float) -> Prior:
        """Return the prior of ``factor * x`` when ``x`` has this prior.

        Parameters
        ----------
        factor
            Positive multiplier (a unit conversion or a sampling scale).

        Returns
        -------
        Prior
            The same family with location, scale and bounds multiplied.
        """
        if not factor > 0.0:
            raise ValueError(f"scale factor must be positive; got {factor!r}.")
        changes = {
            name: getattr(self, name) * factor
            for name in ("mu", "sigma", "low", "high")
            if hasattr(self, name) and getattr(self, name) is not None
        }
        return replace(self, **changes)

    @property
    def is_proper(self) -> bool:
        """Whether the density integrates to one."""
        low, high = self.bounds()
        return isfinite(low) and isfinite(high)

    def log_density(self, xp: Any, value: Any) -> Any:
        """Evaluate the log density of a resolved prior at ``value``.

        Values outside the support are not checked; samplers keep them inside.
        An improper prior returns its unnormalized log density.
        """
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Normal(Prior):
    """Normal prior over the whole real line.

    Only valid on parameters whose domain is the real line (velocity offsets,
    continuum coefficients, power-law indices).

    Parameters
    ----------
    mu
        Mean.
    sigma
        Standard deviation, positive.
    """

    mu: float
    sigma: float

    def __post_init__(self) -> None:
        """Validate and coerce the parameters."""
        _set(self, "mu", _finite(self.mu, "Normal.mu"))
        _set(self, "sigma", _positive(self.sigma, "Normal.sigma"))

    def resolve(self, domain: Domain) -> Normal:
        """Return ``self`` after checking the domain is the real line."""
        if isfinite(domain.low) or isfinite(domain.high):
            raise ValueError(
                f"a Normal prior covers the whole real line, but this parameter "
                f"is {domain.name} {domain}; use TruncatedNormal(mu, sigma, "
                "low=..., high=...) instead."
            )
        return self

    def bounds(self) -> tuple[float, float]:
        """Return ``(-inf, inf)``."""
        return -inf, inf

    def log_density(self, xp: Any, value: Any) -> Any:
        """Evaluate the normal log density."""
        z = (value - self.mu) / self.sigma
        return -0.5 * z * z - log(self.sigma) - _HALF_LOG_TWO_PI

    def __str__(self) -> str:
        """Compact form, e.g. ``Normal(0, 5)``."""
        return f"Normal({self.mu:.4g}, {self.sigma:.4g})"


@dataclass(frozen=True, slots=True)
class TruncatedNormal(Prior):
    """Normal prior restricted to ``[low, high]``.

    Parameters
    ----------
    mu, sigma
        Location and scale of the untruncated normal; ``mu`` may lie outside
        the bounds.
    low, high
        Truncation bounds. An omitted bound is the parameter's domain edge, so
        ``TruncatedNormal(0, s, low=0)`` is a half-normal.
    """

    mu: float
    sigma: float
    low: float | None = None
    high: float | None = None

    def __post_init__(self) -> None:
        """Validate and coerce the parameters."""
        _set(self, "mu", _finite(self.mu, "TruncatedNormal.mu"))
        _set(self, "sigma", _positive(self.sigma, "TruncatedNormal.sigma"))
        _bounds(self, "TruncatedNormal", allow_infinite=True)

    def resolve(self, domain: Domain) -> TruncatedNormal:
        """Fill omitted bounds from ``domain``."""
        return _resolve_bounds(self, domain, "TruncatedNormal")

    def bounds(self) -> tuple[float, float]:
        """Return the truncation interval."""
        return _require_bounds(self)

    @property
    def is_proper(self) -> bool:
        """Always ``True``."""
        return True

    def log_density(self, xp: Any, value: Any) -> Any:
        """Truncated-normal log density inside the bounds."""
        low, high = self.bounds()
        z = (value - self.mu) / self.sigma
        log_mass = _log_normal_mass(
            (low - self.mu) / self.sigma, (high - self.mu) / self.sigma
        )
        return -0.5 * z * z - log(self.sigma) - _HALF_LOG_TWO_PI - log_mass

    def __str__(self) -> str:
        """Compact form, e.g. ``TruncatedNormal(0, 3, low=0)``."""
        return f"TruncatedNormal({self.mu:.4g}, {self.sigma:.4g}{_bound_text(self)})"


@dataclass(frozen=True, slots=True)
class Uniform(Prior):
    """Prior flat in the parameter itself.

    Parameters
    ----------
    low, high
        Bounds. An omitted bound is the parameter's domain edge; ``Uniform()``
        on an unbounded parameter is an improper flat prior.
    """

    low: float | None = None
    high: float | None = None

    def __post_init__(self) -> None:
        """Validate and coerce the bounds."""
        _bounds(self, "Uniform", allow_infinite=True)

    def resolve(self, domain: Domain) -> Uniform:
        """Fill omitted bounds from ``domain``."""
        return _resolve_bounds(self, domain, "Uniform")

    def bounds(self) -> tuple[float, float]:
        """Return the support."""
        return _require_bounds(self)

    def log_density(self, xp: Any, value: Any) -> Any:
        """Constant log density (zero when improper)."""
        low, high = self.bounds()
        constant = -log(high - low) if isfinite(low) and isfinite(high) else 0.0
        return 0.0 * value + constant

    def __str__(self) -> str:
        """Compact form, e.g. ``Uniform(-300, 300)``."""
        return f"Uniform({_bound_pair(self)})"


@dataclass(frozen=True, slots=True)
class LogUniform(Prior):
    """Prior flat in the logarithm of a positive parameter.

    Parameters
    ----------
    low, high
        Nonnegative bounds. An omitted bound is the parameter's domain edge;
        ``LogUniform()`` on a width is the improper scale-invariant prior, and
        a lower bound of ``0`` is improper too.
    """

    low: float | None = None
    high: float | None = None

    def __post_init__(self) -> None:
        """Validate and coerce the bounds."""
        _bounds(self, "LogUniform", allow_infinite=True)
        if self.low is not None and self.low < 0.0:
            raise ValueError("LogUniform.low must be nonnegative.")

    def resolve(self, domain: Domain) -> LogUniform:
        """Fill omitted bounds from ``domain``; the domain must be positive."""
        if domain.low < 0.0:
            raise ValueError(
                f"a LogUniform prior needs a positive parameter, but this "
                f"parameter is {domain.name} {domain}."
            )
        return _resolve_bounds(self, domain, "LogUniform")

    def bounds(self) -> tuple[float, float]:
        """Return the support."""
        return _require_bounds(self)

    @property
    def is_proper(self) -> bool:
        """Whether both bounds are finite and the lower one positive."""
        low, high = self.bounds()
        return low > 0.0 and isfinite(high)

    def log_density(self, xp: Any, value: Any) -> Any:
        """``-log(value)`` minus the normalizer when proper."""
        low, high = self.bounds()
        proper = low > 0.0 and isfinite(high)
        constant = -log(log(high / low)) if proper else 0.0
        return -xp.log(value) + constant

    def __str__(self) -> str:
        """Compact form, e.g. ``LogUniform(10, 700)``."""
        return f"LogUniform({_bound_pair(self)})"


type ParameterValue = float | Prior


def check_value(value: object, domain: Domain, name: str) -> ParameterValue:
    """Validate a parameter value: a fixed ``float`` or a ``Prior``.

    Parameters
    ----------
    value
        The user's declaration.
    domain
        The parameter's domain.
    name
        Parameter name used in error messages.

    Returns
    -------
    float or Prior
        A fixed value coerced to ``float``, or the prior unchanged (it is
        resolved against ``domain`` once here only to fail early).

    Raises
    ------
    TypeError
        If ``value`` is neither a real number nor a prior.
    ValueError
        If a fixed value lies outside ``domain`` or a prior does not fit it.
    """
    if isinstance(value, Prior):
        value.resolve(domain)
        return value
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        if not domain.contains(number):
            raise ValueError(f"{name} must lie in {domain}; got {number!r}.")
        return number
    raise TypeError(
        f"{name} must be a float (fixed) or a prior such as Uniform(...); "
        f"got {type(value).__name__}."
    )


def _resolve_bounds[P: Prior](prior: P, domain: Domain, label: str) -> P:
    low = domain.low if prior.low is None else prior.low  # type: ignore[attr-defined]
    high = domain.high if prior.high is None else prior.high  # type: ignore[attr-defined]
    if not domain.contains_interval(low, high):
        raise ValueError(
            f"{label} bounds ({low:g}, {high:g}) leave this {domain.name} "
            f"parameter's domain {domain}."
        )
    if low >= high:
        raise ValueError(f"{label} bounds must be increasing; got ({low:g}, {high:g}).")
    return replace(prior, low=low, high=high)  # type: ignore[call-arg]


def _require_bounds(prior: Any) -> tuple[float, float]:
    if prior.low is None or prior.high is None:
        raise RuntimeError(
            f"{type(prior).__name__} has omitted bounds; call resolve(domain) first."
        )
    return prior.low, prior.high


def _bounds(prior: Any, label: str, *, allow_infinite: bool) -> None:
    for end in ("low", "high"):
        value = getattr(prior, end)
        if value is None:
            continue
        if not isinstance(value, Real) or isinstance(value, bool):
            raise TypeError(f"{label}.{end} must be a number.")
        number = float(value)
        if number != number or (not allow_infinite and not isfinite(number)):
            raise ValueError(f"{label}.{end} must be finite.")
        _set(prior, end, number)
    if prior.low is not None and prior.high is not None and prior.low >= prior.high:
        raise ValueError(f"{label} bounds must be increasing.")


def _finite(value: object, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number.")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _positive(value: object, name: str) -> float:
    number = _finite(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return number


def _set(prior: Prior, name: str, value: float) -> None:
    object.__setattr__(prior, name, value)


def _log_normal_mass(alpha: float, beta: float) -> float:
    """Return ``log(Phi(beta) - Phi(alpha))`` stably for ``alpha < beta``."""
    if alpha == -inf and beta == inf:
        return 0.0
    from scipy.special import log_ndtr

    if alpha > 0.0:
        alpha, beta = -beta, -alpha
    upper = float(log_ndtr(beta))
    lower = float(log_ndtr(alpha))
    return upper + log1p(-exp(lower - upper))


def _bound_pair(prior: Any) -> str:
    low, high = _shown_bounds(prior)
    if low is None and high is None:
        return ""
    if high is None:
        return f"low={low:.4g}"
    if low is None:
        return f"high={high:.4g}"
    return f"{low:.4g}, {high:.4g}"


def _bound_text(prior: Any) -> str:
    low, high = _shown_bounds(prior)
    parts = []
    if low is not None:
        parts.append(f"low={low:.4g}")
    if high is not None:
        parts.append(f"high={high:.4g}")
    return "".join(f", {part}" for part in parts)


def _shown_bounds(prior: Any) -> tuple[float | None, float | None]:
    """Bounds worth printing: omitted and infinite ends are left out."""
    low = prior.low if prior.low is not None and isfinite(prior.low) else None
    high = prior.high if prior.high is not None and isfinite(prior.high) else None
    return low, high
